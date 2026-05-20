"""confirm_booking tool — two-step in-chat booking completion.

Step 1 (preview): User selects a service from suggestions → price preview shown.
Step 2 (confirm): User confirms → booking created via core-service.

The session state ``pending_booking`` key tracks which step we're on.
"""
from __future__ import annotations

from typing import Any

from src.auth.jwt import AuthenticatedUser
from src.core.core_service import CoreServiceClient, CoreServiceError
from src.core.logging import get_logger

logger = get_logger(__name__)


async def execute_confirm_booking(
    *,
    core_service: CoreServiceClient,
    current_user: AuthenticatedUser | None,
    service_id: str,
    service_type: str,
    tier_id: str | None = None,
    service_name: str | None = None,
    pending_booking: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the confirm_booking tool.

    If *pending_booking* is ``None`` we are in the **preview** step:
      → call ``preview_home_puja_price`` and return a ``booking_price_preview`` event.

    If *pending_booking* is present (i.e. user already saw the preview and
    said "yes"), we are in the **confirm** step:
      → call ``create_home_puja_booking`` / ``create_temple_booking``
        and return a ``booking_confirmed`` event.

    Returns a tool-output dict that the chat service can forward as a
    structured event to the frontend.
    """
    if current_user is None:
        return _error_output("You need to be logged in to book a service. Please sign in and try again.")

    # ── Step 2: Confirm (pending_booking already exists) ─────────────
    if pending_booking is not None:
        return await _create_booking(
            core_service=core_service,
            current_user=current_user,
            pending_booking=pending_booking,
        )

    # ── Step 1: Preview ──────────────────────────────────────────────
    if service_type == "temple":
        # Temple bookings don't have a separate price preview in core-service.
        # Go straight to a confirmation prompt with known price.
        return {
            "tool": "confirm_booking",
            "event_name": "booking_price_preview",
            "service_id": service_id,
            "service_type": service_type,
            "tier_id": tier_id,
            "service_name": service_name or "Temple Puja",
            "summary": (
                f"You're about to book **{service_name or 'a temple puja'}**. "
                "Please confirm to proceed with the booking."
            ),
            "needs_confirmation": True,
            "pending_booking": {
                "service_id": service_id,
                "service_type": service_type,
                "tier_id": tier_id,
                "service_name": service_name,
            },
        }

    # Home puja: try to get a price preview
    preview_payload: dict[str, Any] = {}
    if tier_id:
        preview_payload["service_tier_id"] = tier_id
    else:
        preview_payload["service_id"] = service_id

    try:
        preview = await core_service.preview_home_puja_price(
            preview_payload, current_user,
        )
    except CoreServiceError as exc:
        logger.warning("booking_price_preview_failed | error=%s", exc)
        preview = None

    price_info: dict[str, Any] = {}
    if preview:
        price_info = {
            "base_price": preview.get("base_price"),
            "discount": preview.get("discount"),
            "total": preview.get("total") or preview.get("final_amount"),
            "currency": preview.get("currency", "INR"),
            "breakdown": preview.get("breakdown"),
        }

    return {
        "tool": "confirm_booking",
        "event_name": "booking_price_preview",
        "service_id": service_id,
        "service_type": service_type,
        "tier_id": tier_id,
        "service_name": service_name or "Home Puja",
        "price": price_info if price_info.get("total") else None,
        "summary": _preview_summary(service_name, price_info),
        "needs_confirmation": True,
        "pending_booking": {
            "service_id": service_id,
            "service_type": service_type,
            "tier_id": tier_id,
            "service_name": service_name,
            "price": price_info if price_info.get("total") else None,
        },
    }


async def _create_booking(
    *,
    core_service: CoreServiceClient,
    current_user: AuthenticatedUser,
    pending_booking: dict[str, Any],
) -> dict[str, Any]:
    """Create the actual booking via core-service."""
    service_type = pending_booking.get("service_type", "home_puja")
    service_id = pending_booking.get("service_id")
    tier_id = pending_booking.get("tier_id")
    service_name = pending_booking.get("service_name", "Puja")

    payload: dict[str, Any] = {}
    if tier_id:
        payload["service_tier_id"] = tier_id
    elif service_id:
        payload["service_id"] = service_id

    try:
        if service_type == "temple":
            result = await core_service.create_temple_booking(payload, current_user)
        else:
            result = await core_service.create_home_puja_booking(payload, current_user)
    except CoreServiceError as exc:
        logger.warning("booking_creation_failed | error=%s", exc)
        return _error_output(
            f"Could not complete your booking for {service_name}. "
            f"Error: {exc}. Please try again or contact support."
        )

    if result is None:
        return _error_output(
            f"The booking service is temporarily unavailable. "
            f"Please try again in a few minutes."
        )

    booking_id = result.get("id") or result.get("booking_id")
    return {
        "tool": "confirm_booking",
        "event_name": "booking_confirmed",
        "booking_id": str(booking_id) if booking_id else None,
        "service_name": service_name,
        "service_type": service_type,
        "booking_details": result,
        "summary": (
            f"Your booking for **{service_name}** has been confirmed! "
            f"Booking ID: {booking_id}. "
            "You'll receive a confirmation with all the details shortly."
        ),
        "needs_confirmation": False,
        "pending_booking": None,  # Clear the pending state
    }


def _preview_summary(service_name: str | None, price_info: dict[str, Any]) -> str:
    name = service_name or "the puja service"
    total = price_info.get("total")
    currency = price_info.get("currency", "INR")
    if total:
        return (
            f"You're about to book **{name}** for **{currency} {total}**. "
            "Please confirm to proceed with the booking."
        )
    return (
        f"You're about to book **{name}**. "
        "Please confirm to proceed with the booking."
    )


def _error_output(message: str) -> dict[str, Any]:
    return {
        "tool": "confirm_booking",
        "event_name": "booking_failed",
        "summary": message,
        "needs_confirmation": False,
        "pending_booking": None,
    }
