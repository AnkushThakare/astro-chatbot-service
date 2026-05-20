"""schedule_consultation tool — book a 1:1 video call with a pandit/astrologer.

Flow:
1. User selects a consultant from suggest_consultant suggestions.
2. Planner triggers schedule_consultation with consultant details.
3. Tool creates a consultation booking via core-service.
"""
from __future__ import annotations

from typing import Any

from src.auth.jwt import AuthenticatedUser
from src.core.core_service import CoreServiceClient, CoreServiceError
from src.core.logging import get_logger

logger = get_logger(__name__)


async def execute_schedule_consultation(
    *,
    core_service: CoreServiceClient,
    current_user: AuthenticatedUser | None,
    consultant_id: str,
    consultant_name: str | None = None,
    preferred_date: str | None = None,
    preferred_time: str | None = None,
    concern: str | None = None,
) -> dict[str, Any]:
    """Book a consultation with a pandit/astrologer.

    Returns a tool-output dict forwarded as a structured event to the frontend.
    """
    if current_user is None:
        return _error_output(
            "You need to be logged in to book a consultation. Please sign in and try again."
        )

    payload: dict[str, Any] = {
        "consultant_id": consultant_id,
    }
    if preferred_date:
        payload["preferred_date"] = preferred_date
    if preferred_time:
        payload["preferred_time"] = preferred_time
    if concern:
        payload["concern"] = concern

    try:
        result = await core_service.create_home_puja_booking(payload, current_user)
    except CoreServiceError as exc:
        logger.warning("consultation_booking_failed | error=%s", exc)
        return _error_output(
            f"Could not book your consultation with {consultant_name or 'the astrologer'}. "
            f"Error: {exc}. Please try again or contact support."
        )

    if result is None:
        return _error_output(
            "The consultation booking service is temporarily unavailable. "
            "Please try again in a few minutes."
        )

    booking_id = result.get("id") or result.get("booking_id")
    name = consultant_name or "the astrologer"
    return {
        "tool": "schedule_consultation",
        "event_name": "consultation_booked",
        "booking_id": str(booking_id) if booking_id else None,
        "consultant_name": consultant_name,
        "consultant_id": consultant_id,
        "booking_details": result,
        "summary": (
            f"Your consultation with **{name}** has been booked! "
            f"Booking ID: {booking_id}. "
            "You'll receive a confirmation with the meeting link and details shortly."
        ),
    }


def _error_output(message: str) -> dict[str, Any]:
    return {
        "tool": "schedule_consultation",
        "event_name": "consultation_booking_failed",
        "summary": message,
    }
