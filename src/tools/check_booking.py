"""check_booking tool — view booking status and history in-chat."""
from __future__ import annotations

from typing import Any

from src.auth.jwt import AuthenticatedUser
from src.core.core_service import CoreServiceClient
from src.core.logging import get_logger

logger = get_logger(__name__)


async def execute_check_booking(
    *,
    core_service: CoreServiceClient,
    current_user: AuthenticatedUser | None,
    status_filter: str | None = None,
) -> dict[str, Any]:
    """Fetch and return the user's booking history.

    Returns a tool-output dict with event_name ``booking_history``.
    """
    if current_user is None:
        return {
            "tool": "check_booking",
            "event_name": "booking_history",
            "summary": "You need to be logged in to view your bookings. Please sign in and try again.",
            "bookings": [],
        }

    bookings = await core_service.list_user_bookings(
        current_user,
        status_filter=status_filter,
        limit=10,
    )

    if not bookings:
        return {
            "tool": "check_booking",
            "event_name": "booking_history",
            "summary": "You don't have any bookings yet.",
            "bookings": [],
        }

    items = [
        {
            "id": str(b.get("id", "")),
            "service_name": b.get("service_name") or b.get("name") or "Service",
            "status": b.get("status", "unknown"),
            "scheduled_for": b.get("scheduled_for"),
            "created_at": b.get("created_at"),
            "type": b.get("type") or b.get("service_type", "puja"),
        }
        for b in bookings[:10]
    ]

    summary_parts = []
    for item in items:
        line = f"- **{item['service_name']}** — {item['status']}"
        if item.get("scheduled_for"):
            line += f" (scheduled: {item['scheduled_for']})"
        summary_parts.append(line)

    return {
        "tool": "check_booking",
        "event_name": "booking_history",
        "summary": "Here are your recent bookings:\n" + "\n".join(summary_parts),
        "bookings": items,
        "total": len(items),
    }
