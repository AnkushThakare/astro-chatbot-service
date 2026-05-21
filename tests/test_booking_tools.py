"""Tests for in-chat booking confirmation and consultation scheduling tools."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from src.core.core_service import CoreServiceError


# ── confirm_booking tool tests ──────────────────────────────────────


def test_confirm_booking_preview_step() -> None:
    """First call (no pending_booking) returns a price preview event."""
    from src.tools.confirm_booking import execute_confirm_booking

    mock_core = AsyncMock()
    mock_core.preview_home_puja_price.return_value = {
        "base_price": 2100,
        "discount": 0,
        "total": 2100,
        "currency": "INR",
    }

    result = asyncio.run(
        execute_confirm_booking(
            core_service=mock_core,
            current_user=SimpleNamespace(user_id="test-user", token="tok"),
            service_id="svc-123",
            service_type="home_puja",
            tier_id="tier-1",
            service_name="Shani Puja",
            pending_booking=None,
        )
    )

    assert result["event_name"] == "booking_price_preview"
    assert result["needs_confirmation"] is True
    assert result["pending_booking"] is not None
    assert result["pending_booking"]["service_id"] == "svc-123"
    assert "2100" in result["summary"]
    mock_core.preview_home_puja_price.assert_called_once()


def test_confirm_booking_confirm_step() -> None:
    """Second call (with pending_booking) creates the actual booking."""
    from src.tools.confirm_booking import execute_confirm_booking

    mock_core = AsyncMock()
    mock_core.create_home_puja_booking.return_value = {
        "id": "booking-456",
        "status": "confirmed",
    }

    pending = {
        "service_id": "svc-123",
        "service_type": "home_puja",
        "tier_id": "tier-1",
        "service_name": "Shani Puja",
    }

    result = asyncio.run(
        execute_confirm_booking(
            core_service=mock_core,
            current_user=SimpleNamespace(user_id="test-user", token="tok"),
            service_id="svc-123",
            service_type="home_puja",
            pending_booking=pending,
        )
    )

    assert result["event_name"] == "booking_confirmed"
    assert result["booking_id"] == "booking-456"
    assert result["pending_booking"] is None  # Cleared after confirmation
    mock_core.create_home_puja_booking.assert_called_once()


def test_confirm_booking_temple_skips_preview() -> None:
    """Temple bookings skip the price preview and go to confirmation prompt."""
    from src.tools.confirm_booking import execute_confirm_booking

    mock_core = AsyncMock()

    result = asyncio.run(
        execute_confirm_booking(
            core_service=mock_core,
            current_user=SimpleNamespace(user_id="test-user", token="tok"),
            service_id="temple-789",
            service_type="temple",
            service_name="Kashi Vishwanath Puja",
            pending_booking=None,
        )
    )

    assert result["event_name"] == "booking_price_preview"
    assert result["needs_confirmation"] is True
    assert "Kashi Vishwanath Puja" in result["summary"]
    # No price preview API called for temple
    mock_core.preview_home_puja_price.assert_not_called()


def test_confirm_booking_temple_confirm_step() -> None:
    """Temple booking confirmation creates via create_temple_booking."""
    from src.tools.confirm_booking import execute_confirm_booking

    mock_core = AsyncMock()
    mock_core.create_temple_booking.return_value = {
        "id": "temple-booking-111",
        "status": "confirmed",
    }

    pending = {
        "service_id": "temple-789",
        "service_type": "temple",
        "tier_id": None,
        "service_name": "Kashi Vishwanath Puja",
    }

    result = asyncio.run(
        execute_confirm_booking(
            core_service=mock_core,
            current_user=SimpleNamespace(user_id="test-user", token="tok"),
            service_id="temple-789",
            service_type="temple",
            pending_booking=pending,
        )
    )

    assert result["event_name"] == "booking_confirmed"
    assert result["booking_id"] == "temple-booking-111"
    mock_core.create_temple_booking.assert_called_once()


def test_confirm_booking_requires_auth() -> None:
    """Returns error when no user is authenticated."""
    from src.tools.confirm_booking import execute_confirm_booking

    result = asyncio.run(
        execute_confirm_booking(
            core_service=AsyncMock(),
            current_user=None,
            service_id="svc-123",
            service_type="home_puja",
        )
    )

    assert result["event_name"] == "booking_failed"
    assert "logged in" in result["summary"].lower()


def test_confirm_booking_handles_core_service_error() -> None:
    """Gracefully handles core-service failures."""
    from src.tools.confirm_booking import execute_confirm_booking

    mock_core = AsyncMock()
    mock_core.create_home_puja_booking.side_effect = CoreServiceError("Service unavailable")

    pending = {
        "service_id": "svc-123",
        "service_type": "home_puja",
        "service_name": "Shani Puja",
    }

    result = asyncio.run(
        execute_confirm_booking(
            core_service=mock_core,
            current_user=SimpleNamespace(user_id="test-user", token="tok"),
            service_id="svc-123",
            service_type="home_puja",
            pending_booking=pending,
        )
    )

    assert result["event_name"] == "booking_failed"
    assert "Service unavailable" in result["summary"]


def test_confirm_booking_handles_none_response() -> None:
    """Handles when core-service returns None (transient failure)."""
    from src.tools.confirm_booking import execute_confirm_booking

    mock_core = AsyncMock()
    mock_core.create_home_puja_booking.return_value = None

    pending = {
        "service_id": "svc-123",
        "service_type": "home_puja",
        "service_name": "Shani Puja",
    }

    result = asyncio.run(
        execute_confirm_booking(
            core_service=mock_core,
            current_user=SimpleNamespace(user_id="test-user", token="tok"),
            service_id="svc-123",
            service_type="home_puja",
            pending_booking=pending,
        )
    )

    assert result["event_name"] == "booking_failed"
    assert "temporarily unavailable" in result["summary"].lower()


def test_confirm_booking_preview_without_price() -> None:
    """Preview step works even when price preview API fails."""
    from src.tools.confirm_booking import execute_confirm_booking

    mock_core = AsyncMock()
    mock_core.preview_home_puja_price.return_value = None

    result = asyncio.run(
        execute_confirm_booking(
            core_service=mock_core,
            current_user=SimpleNamespace(user_id="test-user", token="tok"),
            service_id="svc-123",
            service_type="home_puja",
            service_name="Navgraha Homam",
            pending_booking=None,
        )
    )

    assert result["event_name"] == "booking_price_preview"
    assert result["price"] is None
    assert "Navgraha Homam" in result["summary"]
    assert result["needs_confirmation"] is True


# ── schedule_consultation tool tests ────────────────────────────────


def test_schedule_consultation_success() -> None:
    """Successfully books a consultation."""
    from src.tools.schedule_consultation import execute_schedule_consultation

    mock_core = AsyncMock()
    mock_core.create_home_puja_booking.return_value = {
        "id": "consult-789",
        "status": "confirmed",
    }

    result = asyncio.run(
        execute_schedule_consultation(
            core_service=mock_core,
            current_user=SimpleNamespace(user_id="test-user", token="tok"),
            consultant_id="pandit-42",
            consultant_name="Pandit Sharma",
            preferred_date="2026-05-25",
            preferred_time="15:00",
            concern="career guidance",
        )
    )

    assert result["event_name"] == "consultation_booked"
    assert result["booking_id"] == "consult-789"
    assert "Pandit Sharma" in result["summary"]


def test_schedule_consultation_requires_auth() -> None:
    """Returns error when no user is authenticated."""
    from src.tools.schedule_consultation import execute_schedule_consultation

    result = asyncio.run(
        execute_schedule_consultation(
            core_service=AsyncMock(),
            current_user=None,
            consultant_id="pandit-42",
        )
    )

    assert result["event_name"] == "consultation_booking_failed"
    assert "logged in" in result["summary"].lower()


def test_schedule_consultation_handles_error() -> None:
    """Gracefully handles core-service failures."""
    from src.tools.schedule_consultation import execute_schedule_consultation

    mock_core = AsyncMock()
    mock_core.create_home_puja_booking.side_effect = CoreServiceError("Pandit unavailable")

    result = asyncio.run(
        execute_schedule_consultation(
            core_service=mock_core,
            current_user=SimpleNamespace(user_id="test-user", token="tok"),
            consultant_id="pandit-42",
            consultant_name="Pandit Sharma",
        )
    )

    assert result["event_name"] == "consultation_booking_failed"
    assert "Pandit unavailable" in result["summary"]


def test_schedule_consultation_handles_none_response() -> None:
    """Handles transient failures gracefully."""
    from src.tools.schedule_consultation import execute_schedule_consultation

    mock_core = AsyncMock()
    mock_core.create_home_puja_booking.return_value = None

    result = asyncio.run(
        execute_schedule_consultation(
            core_service=mock_core,
            current_user=SimpleNamespace(user_id="test-user", token="tok"),
            consultant_id="pandit-42",
        )
    )

    assert result["event_name"] == "consultation_booking_failed"
    assert "temporarily unavailable" in result["summary"].lower()


# ── Planner action detection tests ──────────────────────────────────


def test_planner_recognizes_confirm_booking_action() -> None:
    """Planner TOOL_ACTIONS includes confirm_booking."""
    from src.core.planner import TOOL_ACTIONS

    assert "confirm_booking" in TOOL_ACTIONS
    assert "schedule_consultation" in TOOL_ACTIONS


def test_planner_action_type_includes_new_actions() -> None:
    """PlannerResult can hold the new action types."""
    from src.core.planner import PlannerResult

    result = PlannerResult(
        action="confirm_booking",
        confidence=0.92,
        reasoning="User wants to confirm the booking",
        arguments={"service_id": "svc-1", "service_type": "home_puja"},
    )
    assert result.action == "confirm_booking"

    result2 = PlannerResult(
        action="schedule_consultation",
        confidence=0.90,
        reasoning="User wants to schedule a consultation",
        arguments={"consultant_id": "p-1"},
    )
    assert result2.action == "schedule_consultation"


# ── Session state tracking tests ────────────────────────────────────


def test_session_state_format_includes_pending_booking() -> None:
    """_format_compact_session_context surfaces pending_booking."""
    from src.core.chat_service import ChatService

    state = {
        "active_intent": "confirm_booking",
        "pending_booking": {
            "service_id": "svc-123",
            "service_type": "home_puja",
            "service_name": "Shani Puja",
        },
    }

    formatted = ChatService._format_compact_session_context(state)
    assert formatted is not None
    assert "Shani Puja" in formatted
    assert "pending" in formatted.lower()


def test_session_state_format_no_pending_booking() -> None:
    """No pending booking line when key is absent."""
    from src.core.chat_service import ChatService

    state = {
        "active_intent": "book_pooja",
        "response_language": "hinglish",
        "pending_booking": None,
    }

    formatted = ChatService._format_compact_session_context(state)
    if formatted:
        assert "Preferred language: hinglish" in formatted
        assert "pending booking" not in formatted.lower()


# ── Guardrail integration tests ─────────────────────────────────────


def test_guardrail_blocks_fear_monetization_for_confirm_booking() -> None:
    """Fear-based monetization check applies to confirm_booking too."""
    from src.core.guardrails import tool_specific_guardrail

    result = tool_specific_guardrail(
        "confirm_booking",
        "someone put a curse on my family I need to book this puja to remove it",
        {},
    )
    assert result.allowed is False
    assert "fear" in result.reason.lower() or "monetization" in result.reason.lower()


# ── check_booking tool tests ────────────────────────────────────────


def test_check_booking_returns_history() -> None:
    """Returns booking history when bookings exist."""
    from src.tools.check_booking import execute_check_booking

    mock_core = AsyncMock()
    mock_core.list_user_bookings.return_value = [
        {"id": "b1", "service_name": "Shani Puja", "status": "confirmed"},
        {"id": "b2", "service_name": "Navgraha Homam", "status": "completed"},
    ]

    result = asyncio.run(
        execute_check_booking(
            core_service=mock_core,
            current_user=SimpleNamespace(user_id="test-user", token="tok"),
        )
    )

    assert result["event_name"] == "booking_history"
    assert len(result["bookings"]) == 2
    assert result["bookings"][0]["service_name"] == "Shani Puja"
    assert "Shani Puja" in result["summary"]


def test_check_booking_empty_history() -> None:
    """Returns empty message when no bookings exist."""
    from src.tools.check_booking import execute_check_booking

    mock_core = AsyncMock()
    mock_core.list_user_bookings.return_value = []

    result = asyncio.run(
        execute_check_booking(
            core_service=mock_core,
            current_user=SimpleNamespace(user_id="test-user", token="tok"),
        )
    )

    assert result["event_name"] == "booking_history"
    assert result["bookings"] == []
    assert "don't have" in result["summary"].lower()


def test_check_booking_requires_auth() -> None:
    """Returns error when no user is authenticated."""
    from src.tools.check_booking import execute_check_booking

    result = asyncio.run(
        execute_check_booking(
            core_service=AsyncMock(),
            current_user=None,
        )
    )

    assert result["event_name"] == "booking_history"
    assert "logged in" in result["summary"].lower()


def test_check_booking_with_status_filter() -> None:
    """Passes status filter to core service."""
    from src.tools.check_booking import execute_check_booking

    mock_core = AsyncMock()
    mock_core.list_user_bookings.return_value = [
        {"id": "b1", "service_name": "Shani Puja", "status": "pending"},
    ]

    asyncio.run(
        execute_check_booking(
            core_service=mock_core,
            current_user=SimpleNamespace(user_id="test-user", token="tok"),
            status_filter="pending",
        )
    )

    mock_core.list_user_bookings.assert_called_once()
    call_kwargs = mock_core.list_user_bookings.call_args
    assert call_kwargs.kwargs.get("status_filter") == "pending"
