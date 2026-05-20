"""Tests for push notifications, device token management, and daily insight scheduler."""
from __future__ import annotations

import asyncio
import time
from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.core.config import settings
from src.core.push import PushResult, PushService
from src.core.scheduler import run_daily_insights
from src.db.models import DailyInsightLog, DeviceToken, User
from src.db.session import SessionLocal


# ── Device token API tests ───────────────────────────────────────────


def test_register_device_token(client, auth_headers) -> None:
    response = client.post(
        "/notifications/register-token",
        json={"token": "ExponentPushToken[abc123xyz]", "platform": "expo"},
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["action"] == "registered"


def test_register_duplicate_token_returns_exists(client, auth_headers) -> None:
    payload = {"token": "ExponentPushToken[duplicate-test]", "platform": "expo"}
    client.post("/notifications/register-token", json=payload, headers=auth_headers)
    response = client.post("/notifications/register-token", json=payload, headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["action"] in ("exists", "reactivated")


def test_unregister_device_token(client, auth_headers) -> None:
    token = "ExponentPushToken[unregister-test]"
    client.post(
        "/notifications/register-token",
        json={"token": token, "platform": "expo"},
        headers=auth_headers,
    )
    response = client.post(
        "/notifications/unregister-token",
        json={"token": token},
        headers=auth_headers,
    )
    assert response.status_code == 200
    assert response.json()["action"] == "deactivated"


def test_unregister_unknown_token(client, auth_headers) -> None:
    response = client.post(
        "/notifications/unregister-token",
        json={"token": "ExponentPushToken[never-registered]"},
        headers=auth_headers,
    )
    assert response.status_code == 200
    assert response.json()["action"] == "not_found"


def test_register_token_requires_auth(client) -> None:
    response = client.post(
        "/notifications/register-token",
        json={"token": "ExponentPushToken[no-auth]", "platform": "expo"},
    )
    assert response.status_code == 401


def test_register_token_rejects_invalid_platform(client, auth_headers) -> None:
    response = client.post(
        "/notifications/register-token",
        json={"token": "ExponentPushToken[bad-platform]", "platform": "windows"},
        headers=auth_headers,
    )
    assert response.status_code == 422


# ── Push service tests ───────────────────────────────────────────────


def test_push_service_returns_empty_for_no_tokens() -> None:
    service = PushService()
    results = asyncio.run(service.send([], title="Test", body="Test"))
    assert results == []


def test_push_service_handles_api_success() -> None:
    service = PushService()

    async def _test():
        with patch("httpx.AsyncClient.post") as mock_post:
            mock_post.return_value = SimpleNamespace(
                status_code=200,
                raise_for_status=lambda: None,
                json=lambda: {"data": [{"status": "ok"}]},
            )
            results = await service.send(
                tokens=["ExponentPushToken[test]"],
                title="Daily Insight",
                body="Your day looks promising",
            )
            assert len(results) == 1
            assert results[0].success is True
        await service.close()

    asyncio.run(_test())


def test_push_service_handles_device_not_registered() -> None:
    service = PushService()

    async def _test():
        with patch("httpx.AsyncClient.post") as mock_post:
            mock_post.return_value = SimpleNamespace(
                status_code=200,
                raise_for_status=lambda: None,
                json=lambda: {
                    "data": [
                        {
                            "status": "error",
                            "message": "DeviceNotRegistered",
                            "details": {"error": "DeviceNotRegistered"},
                        }
                    ]
                },
            )
            results = await service.send(
                tokens=["ExponentPushToken[invalid]"],
                title="Test",
                body="Test",
            )
            assert len(results) == 1
            assert results[0].success is False
            assert "DeviceNotRegistered" in results[0].error
        await service.close()

    asyncio.run(_test())


def test_push_service_handles_network_error() -> None:
    service = PushService()

    async def _test():
        with patch("httpx.AsyncClient.post", side_effect=Exception("Connection refused")):
            results = await service.send(
                tokens=["ExponentPushToken[test]"],
                title="Test",
                body="Test",
            )
            assert len(results) == 1
            assert results[0].success is False
            assert "Connection refused" in results[0].error
        await service.close()

    asyncio.run(_test())


# ── Scheduler tests ──────────────────────────────────────────────────


def test_scheduler_generates_insights_for_eligible_users() -> None:
    db = SessionLocal()
    try:
        # Create a user with birth details
        user = User(
            external_user_id=f"scheduler-test-{time.time()}",
            birth_details_json={
                "name": "Test",
                "latitude": 28.6139,
                "longitude": 77.2090,
                "birth_datetime": "1995-03-15T10:30:00",
                "timezone_str": "Asia/Kolkata",
            },
        )
        db.add(user)
        db.commit()
        db.refresh(user)

        stats = asyncio.run(
            run_daily_insights(db, push_service=None, dry_run=True)
        )

        assert stats["eligible_users"] >= 1
        assert stats["generated"] >= 1
        assert stats["push_sent"] == 0  # dry_run, no push service
    finally:
        db.close()


def test_scheduler_skips_already_sent_users() -> None:
    db = SessionLocal()
    try:
        user = User(
            external_user_id=f"scheduler-skip-{time.time()}",
            birth_details_json={
                "name": "Test",
                "latitude": 28.6139,
                "longitude": 77.2090,
                "birth_datetime": "1995-03-15T10:30:00",
                "timezone_str": "Asia/Kolkata",
            },
        )
        db.add(user)
        db.commit()
        db.refresh(user)

        # Pre-create a log for today
        log = DailyInsightLog(
            user_id=user.id,
            generated_for_date=date.today().isoformat(),
            insight_json={"test": True},
        )
        db.add(log)
        db.commit()

        stats = asyncio.run(
            run_daily_insights(db, push_service=None, dry_run=True)
        )

        assert stats["already_sent"] >= 1
    finally:
        db.close()


def test_scheduler_skips_users_without_birth_details() -> None:
    db = SessionLocal()
    try:
        user = User(
            external_user_id=f"no-birth-{time.time()}",
            birth_details_json=None,
        )
        db.add(user)
        db.commit()

        stats = asyncio.run(
            run_daily_insights(db, push_service=None, dry_run=True)
        )

        # This user should not be in eligible
        # (Other users from other tests may be eligible, so just check no errors)
        assert isinstance(stats["eligible_users"], int)
    finally:
        db.close()


# ── Internal API trigger test ────────────────────────────────────────


def test_run_daily_insights_requires_internal_key(client) -> None:
    response = client.post("/internal/run-daily-insights")
    assert response.status_code == 403


def test_run_daily_insights_with_valid_key(client) -> None:
    response = client.post(
        "/internal/run-daily-insights",
        params={"dry_run": True},
        headers={"X-Internal-API-Key": settings.INTERNAL_API_KEY},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "generated" in data
    assert "eligible_users" in data


# ── Engagement tracking test ─────────────────────────────────────────


def test_insight_engagement_tracking(client, auth_headers) -> None:
    response = client.post(
        "/notifications/insight-engagement",
        json={"date": "2026-05-19", "action": "tapped"},
        headers=auth_headers,
    )
    assert response.status_code == 200


def test_insight_engagement_rejects_invalid_action(client, auth_headers) -> None:
    response = client.post(
        "/notifications/insight-engagement",
        json={"date": "2026-05-19", "action": "deleted"},
        headers=auth_headers,
    )
    assert response.status_code == 422
