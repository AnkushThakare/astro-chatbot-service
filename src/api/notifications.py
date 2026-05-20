"""Notification API — device token registration + daily insight trigger."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from src.auth.jwt import AuthenticatedUser, get_current_user
from src.core.config import settings
from src.core.logging import get_logger
from src.core.push import PushService
from src.core.scheduler import run_daily_insights
from src.db.models import DailyInsightLog, DeviceToken
from src.db.repositories.users import UserRepository
from src.db.session import get_db

logger = get_logger(__name__)
router = APIRouter()


# ── Device token endpoints ──────────────────────────────────────────


class RegisterTokenRequest(BaseModel):
    token: str = Field(min_length=10, max_length=512)
    platform: str = Field(default="expo", pattern="^(expo|fcm|apns)$")


class UnregisterTokenRequest(BaseModel):
    token: str = Field(min_length=10, max_length=512)


@router.post("/notifications/register-token")
async def register_device_token(
    request: RegisterTokenRequest,
    db: Session = Depends(get_db),
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> dict[str, str]:
    """Register a push notification token for the authenticated user."""
    user = UserRepository(db).get_or_create(current_user.user_id)

    # Check if token already exists for this user
    existing = db.execute(
        select(DeviceToken).where(
            and_(
                DeviceToken.user_id == user.id,
                DeviceToken.token == request.token,
            )
        )
    ).scalar_one_or_none()

    if existing:
        if not existing.is_active:
            existing.is_active = True
            existing.platform = request.platform
            db.commit()
        return {"status": "ok", "action": "reactivated" if not existing.is_active else "exists"}

    device_token = DeviceToken(
        user_id=user.id,
        token=request.token,
        platform=request.platform,
    )
    db.add(device_token)
    db.commit()
    return {"status": "ok", "action": "registered"}


@router.post("/notifications/unregister-token")
async def unregister_device_token(
    request: UnregisterTokenRequest,
    db: Session = Depends(get_db),
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> dict[str, str]:
    """Deactivate a push notification token."""
    user = UserRepository(db).get_or_create(current_user.user_id)

    existing = db.execute(
        select(DeviceToken).where(
            and_(
                DeviceToken.user_id == user.id,
                DeviceToken.token == request.token,
            )
        )
    ).scalar_one_or_none()

    if existing and existing.is_active:
        existing.is_active = False
        db.commit()
        return {"status": "ok", "action": "deactivated"}

    return {"status": "ok", "action": "not_found"}


# ── Daily insight scheduler trigger ─────────────────────────────────


@router.post("/internal/run-daily-insights")
async def trigger_daily_insights(
    db: Session = Depends(get_db),
    x_internal_api_key: str | None = Header(default=None),
    dry_run: bool = False,
) -> dict[str, Any]:
    """Trigger daily insight generation for all eligible users.

    Protected by internal API key. Call from cron or scheduler.
    """
    if x_internal_api_key != settings.INTERNAL_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid internal API key",
        )

    push_service = PushService()
    try:
        stats = await run_daily_insights(
            db,
            push_service=push_service,
            dry_run=dry_run,
        )
    finally:
        await push_service.close()

    return {"status": "ok", **stats}


# ── Daily insight engagement tracking ────────────────────────────────


class InsightEngagementRequest(BaseModel):
    date: str = Field(min_length=10, max_length=10)  # YYYY-MM-DD
    action: str = Field(pattern="^(opened|dismissed|tapped)$")


@router.post("/notifications/insight-engagement")
async def track_insight_engagement(
    request: InsightEngagementRequest,
    db: Session = Depends(get_db),
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> dict[str, str]:
    """Track when a user opens/taps/dismisses a daily insight."""
    user = UserRepository(db).get_or_create(current_user.user_id)

    log = db.execute(
        select(DailyInsightLog).where(
            and_(
                DailyInsightLog.user_id == user.id,
                DailyInsightLog.generated_for_date == request.date,
            )
        )
    ).scalar_one_or_none()

    if log is None:
        return {"status": "ok", "action": "no_insight_found"}

    if request.action == "tapped":
        log.push_tapped = True

    db.commit()
    return {"status": "ok", "action": request.action, "recorded": True}
