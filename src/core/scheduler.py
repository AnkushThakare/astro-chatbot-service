"""Daily insight scheduler — generates and delivers morning insights to all eligible users.

Designed to be triggered by:
  1. Internal API endpoint (POST /internal/run-daily-insights)
  2. External cron (Railway cron, Cloud Scheduler, etc.)
  3. APScheduler if added later

Processes users in batches to avoid memory spikes.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any

from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from src.core.config import settings
from src.core.daily_insights import generate_daily_insight
from src.core.energy_flow import EnergyFlowService
from src.core.logging import get_logger
from src.core.memory import MemoryService
from src.core.push import PushResult, PushService
from src.db.models import DailyInsightLog, DeviceToken, User
from src.db.repositories.users import UserRepository

logger = get_logger(__name__)

_BATCH_SIZE = 50


async def run_daily_insights(
    db: Session,
    *,
    push_service: PushService | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Generate and deliver daily insights for all eligible users.

    Args:
        db: Database session.
        push_service: Push notification service. If None, insights are
            generated but not delivered via push.
        dry_run: If True, generate insights but don't save logs or send push.

    Returns:
        Summary dict with counts and errors.
    """
    today = date.today().isoformat()
    user_repo = UserRepository(db)
    memory_service = MemoryService(db)
    energy_service = EnergyFlowService(db)

    # Find users with birth details who haven't received today's insight
    already_sent = set(
        row[0]
        for row in db.execute(
            select(DailyInsightLog.user_id).where(
                DailyInsightLog.generated_for_date == today,
            )
        ).all()
    )

    eligible_users = db.execute(
        select(User).where(User.birth_details_json.isnot(None))
    ).scalars().all()

    stats: dict[str, Any] = {
        "date": today,
        "eligible_users": len(eligible_users),
        "already_sent": len(already_sent),
        "generated": 0,
        "push_sent": 0,
        "push_failed": 0,
        "errors": [],
    }

    for user in eligible_users:
        if user.id in already_sent:
            continue

        try:
            insight = await _generate_for_user(
                user=user,
                memory_service=memory_service,
                energy_service=energy_service,
            )

            if not dry_run:
                # Log the insight
                log = DailyInsightLog(
                    user_id=user.id,
                    generated_for_date=today,
                    push_delivered=False,
                    insight_json=insight,
                )
                db.add(log)
                db.flush()

                # Deliver via push
                if push_service is not None:
                    push_results = await _deliver_push(
                        db=db,
                        user=user,
                        insight=insight,
                        push_service=push_service,
                    )
                    delivered = any(r.success for r in push_results)
                    if delivered:
                        log.push_delivered = True
                        stats["push_sent"] += 1
                    elif push_results:
                        stats["push_failed"] += 1

            stats["generated"] += 1

        except Exception as exc:
            logger.warning(
                "daily_insight_generation_failed | user_id=%d | error=%s",
                user.id,
                exc,
            )
            stats["errors"].append({"user_id": user.id, "error": str(exc)})

    if not dry_run:
        db.commit()

    logger.info(
        "daily_insight_batch_complete | generated=%d | push_sent=%d | push_failed=%d | errors=%d",
        stats["generated"],
        stats["push_sent"],
        stats["push_failed"],
        len(stats["errors"]),
    )
    return stats


async def _generate_for_user(
    *,
    user: User,
    memory_service: MemoryService,
    energy_service: EnergyFlowService,
) -> dict[str, Any]:
    """Generate a daily insight for a single user."""
    birth_details = dict(user.birth_details_json) if user.birth_details_json else {}

    # Load memory context
    memory_context = None
    session_id = f"user-{user.external_user_id}"
    try:
        memory_context = memory_service.long_term_context(
            session_id,
            user_id=user.id,
        )
    except Exception:
        pass

    # Load energy snapshot
    energy_snapshot = None
    try:
        snapshot = energy_service.get_snapshot(
            session_id=session_id,
            user_id=user.id,
        )
        if snapshot and snapshot.signal_count > 0:
            energy_snapshot = snapshot
    except Exception:
        pass

    # Resolve user name
    user_name = None
    if user.full_name:
        user_name = user.full_name.split()[0]

    # Resolve preferred language
    preferred_language = getattr(user, "preferred_language", "en") or "en"

    return await generate_daily_insight(
        birth_details=birth_details,
        memory_context=memory_context,
        preferred_language=preferred_language,
        energy_snapshot=energy_snapshot,
        user_name=user_name,
    )


async def _deliver_push(
    *,
    db: Session,
    user: User,
    insight: dict[str, Any],
    push_service: PushService,
) -> list[PushResult]:
    """Send push notification for a daily insight."""
    active_tokens = db.execute(
        select(DeviceToken.token).where(
            and_(
                DeviceToken.user_id == user.id,
                DeviceToken.is_active.is_(True),
            )
        )
    ).scalars().all()

    if not active_tokens:
        return []

    title = insight.get("headline", "Your Daily Insight")
    body = insight.get("push_text") or insight.get("action", "Check your daily insight")

    results = await push_service.send(
        tokens=list(active_tokens),
        title=title,
        body=body,
        data={
            "type": "daily_insight",
            "date": insight.get("generated_for_date", ""),
            "focus_area": insight.get("focus_area", ""),
        },
    )

    # Deactivate tokens that are permanently invalid
    for result in results:
        if not result.success and result.error and "DeviceNotRegistered" in result.error:
            db.execute(
                DeviceToken.__table__.update()
                .where(DeviceToken.token == result.token)
                .values(is_active=False)
            )
            logger.info("deactivated_invalid_push_token | token=%s...", result.token[:20])

    return results
