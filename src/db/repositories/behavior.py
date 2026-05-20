from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.models import BehaviorEvent, BehaviorProfile, Conversation


class BehaviorRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def resolve_conversation_id(self, session_id: str) -> int | None:
        statement = select(Conversation.id).where(Conversation.session_id == session_id)
        return self.db.execute(statement).scalar_one_or_none()

    def add_events(
        self,
        *,
        session_id: str,
        events: list[dict[str, Any]],
        user_id: int | None = None,
        conversation_id: int | None = None,
    ) -> list[BehaviorEvent]:
        rows: list[BehaviorEvent] = []
        for event in events:
            occurred_at = event.get("occurred_at")
            if not isinstance(occurred_at, datetime):
                occurred_at = datetime.now(timezone.utc)
            row = BehaviorEvent(
                user_id=user_id,
                conversation_id=conversation_id,
                session_id=session_id,
                event_type=str(event["event_type"]),
                source=str(event.get("source") or "client"),
                payload_json=event.get("payload") if isinstance(event.get("payload"), dict) else {},
                occurred_at=occurred_at,
            )
            self.db.add(row)
            rows.append(row)
        self.db.commit()
        for row in rows:
            self.db.refresh(row)
        return rows

    def list_recent_events(
        self,
        *,
        session_id: str,
        user_id: int | None = None,
        limit: int = 240,
        lookback_days: int = 30,
    ) -> list[BehaviorEvent]:
        since = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        if user_id is not None:
            statement = (
                select(BehaviorEvent)
                .where(
                    BehaviorEvent.user_id == user_id,
                    BehaviorEvent.occurred_at >= since,
                )
                .order_by(BehaviorEvent.occurred_at.desc(), BehaviorEvent.id.desc())
                .limit(limit)
            )
        else:
            statement = (
                select(BehaviorEvent)
                .where(
                    BehaviorEvent.session_id == session_id,
                    BehaviorEvent.occurred_at >= since,
                )
                .order_by(BehaviorEvent.occurred_at.desc(), BehaviorEvent.id.desc())
                .limit(limit)
            )
        rows = self.db.execute(statement).scalars().all()
        return list(reversed(rows))

    def get_profile(self, scope_key: str) -> BehaviorProfile | None:
        statement = select(BehaviorProfile).where(BehaviorProfile.scope_key == scope_key)
        return self.db.execute(statement).scalar_one_or_none()

    def upsert_profile(
        self,
        *,
        scope_key: str,
        scope_type: str,
        session_id: str | None,
        user_id: int | None,
        conversation_id: int | None,
        metrics: dict[str, Any],
    ) -> BehaviorProfile:
        row = self.get_profile(scope_key)
        if row is None:
            row = BehaviorProfile(
                scope_key=scope_key,
                scope_type=scope_type,
                session_id=session_id,
                user_id=user_id,
                conversation_id=conversation_id,
            )
            self.db.add(row)

        row.scope_type = scope_type
        row.session_id = session_id
        row.user_id = user_id
        row.conversation_id = conversation_id
        row.overall_alignment = int(metrics.get("overall_alignment", 50))
        row.stress_score = int(metrics.get("stress_score", 0))
        row.focus_score = int(metrics.get("focus_score", 50))
        row.emotional_drift_score = int(metrics.get("emotional_drift_score", 0))
        row.cognitive_overload_score = int(metrics.get("cognitive_overload_score", 0))
        row.clarity_score = int(metrics.get("clarity_score", 50))
        row.behavioral_consistency_score = int(metrics.get("behavioral_consistency_score", 50))
        row.emotional_state = str(metrics.get("emotional_state") or "steady")
        row.focus_state = str(metrics.get("focus_state") or "neutral")
        row.behavioral_state = str(metrics.get("behavioral_state") or "steady")
        row.signal_count = int(metrics.get("signal_count", 0))
        row.summary_text = metrics.get("summary_text")
        row.signals_json = metrics.get("signals_json") if isinstance(metrics.get("signals_json"), dict) else {}
        row.last_event_at = metrics.get("last_event_at")
        self.db.commit()
        self.db.refresh(row)
        return row
