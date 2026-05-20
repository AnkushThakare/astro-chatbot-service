from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.auth.jwt import AuthenticatedUser, get_optional_current_user
from src.core.energy_flow import EnergyFlowService
from src.db.repositories.users import UserRepository
from src.db.session import get_db

router = APIRouter()


class BehaviorEventInput(BaseModel):
    event_type: str = Field(min_length=1, max_length=64)
    occurred_at: datetime | None = None
    source: str = Field(default="client", min_length=1, max_length=32)
    payload: dict[str, Any] = Field(default_factory=dict)


class BehaviorEventBatchRequest(BaseModel):
    session_id: str = Field(min_length=1)
    events: list[BehaviorEventInput] = Field(min_length=1, max_length=100)


@router.post("/behavior/events")
async def ingest_behavior_events(
    request: BehaviorEventBatchRequest,
    db: Session = Depends(get_db),
    current_user: AuthenticatedUser | None = Depends(get_optional_current_user),
) -> dict[str, Any]:
    service = EnergyFlowService(db)
    internal_user_id = None
    if current_user is not None:
        user = UserRepository(db).get_or_create(current_user.user_id)
        internal_user_id = user.id
    snapshot = service.record_events(
        session_id=request.session_id,
        user_id=internal_user_id,
        events=[
            {
                "event_type": event.event_type,
                "occurred_at": event.occurred_at,
                "source": event.source,
                "payload": event.payload,
            }
            for event in request.events
        ],
    )
    return {
        "status": "ok",
        "ingested_count": len(request.events),
        "state": service.serialize_snapshot(snapshot),
    }


@router.get("/behavior/state/{session_id}")
async def get_behavior_state(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: AuthenticatedUser | None = Depends(get_optional_current_user),
) -> dict[str, Any]:
    service = EnergyFlowService(db)
    internal_user_id = None
    if current_user is not None:
        user = UserRepository(db).get_or_create(current_user.user_id)
        internal_user_id = user.id
    snapshot = service.get_snapshot(session_id=session_id, user_id=internal_user_id)
    return {
        "status": "ok",
        "state": service.serialize_snapshot(snapshot),
    }
