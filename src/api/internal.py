from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.api.kundali import BirthDetails
from src.core.config import settings
from src.core.core_service import CoreServiceClient
from src.core.daily_insights import generate_daily_insight
from src.core.energy_flow import EnergyFlowService
from src.core.memory import MemoryService
from src.db.repositories.users import UserRepository
from src.db.session import get_db

router = APIRouter()


class InvalidateCacheRequest(BaseModel):
    user_id: str = Field(min_length=1)


class DailyInsightRequest(BaseModel):
    birth_details: BirthDetails | None = Field(
        default=None,
        description="Birth details. If omitted, loads from the user's stored profile.",
    )
    session_id: str | None = Field(default=None, min_length=1)
    external_user_id: str | None = Field(default=None, min_length=1)
    long_term_memory: str | None = None
    preferred_language: str = Field(default="en", min_length=2, max_length=16)


@router.post("/internal/invalidate-cache")
async def invalidate_cache(
    request: InvalidateCacheRequest,
    x_internal_api_key: str | None = Header(default=None),
) -> dict[str, str]:
    if x_internal_api_key != settings.INTERNAL_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid internal API key",
        )

    CoreServiceClient.invalidate_birth_details_cache(request.user_id)
    return {"status": "ok", "user_id": request.user_id}


@router.post("/internal/daily-insight")
async def daily_insight(
    request: DailyInsightRequest,
    x_internal_api_key: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    if x_internal_api_key != settings.INTERNAL_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid internal API key",
        )

    user_repo = UserRepository(db)

    # Resolve birth details: request body → stored in DB
    birth_payload: dict[str, Any] | None = None
    if request.birth_details is not None:
        birth_payload = request.birth_details.model_dump(mode="json")
    elif request.external_user_id:
        birth_payload = user_repo.get_birth_details(request.external_user_id)

    if not birth_payload:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No birth details provided or stored for this user.",
        )

    # Resolve memory context
    memory_context = request.long_term_memory
    if not memory_context:
        memory_service = MemoryService(db)
        if request.external_user_id:
            user = user_repo.get_by_external_id(request.external_user_id)
            if user is not None:
                memory_context = memory_service.long_term_context(
                    request.session_id or f"user-{request.external_user_id}",
                    user_id=user.id,
                )
        elif request.session_id:
            memory_context = memory_service.long_term_context(request.session_id)

    # Resolve energy flow snapshot
    energy_snapshot = None
    if request.external_user_id:
        try:
            energy_service = EnergyFlowService(db)
            user = user_repo.get_by_external_id(request.external_user_id)
            session_id = request.session_id or f"user-{request.external_user_id}"
            energy_snapshot = energy_service.get_snapshot(
                session_id=session_id,
                user_id=user.id if user else None,
            )
            if energy_snapshot and energy_snapshot.signal_count == 0:
                energy_snapshot = None
        except Exception:
            pass

    # Resolve user name
    user_name = None
    if request.external_user_id:
        user = user_repo.get_by_external_id(request.external_user_id)
        if user and user.full_name:
            user_name = user.full_name.split()[0]

    insight = await generate_daily_insight(
        birth_details=birth_payload,
        memory_context=memory_context,
        preferred_language=request.preferred_language,
        energy_snapshot=energy_snapshot,
        user_name=user_name,
    )
    return {
        "status": "ok",
        "insight": insight,
    }
