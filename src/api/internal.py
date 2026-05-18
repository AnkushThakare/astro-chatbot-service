from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, Field

from src.api.kundali import BirthDetails
from src.core.config import settings
from src.core.core_service import CoreServiceClient
from src.core.daily_insights import generate_daily_insight
from src.core.memory import MemoryService
from src.db.repositories.users import UserRepository
from src.db.session import get_db
from sqlalchemy.orm import Session

router = APIRouter()


class InvalidateCacheRequest(BaseModel):
    user_id: str = Field(min_length=1)


class DailyInsightRequest(BaseModel):
    birth_details: BirthDetails
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

    memory_context = request.long_term_memory
    if not memory_context:
        memory_service = MemoryService(db)
        if request.external_user_id:
            user = UserRepository(db).get_by_external_id(request.external_user_id)
            if user is not None:
                memory_context = memory_service.long_term_context(
                    request.session_id or f"user-{request.external_user_id}",
                    user_id=user.id,
                )
        elif request.session_id:
            memory_context = memory_service.long_term_context(request.session_id)

    insight = await generate_daily_insight(
        birth_details=request.birth_details.model_dump(mode="json"),
        memory_context=memory_context,
        preferred_language=request.preferred_language,
    )
    return {
        "status": "ok",
        "insight": insight,
    }
