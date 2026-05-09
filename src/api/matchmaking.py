from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, model_validator

from src.api.kundali import BirthDetails
from src.auth.jwt import AuthenticatedUser, get_current_user
from src.core.config import settings
from src.core.core_service import CoreServiceClient
from src.tools.show_matchmaking import show_matchmaking

router = APIRouter()


class MatchmakingPerson(BaseModel):
    birth_details: BirthDetails
    gender: str = Field(pattern="^(male|female)$")


class MatchmakingDetails(BaseModel):
    primary: MatchmakingPerson
    partner: MatchmakingPerson
    lang: str = Field(default="en", min_length=2, max_length=8)
    include_summary: bool = True
    summary_type: str = Field(default="medium", pattern="^(short|medium|detailed)$")

    @model_validator(mode="after")
    def validate_pairing(self) -> "MatchmakingDetails":
        if self.primary.gender == self.partner.gender:
            raise ValueError("Matchmaking requires one male and one female profile")
        return self


@router.post("/matchmaking/compute")
async def compute_matchmaking_endpoint(
    request: MatchmakingDetails,
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> dict[str, Any]:
    core_service_client = CoreServiceClient(settings)
    result = await core_service_client.generate_matchmaking(
        request.model_dump(mode="json"),
        current_user,
    )
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Unable to compute matchmaking right now.",
        )
    return {
        "matchmaking": result,
        "summary": show_matchmaking(result),
    }
