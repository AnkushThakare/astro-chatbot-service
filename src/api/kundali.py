from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from src.auth.jwt import AuthenticatedUser, get_optional_current_user
from src.astro.kundli import compute_kundali
from src.core.config import settings
from src.core.core_service import CoreServiceClient
from src.tools.show_kundali import show_kundali

router = APIRouter()


class BirthDetails(BaseModel):
    name: str | None = None
    latitude: float
    longitude: float
    birth_datetime: datetime
    timezone_str: str | None = None
    ayanamsha: str = Field(default="LAHIRI")
    house_system: str = Field(default="W")


@router.post("/kundali/compute")
async def compute_kundali_endpoint(
    request: BirthDetails,
    current_user: AuthenticatedUser | None = Depends(get_optional_current_user),
) -> dict[str, Any]:
    core_service_client = CoreServiceClient(settings)
    chart = await core_service_client.generate_kundli(
        request.model_dump(mode="json"),
        current_user,
    )
    if chart is None:
        chart = await compute_kundali(request.model_dump(mode="json"))
    return {
        "chart": chart,
        "summary": show_kundali(chart),
    }
