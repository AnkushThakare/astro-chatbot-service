from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from src.astro.kundli import compute_kundali
from src.tools.show_kundali import show_kundali

router = APIRouter()


class BirthDetails(BaseModel):
    name: str | None = None
    latitude: float
    longitude: float
    birth_datetime: datetime
    ayanamsha: str = Field(default="LAHIRI")
    house_system: str = Field(default="W")


@router.post("/kundali/compute")
async def compute_kundali_endpoint(request: BirthDetails) -> dict[str, Any]:
    chart = await compute_kundali(request.model_dump())
    return {
        "chart": chart,
        "summary": show_kundali(chart),
    }
