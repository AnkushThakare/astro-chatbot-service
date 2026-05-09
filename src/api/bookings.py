from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from src.auth.jwt import AuthenticatedUser, get_current_user
from src.core.config import settings
from src.core.core_service import CoreServiceClient

router = APIRouter()


class PreviewPriceRequest(BaseModel):
    service_tier_id: str = Field(min_length=1)
    use_rewards: bool = False
    use_wallet: bool = False


def _core_service_failure(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=detail)


@router.get("/booking/services")
async def list_booking_services(
    search: str | None = Query(default=None, min_length=1),
    limit: int = Query(default=3, ge=1, le=20),
) -> dict[str, Any]:
    core_service_client = CoreServiceClient(settings)
    items = await core_service_client.list_home_puja_services(search or "", limit=limit)
    return {"items": items}


@router.get("/booking/temple-services")
async def list_temple_booking_services(
    search: str | None = Query(default=None, min_length=1),
    limit: int = Query(default=3, ge=1, le=20),
) -> dict[str, Any]:
    core_service_client = CoreServiceClient(settings)
    items = await core_service_client.list_temple_services(search or "", limit=limit)
    return {"items": items}


@router.get("/booking/pandits")
async def list_booking_pandits(
    search: str | None = Query(default=None, min_length=1),
    limit: int = Query(default=3, ge=1, le=20),
) -> dict[str, Any]:
    core_service_client = CoreServiceClient(settings)
    items = await core_service_client.list_public_pandits(search or "", limit=limit)
    return {"items": items}


@router.post("/booking/preview-price")
async def preview_booking_price(
    request: PreviewPriceRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> dict[str, Any]:
    core_service_client = CoreServiceClient(settings)
    result = await core_service_client.preview_home_puja_price(
        request.model_dump(mode="json"),
        current_user,
    )
    if result is None:
        raise _core_service_failure("Unable to preview booking price right now.")
    return result


@router.post("/booking/home-puja", status_code=status.HTTP_201_CREATED)
async def create_home_puja_booking_endpoint(
    request: dict[str, Any] = Body(...),
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> dict[str, Any]:
    core_service_client = CoreServiceClient(settings)
    result = await core_service_client.create_home_puja_booking(request, current_user)
    if result is None:
        raise _core_service_failure("Unable to create home puja booking right now.")
    return result


@router.post("/booking/temple-puja", status_code=status.HTTP_201_CREATED)
async def create_temple_booking_endpoint(
    request: dict[str, Any] = Body(...),
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> dict[str, Any]:
    core_service_client = CoreServiceClient(settings)
    result = await core_service_client.create_temple_booking(request, current_user)
    if result is None:
        raise _core_service_failure("Unable to create temple booking right now.")
    return result
