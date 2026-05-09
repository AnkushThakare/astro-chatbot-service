from fastapi import APIRouter

from app.core.config import settings

router = APIRouter()


@router.get("/health")
def healthcheck() -> dict[str, str]:
    return {
        "status": "ok",
        "service": settings.APP_NAME,
        "environment": settings.APP_ENV,
    }


@router.get("/ready")
def readiness() -> dict[str, str]:
    return {"status": "ready"}
