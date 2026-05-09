from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from app.api.endpoints import health
from app.api.router import api_router
from app.core.config import settings
from app.core.database import configure_engine, init_db
from app.core.logging import get_logger, setup_logging
from app.middleware.correlation import CorrelationMiddleware
from app.middleware.error_handler import ErrorHandlerMiddleware

setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_engine()
    init_db()
    logger.info("Database initialized successfully")
    yield


def create_app() -> FastAPI:
    docs_url = f"{settings.API_V1_PREFIX}/docs" if settings.DOCS_ENABLED else None
    redoc_url = f"{settings.API_V1_PREFIX}/redoc" if settings.DOCS_ENABLED else None
    openapi_url = f"{settings.API_V1_PREFIX}/openapi.json" if settings.DOCS_ENABLED else None

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        docs_url=docs_url,
        redoc_url=redoc_url,
        openapi_url=openapi_url,
        redirect_slashes=False,
        lifespan=lifespan,
        response_model_exclude_none=True,
    )
    app.add_middleware(ErrorHandlerMiddleware)
    app.add_middleware(CorrelationMiddleware)
    app.include_router(health.router, tags=["health"])
    app.include_router(api_router, prefix=settings.API_V1_PREFIX)
    return app


app = create_app()

__all__ = ["app", "create_app"]
