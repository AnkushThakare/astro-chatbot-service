from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.bookings import router as bookings_router
from src.api.chat import router as chat_router
from src.api.health import router as health_router
from src.api.kundali import router as kundali_router
from src.api.matchmaking import router as matchmaking_router
from src.core.config import settings
from src.core.logging import get_logger, setup_logging
from src.core.middleware import CorrelationMiddleware, ErrorHandlerMiddleware
from src.db.session import configure_database, init_db

setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_database()
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
    )
    app.add_middleware(ErrorHandlerMiddleware)
    app.add_middleware(CorrelationMiddleware)
    app.include_router(health_router, tags=["health"])
    app.include_router(chat_router, tags=["chat"])
    app.include_router(kundali_router, tags=["kundali"])
    app.include_router(matchmaking_router, tags=["matchmaking"])
    app.include_router(bookings_router, tags=["bookings"])
    app.include_router(health_router, prefix=settings.API_V1_PREFIX, tags=["health"])
    app.include_router(chat_router, prefix=settings.API_V1_PREFIX, tags=["chat"])
    app.include_router(kundali_router, prefix=settings.API_V1_PREFIX, tags=["kundali"])
    app.include_router(matchmaking_router, prefix=settings.API_V1_PREFIX, tags=["matchmaking"])
    app.include_router(bookings_router, prefix=settings.API_V1_PREFIX, tags=["bookings"])
    return app


app = create_app()

__all__ = ["app", "create_app"]
