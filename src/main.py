from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.behavior import router as behavior_router
from src.api.bookings import router as bookings_router
from src.api.chat import router as chat_router
from src.api.health import router as health_router
from src.api.internal import router as internal_router
from src.api.memory import router as memory_router
from src.api.kundali import router as kundali_router
from src.api.matchmaking import router as matchmaking_router
from src.api.notifications import router as notifications_router
from src.core.config import settings
from src.core.core_service import CoreServiceClient
from src.core.idempotency import chat_idempotency_store
from src.core.llm import LLMClient
from src.core.logging import get_logger, setup_logging
from src.core.product_catalog import catalog_index
from src.core.middleware import CorrelationMiddleware, ErrorHandlerMiddleware
from src.db.session import configure_database, init_db, shutdown_database

setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_database()
    init_db()
    logger.info("Database initialized successfully")
    # Pre-warm product catalog index in background
    try:
        _core_client = CoreServiceClient(settings)
        await catalog_index.refresh(_core_client)
        logger.info("Product catalog index pre-warmed: %d products", catalog_index.product_count)
    except Exception as exc:
        logger.warning("Product catalog pre-warm failed (will retry on first request): %s", exc)
    yield
    await chat_idempotency_store.close()
    await LLMClient.close()
    await CoreServiceClient.close()
    shutdown_database()
    logger.info("Database and HTTP connections closed")


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
    app.include_router(behavior_router, tags=["behavior"])
    app.include_router(health_router, tags=["health"])
    app.include_router(internal_router, tags=["internal"])
    app.include_router(chat_router, tags=["chat"])
    app.include_router(kundali_router, tags=["kundali"])
    app.include_router(matchmaking_router, tags=["matchmaking"])
    app.include_router(bookings_router, tags=["bookings"])
    app.include_router(notifications_router, tags=["notifications"])
    app.include_router(memory_router, tags=["memory"])
    app.include_router(behavior_router, prefix=settings.API_V1_PREFIX, tags=["behavior"])
    app.include_router(health_router, prefix=settings.API_V1_PREFIX, tags=["health"])
    app.include_router(internal_router, prefix=settings.API_V1_PREFIX, tags=["internal"])
    app.include_router(chat_router, prefix=settings.API_V1_PREFIX, tags=["chat"])
    app.include_router(kundali_router, prefix=settings.API_V1_PREFIX, tags=["kundali"])
    app.include_router(matchmaking_router, prefix=settings.API_V1_PREFIX, tags=["matchmaking"])
    app.include_router(bookings_router, prefix=settings.API_V1_PREFIX, tags=["bookings"])
    app.include_router(notifications_router, prefix=settings.API_V1_PREFIX, tags=["notifications"])
    app.include_router(memory_router, prefix=settings.API_V1_PREFIX, tags=["memory"])
    return app


app = create_app()

__all__ = ["app", "create_app"]
