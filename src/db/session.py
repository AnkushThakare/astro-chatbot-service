from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Generator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, close_all_sessions, sessionmaker

from src.core.config import settings
from src.db.models import Base

sync_engine: Engine | None = None
async_engine: AsyncEngine | None = None
SessionLocal = sessionmaker(autoflush=False, autocommit=False, future=True)
AsyncSessionLocal = async_sessionmaker(class_=AsyncSession, autoflush=False, expire_on_commit=False)


def configure_database(
    sync_database_url: str | None = None,
    async_database_url: str | None = None,
) -> tuple[Engine, AsyncEngine]:
    global sync_engine, async_engine

    resolved_sync = sync_database_url or settings.sync_database_url
    resolved_async = async_database_url or settings.async_database_url
    sync_connect_args = {"check_same_thread": False} if resolved_sync.startswith("sqlite") else {}

    if sync_engine is not None:
        sync_engine.dispose()
    if async_engine is not None:
        async_engine.sync_engine.dispose()

    sync_engine = create_engine(resolved_sync, future=True, connect_args=sync_connect_args)
    async_engine = create_async_engine(resolved_async, future=True)
    SessionLocal.configure(bind=sync_engine)
    AsyncSessionLocal.configure(bind=async_engine)
    return sync_engine, async_engine


def init_db() -> None:
    db_engine, _ = configure_database() if sync_engine is None or async_engine is None else (sync_engine, async_engine)
    Base.metadata.create_all(bind=db_engine)


def shutdown_database() -> None:
    global sync_engine, async_engine

    close_all_sessions()

    if async_engine is not None:
        try:
            asyncio.run(async_engine.dispose())
        except RuntimeError:
            async_engine.sync_engine.dispose()
        async_engine = None

    if sync_engine is not None:
        sync_engine.dispose()
        sync_engine = None


def get_db() -> Generator[Session, None, None]:
    if sync_engine is None:
        configure_database()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    if async_engine is None:
        configure_database()
    async with AsyncSessionLocal() as db:
        yield db
