from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings
from astro_chatbot_service.models.database import Base

engine: Engine | None = None
SessionLocal = sessionmaker(autoflush=False, autocommit=False, future=True)


def configure_engine(database_url: str | None = None) -> Engine:
    global engine

    resolved_url = database_url or settings.DATABASE_URL
    connect_args = {"check_same_thread": False} if resolved_url.startswith("sqlite") else {}
    if engine is not None:
        engine.dispose()

    engine = create_engine(
        resolved_url,
        future=True,
        connect_args=connect_args,
    )
    SessionLocal.configure(bind=engine)
    return engine


def init_db() -> None:
    db_engine = engine or configure_engine()
    Base.metadata.create_all(bind=db_engine)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
