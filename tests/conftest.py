from __future__ import annotations

import os
import time
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from jose import jwt

os.environ.setdefault("ENV_FILE", ".env.test")

from src.db.session import configure_database, init_db, shutdown_database
from src.main import app


@pytest.fixture(scope="session", autouse=True)
def test_database() -> Generator[None, None, None]:
    db_path = Path("astro_chatbot_test.db")
    if db_path.exists():
        db_path.unlink()

    configure_database(
        sync_database_url="sqlite:///./astro_chatbot_test.db",
        async_database_url="sqlite+aiosqlite:///./astro_chatbot_test.db",
    )
    init_db()

    yield

    shutdown_database()

    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def client(test_database: None) -> Generator[TestClient, None, None]:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Return Authorization headers with a valid JWT for testing."""
    from src.core.config import settings

    token = jwt.encode(
        {
            "sub": "test-user-123",
            "role": "customer",
            "type": "access",
            "exp": int(time.time()) + 3600,
        },
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )
    return {"Authorization": f"Bearer {token}"}
