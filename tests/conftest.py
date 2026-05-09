from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

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
