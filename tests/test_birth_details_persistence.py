"""Tests for birth details persistence in local DB."""
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.db.models import Base, User
from src.db.repositories.users import UserRepository


def _make_session() -> Session:
    """Create an in-memory SQLite session with tables."""
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    return factory()


def _create_user(db: Session, external_id: str = "ext-123", name: str = "Test User") -> User:
    repo = UserRepository(db)
    return repo.get_or_create(external_id, full_name=name)


# ── Model column exists ─────────────────────────────────────────

def test_user_model_has_birth_details_column():
    db = _make_session()
    user = _create_user(db)
    assert user.birth_details_json is None


# ── Save and retrieve ────────────────────────────────────────────

def test_save_birth_details():
    db = _make_session()
    _create_user(db)
    repo = UserRepository(db)

    details = {
        "name": "Ankush",
        "latitude": 28.6139,
        "longitude": 77.209,
        "birth_datetime": "1990-05-15T10:30:00",
        "ayanamsha": "LAHIRI",
        "house_system": "W",
    }
    result = repo.save_birth_details("ext-123", details)
    assert result is True

    loaded = repo.get_birth_details("ext-123")
    assert loaded is not None
    assert loaded["latitude"] == 28.6139
    assert loaded["longitude"] == 77.209
    assert loaded["birth_datetime"] == "1990-05-15T10:30:00"
    assert loaded["ayanamsha"] == "LAHIRI"


def test_save_birth_details_overwrites_previous():
    db = _make_session()
    _create_user(db)
    repo = UserRepository(db)

    repo.save_birth_details("ext-123", {
        "latitude": 28.0,
        "longitude": 77.0,
        "birth_datetime": "1990-01-01T00:00:00",
    })
    repo.save_birth_details("ext-123", {
        "latitude": 19.076,
        "longitude": 72.8777,
        "birth_datetime": "1995-06-20T14:00:00",
    })

    loaded = repo.get_birth_details("ext-123")
    assert loaded is not None
    assert loaded["latitude"] == 19.076
    assert loaded["birth_datetime"] == "1995-06-20T14:00:00"


def test_save_birth_details_user_not_found():
    db = _make_session()
    repo = UserRepository(db)
    result = repo.save_birth_details("nonexistent", {"latitude": 0, "longitude": 0})
    assert result is False


def test_get_birth_details_user_not_found():
    db = _make_session()
    repo = UserRepository(db)
    assert repo.get_birth_details("nonexistent") is None


def test_get_birth_details_not_set():
    db = _make_session()
    _create_user(db)
    repo = UserRepository(db)
    assert repo.get_birth_details("ext-123") is None


# ── Sanitization ─────────────────────────────────────────────────

def test_save_strips_extra_keys():
    db = _make_session()
    _create_user(db)
    repo = UserRepository(db)

    details = {
        "latitude": 28.6139,
        "longitude": 77.209,
        "birth_datetime": "1990-05-15T10:30:00",
        "some_random_key": "should not be stored",
        "password": "definitely not",
    }
    repo.save_birth_details("ext-123", details)
    loaded = repo.get_birth_details("ext-123")
    assert loaded is not None
    assert "some_random_key" not in loaded
    assert "password" not in loaded
    assert loaded["latitude"] == 28.6139


def test_save_converts_datetime_object_to_string():
    db = _make_session()
    _create_user(db)
    repo = UserRepository(db)

    details = {
        "latitude": 28.6139,
        "longitude": 77.209,
        "birth_datetime": datetime(1990, 5, 15, 10, 30),
    }
    repo.save_birth_details("ext-123", details)
    loaded = repo.get_birth_details("ext-123")
    assert loaded is not None
    assert isinstance(loaded["birth_datetime"], str)
    assert "1990-05-15" in loaded["birth_datetime"]


# ── get_user_by_id ───────────────────────────────────────────────

def test_get_user_by_id():
    db = _make_session()
    user = _create_user(db)
    repo = UserRepository(db)
    found = repo.get_user_by_id(user.id)
    assert found is not None
    assert found.external_user_id == "ext-123"


def test_get_user_by_id_not_found():
    db = _make_session()
    repo = UserRepository(db)
    assert repo.get_user_by_id(9999) is None
