from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.models import User


# Keys we persist from the birth details payload.
_BIRTH_DETAIL_KEYS = (
    "name", "latitude", "longitude", "birth_datetime",
    "timezone_str", "ayanamsha", "house_system",
)


class UserRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def get_by_external_id(self, external_user_id: str) -> User | None:
        statement = select(User).where(User.external_user_id == external_user_id)
        return self.db.execute(statement).scalar_one_or_none()

    def get_or_create(
        self,
        external_user_id: str,
        *,
        email: str | None = None,
        phone: str | None = None,
        full_name: str | None = None,
        role: str = "customer",
        preferred_language: str = "en",
    ) -> User:
        user = self.get_by_external_id(external_user_id)
        if user is not None:
            return user
        user = User(
            external_user_id=external_user_id,
            email=email,
            phone=phone,
            full_name=full_name,
            role=role,
            preferred_language=preferred_language,
        )
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user

    def update_language(self, external_user_id: str, language: str) -> User | None:
        user = self.get_by_external_id(external_user_id)
        if user is None:
            return None
        user.preferred_language = language
        self.db.commit()
        self.db.refresh(user)
        return user

    # ── Birth details persistence ────────────────────────────────

    @staticmethod
    def _sanitize_birth_details(details: dict[str, Any]) -> dict[str, Any]:
        """Keep only the keys we need and convert birth_datetime to string."""
        sanitized: dict[str, Any] = {}
        for key in _BIRTH_DETAIL_KEYS:
            if key in details:
                value = details[key]
                # Ensure birth_datetime is stored as ISO string
                if key == "birth_datetime" and hasattr(value, "isoformat"):
                    value = value.isoformat()
                sanitized[key] = value
        return sanitized

    def save_birth_details(
        self,
        external_user_id: str,
        birth_details: dict[str, Any],
    ) -> bool:
        """Persist birth details for an authenticated user.

        Returns True if saved successfully, False if user not found.
        """
        user = self.get_by_external_id(external_user_id)
        if user is None:
            return False
        user.birth_details_json = self._sanitize_birth_details(birth_details)
        self.db.commit()
        self.db.refresh(user)
        return True

    def get_birth_details(self, external_user_id: str) -> dict[str, Any] | None:
        """Load stored birth details for a user.

        Returns the birth details dict or None if not stored.
        """
        user = self.get_by_external_id(external_user_id)
        if user is None or not user.birth_details_json:
            return None
        return dict(user.birth_details_json)

    def get_user_by_id(self, user_id: int) -> User | None:
        """Load a user by internal integer ID."""
        statement = select(User).where(User.id == user_id)
        return self.db.execute(statement).scalar_one_or_none()
