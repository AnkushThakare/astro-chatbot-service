from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.models import User


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
