from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.models import Conversation, Memory, Message


class ConversationRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def get_or_create_conversation(
        self,
        session_id: str,
        user_id: int | None = None,
    ) -> Conversation:
        statement = select(Conversation).where(Conversation.session_id == session_id)
        conversation = self.db.execute(statement).scalar_one_or_none()
        if conversation is None:
            conversation = Conversation(session_id=session_id, user_id=user_id)
            self.db.add(conversation)
            self.db.commit()
            self.db.refresh(conversation)
        elif user_id is not None and conversation.user_id is None:
            conversation.user_id = user_id
            self.db.commit()
            self.db.refresh(conversation)
        return conversation

    def add_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        intent: str | None = None,
        metadata_json: str | None = None,
        user_id: int | None = None,
    ) -> Message:
        conversation = self.get_or_create_conversation(session_id, user_id=user_id)
        row = Message(
            conversation_id=conversation.id,
            role=role,
            content=content,
            provider=provider,
            model=model,
            intent=intent,
            metadata_json=metadata_json,
        )
        self.db.add(row)
        self.db.commit()
        self.db.refresh(row)
        return row

    def list_recent_turns(self, session_id: str, limit: int) -> list[Message]:
        statement = (
            select(Message)
            .join(Conversation, Conversation.id == Message.conversation_id)
            .where(Conversation.session_id == session_id)
            .order_by(Message.created_at.desc(), Message.id.desc())
            .limit(limit)
        )
        rows = self.db.execute(statement).scalars().all()
        return list(reversed(rows))

    def upsert_fact(
        self,
        session_id: str,
        fact_key: str,
        fact_value: str,
        *,
        user_id: int | None = None,
    ) -> Memory:
        conversation = self.get_or_create_conversation(session_id, user_id=user_id)

        # When user is known, deduplicate by user_id + fact_key so the same
        # fact is shared across all of the user's sessions (like ChatGPT memory).
        if user_id is not None:
            statement = select(Memory).where(
                Memory.user_id == user_id,
                Memory.fact_key == fact_key,
            )
        else:
            statement = select(Memory).where(
                Memory.session_id == session_id,
                Memory.fact_key == fact_key,
            )
        row = self.db.execute(statement).scalar_one_or_none()
        if row is None:
            row = Memory(
                session_id=session_id,
                conversation_id=conversation.id,
                user_id=user_id,
                fact_key=fact_key,
                fact_value=fact_value,
            )
            self.db.add(row)
        else:
            row.fact_value = fact_value
            if user_id is not None:
                row.user_id = user_id
            row.session_id = session_id
            row.conversation_id = conversation.id
        self.db.commit()
        self.db.refresh(row)
        return row

    def list_facts(self, session_id: str) -> list[Memory]:
        statement = (
            select(Memory)
            .where(Memory.session_id == session_id)
            .order_by(Memory.updated_at.desc(), Memory.id.desc())
        )
        return self.db.execute(statement).scalars().all()

    def list_facts_for_user(self, user_id: int) -> list[Memory]:
        """Load ALL facts across every session for this user.

        This is what powers cross-session memory — when the user told us
        their DOB in session A, we recall it in session B.
        """
        statement = (
            select(Memory)
            .where(Memory.user_id == user_id)
            .order_by(Memory.updated_at.desc(), Memory.id.desc())
        )
        return self.db.execute(statement).scalars().all()

    def update_conversation_summary(self, session_id: str, summary: str) -> None:
        statement = select(Conversation).where(Conversation.session_id == session_id)
        conversation = self.db.execute(statement).scalar_one_or_none()
        if conversation is not None:
            conversation.summary = summary
            self.db.commit()
