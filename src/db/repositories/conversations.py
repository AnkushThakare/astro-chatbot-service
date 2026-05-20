from __future__ import annotations

import json

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.models import Conversation, Memory, Message


class ConversationRepository:
    SESSION_STATE_FACT_KEY = "__session_state__"

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
        prompt_versions: dict | None = None,
        model_used: str | None = None,
        route_taken: str | None = None,
        tool_called: str | None = None,
        variant_id: str | None = None,
        total_tokens_input: int | None = None,
        total_tokens_output: int | None = None,
        latency_ms: int | None = None,
        partial: bool = False,
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
            prompt_versions=prompt_versions,
            model_used=model_used,
            route_taken=route_taken,
            tool_called=tool_called,
            variant_id=variant_id,
            total_tokens_input=total_tokens_input,
            total_tokens_output=total_tokens_output,
            latency_ms=latency_ms,
            partial=partial,
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
            .where(
                Memory.session_id == session_id,
                Memory.fact_key != self.SESSION_STATE_FACT_KEY,
            )
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
            .where(
                Memory.user_id == user_id,
                Memory.fact_key != self.SESSION_STATE_FACT_KEY,
            )
            .order_by(Memory.updated_at.desc(), Memory.id.desc())
        )
        return self.db.execute(statement).scalars().all()

    def get_session_state(self, session_id: str) -> dict | None:
        statement = select(Memory).where(
            Memory.session_id == session_id,
            Memory.fact_key == self.SESSION_STATE_FACT_KEY,
        )
        row = self.db.execute(statement).scalar_one_or_none()
        if row is None or not row.fact_value:
            return None
        try:
            payload = json.loads(row.fact_value)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    def save_session_state(self, session_id: str, state: dict, *, user_id: int | None = None) -> Memory:
        conversation = self.get_or_create_conversation(session_id, user_id=user_id)
        statement = select(Memory).where(
            Memory.session_id == session_id,
            Memory.fact_key == self.SESSION_STATE_FACT_KEY,
        )
        row = self.db.execute(statement).scalar_one_or_none()
        serialized = json.dumps(state, ensure_ascii=True, default=str)
        if row is None:
            row = Memory(
                session_id=session_id,
                conversation_id=conversation.id,
                user_id=None,
                fact_key=self.SESSION_STATE_FACT_KEY,
                fact_value=serialized,
            )
            self.db.add(row)
        else:
            row.fact_value = serialized
            row.session_id = session_id
            row.conversation_id = conversation.id
        self.db.commit()
        self.db.refresh(row)
        return row

    def merge_session_state(
        self,
        session_id: str,
        updates: dict,
        *,
        user_id: int | None = None,
    ) -> Memory:
        current = self.get_session_state(session_id) or {}
        current.update(updates)
        return self.save_session_state(session_id, current, user_id=user_id)

    def update_conversation_summary(self, session_id: str, summary: str) -> None:
        statement = select(Conversation).where(Conversation.session_id == session_id)
        conversation = self.db.execute(statement).scalar_one_or_none()
        if conversation is not None:
            conversation.summary = summary
            self.db.commit()
