from __future__ import annotations

import json

from sqlalchemy.orm import Session

from src.db.repositories.conversations import ConversationRepository


class MemoryService:
    def __init__(self, db: Session) -> None:
        self.repository = ConversationRepository(db)

    def add_turn(self, session_id: str, role: str, content: str) -> None:
        self.repository.add_turn(session_id, role, content)

    def recent_messages(self, session_id: str, limit: int) -> list[dict[str, str]]:
        rows = self.repository.list_recent_turns(session_id, limit)
        return [{"role": row.role, "content": row.content} for row in rows]

    def remember_birth_details(self, session_id: str, payload: dict[str, str]) -> None:
        self.repository.upsert_fact(
            session_id=session_id,
            fact_key="birth_details",
            fact_value=json.dumps(payload),
        )

    def long_term_context(self, session_id: str) -> str | None:
        facts = self.repository.list_facts(session_id)
        if not facts:
            return None
        return "\n".join(f"- {fact.fact_key}: {fact.fact_value}" for fact in facts)
