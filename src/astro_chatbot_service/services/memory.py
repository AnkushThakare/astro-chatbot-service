from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from astro_chatbot_service.models.database import ConversationTurn
from astro_chatbot_service.models.schemas import MemoryTurnRead


class MemoryService:
    def __init__(self, db: Session) -> None:
        self.db = db

    def add_turn(self, session_id: str, role: str, content: str) -> ConversationTurn:
        turn = ConversationTurn(session_id=session_id, role=role, content=content)
        self.db.add(turn)
        self.db.commit()
        self.db.refresh(turn)
        return turn

    def list_session(self, session_id: str, limit: int | None = None) -> list[MemoryTurnRead]:
        statement = (
            select(ConversationTurn)
            .where(ConversationTurn.session_id == session_id)
            .order_by(ConversationTurn.created_at.asc(), ConversationTurn.id.asc())
        )
        if limit:
            statement = statement.limit(limit)
        turns = self.db.execute(statement).scalars().all()
        return [
            MemoryTurnRead(
                id=turn.id,
                role=turn.role,
                content=turn.content,
                created_at=turn.created_at,
            )
            for turn in turns
        ]

    def recent_for_prompt(self, session_id: str, window_size: int) -> list[dict[str, str]]:
        statement = (
            select(ConversationTurn)
            .where(ConversationTurn.session_id == session_id)
            .order_by(ConversationTurn.created_at.desc(), ConversationTurn.id.desc())
            .limit(window_size)
        )
        turns = list(reversed(self.db.execute(statement).scalars().all()))
        return [{"role": turn.role, "content": turn.content} for turn in turns]

