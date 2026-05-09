from __future__ import annotations

from collections import defaultdict
import json

from sqlalchemy import select
from sqlalchemy.orm import Session

from astro_chatbot_service.models.database import ConversationTurn
from astro_chatbot_service.models.schemas import FineTuneDatasetRequest, FineTuneJobRequest


class FineTuningService:
    def __init__(self, db: Session) -> None:
        self.db = db

    def build_dataset_payload(self, request: FineTuneDatasetRequest) -> dict[str, object]:
        statement = select(ConversationTurn).order_by(
            ConversationTurn.session_id.asc(),
            ConversationTurn.created_at.asc(),
            ConversationTurn.id.asc(),
        )
        if request.session_ids:
            statement = statement.where(ConversationTurn.session_id.in_(request.session_ids))

        rows = self.db.execute(statement).scalars().all()
        grouped: dict[str, list[ConversationTurn]] = defaultdict(list)
        for row in rows:
            grouped[row.session_id].append(row)

        sessions = list(grouped.items())[: request.limit_sessions]
        system_prompt = request.system_prompt or "You are an astrology assistant."
        dataset = []
        for session_id, turns in sessions:
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend({"role": turn.role, "content": turn.content} for turn in turns)
            dataset.append({"session_id": session_id, "messages": messages})

        jsonl_lines = [json.dumps(item) for item in dataset]
        return {
            "session_count": len(dataset),
            "dataset_preview": dataset[:3],
            "jsonl_lines": jsonl_lines,
        }

    def build_job_payload(self, request: FineTuneJobRequest) -> dict[str, object]:
        return {
            "provider": "groq-compatible",
            "training_file": request.training_file,
            "model": request.base_model,
            "suffix": request.suffix,
            "hyperparameters": request.hyperparameters,
        }
