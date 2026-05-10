from __future__ import annotations

import json

from sqlalchemy.orm import Session

from src.core.config import settings
from src.core.logging import get_logger
from src.db.repositories.conversations import ConversationRepository

logger = get_logger(__name__)


class MemoryService:
    def __init__(self, db: Session) -> None:
        self.repository = ConversationRepository(db)

    def add_turn(self, session_id: str, role: str, content: str) -> None:
        self.repository.add_turn(session_id, role, content)

    def recent_messages(self, session_id: str, limit: int) -> list[dict[str, str]]:
        rows = self.repository.list_recent_turns(session_id, limit)
        return [{"role": row.role, "content": row.content} for row in rows]

    def remember_birth_details(
        self,
        session_id: str,
        payload: dict[str, str],
        *,
        user_id: int | None = None,
    ) -> None:
        self.repository.upsert_fact(
            session_id=session_id,
            fact_key="birth_details",
            fact_value=json.dumps(payload),
            user_id=user_id,
        )

    def long_term_context(
        self,
        session_id: str,
        *,
        user_id: int | None = None,
    ) -> str | None:
        """Load facts for this conversation.

        When *user_id* is provided, loads ALL facts the user has ever told
        us — across every session — so memory persists like ChatGPT.
        Falls back to session-scoped facts for anonymous users.
        """
        if user_id is not None:
            facts = self.repository.list_facts_for_user(user_id)
        else:
            facts = self.repository.list_facts(session_id)
        if not facts:
            return None
        # Deduplicate by fact_key (keep the most recently updated one)
        seen: set[str] = set()
        unique_facts: list[str] = []
        for fact in facts:
            if fact.fact_key not in seen:
                seen.add(fact.fact_key)
                unique_facts.append(f"- {fact.fact_key}: {fact.fact_value}")
        return "\n".join(unique_facts) if unique_facts else None

    async def extract_and_store_facts(
        self,
        session_id: str,
        groq_client: "GroqClient",  # noqa: F821
        *,
        user_id: int | None = None,
    ) -> list[dict[str, str]]:
        """Extract facts from recent conversation and store them.

        Uses the memory_extractor prompt to mine the conversation for
        user-stated facts worth remembering across sessions.
        When *user_id* is provided, facts are stored at the user level
        so they persist across all future sessions.
        """
        recent = self.recent_messages(session_id, limit=20)
        if len(recent) < 2:
            return []

        conversation_text = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}" for msg in recent
        )

        prompt_path = settings.prompts_dir / "memory_extractor.txt"
        if not prompt_path.exists():
            logger.warning("memory_extractor.txt not found, skipping extraction")
            return []

        prompt_template = prompt_path.read_text(encoding="utf-8").strip()
        prompt = prompt_template.replace("{conversation}", conversation_text)

        try:
            raw = await groq_client.generate(
                [{"role": "user", "content": prompt}],
                model=settings.GROQ_PLANNER_MODEL,
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            facts = json.loads(raw)
            if isinstance(facts, dict):
                facts = facts.get("facts", [])
            if not isinstance(facts, list):
                return []
        except Exception as exc:
            logger.warning("Memory extraction failed: %s", exc)
            return []

        stored: list[dict[str, str]] = []
        for item in facts:
            if not isinstance(item, dict):
                continue
            key = item.get("fact_key", "").strip()
            value = item.get("fact_value", "").strip()
            if key and value and key != "birth_details":
                self.repository.upsert_fact(
                    session_id=session_id,
                    fact_key=key,
                    fact_value=value,
                    user_id=user_id,
                )
                stored.append({"fact_key": key, "fact_value": value})

        if stored:
            logger.info("Extracted %d facts for session %s", len(stored), session_id)
        return stored

    async def summarize_conversation(
        self,
        session_id: str,
        groq_client: "GroqClient",  # noqa: F821
    ) -> str | None:
        """Generate and store a conversation summary."""
        recent = self.recent_messages(session_id, limit=30)
        if len(recent) < 2:
            return None

        conversation_text = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}" for msg in recent
        )

        prompt_path = settings.prompts_dir / "conversation_summarizer.txt"
        if not prompt_path.exists():
            logger.warning("conversation_summarizer.txt not found, skipping summary")
            return None

        prompt_template = prompt_path.read_text(encoding="utf-8").strip()
        prompt = prompt_template.replace("{conversation}", conversation_text)

        try:
            summary = await groq_client.generate(
                [{"role": "user", "content": prompt}],
                model=settings.GROQ_PLANNER_MODEL,
                temperature=0.2,
            )
        except Exception as exc:
            logger.warning("Conversation summarization failed: %s", exc)
            return None

        summary = summary.strip()
        if summary:
            self.repository.update_conversation_summary(session_id, summary)
            logger.info("Stored summary for session %s", session_id)
        return summary
