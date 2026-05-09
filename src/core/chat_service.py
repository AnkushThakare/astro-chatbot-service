from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from sqlalchemy.orm import Session

from src.auth.jwt import AuthenticatedUser
from src.astro.kundli import compute_full_chart
from src.core.emotion import detect_emotion
from src.core.intent import IntentClassifier, IntentResult
from src.core.llm import GroqClient
from src.core.memory import MemoryService
from src.core.persona import build_persona_prompt
from src.core.rag import RAGService
from src.core.router import pick_model_route
from src.core.streaming import chunk_text
from src.tools.recommend_product import recommend_product
from src.tools.show_kundali import show_kundali
from src.tools.suggest_consultant import suggest_consultant


class ChatService:
    def __init__(self, db: Session, settings: Any) -> None:
        self.db = db
        self.settings = settings
        self.memory_service = MemoryService(db)
        self.rag_service = RAGService()
        self.intent_classifier = IntentClassifier()
        self.groq_client = GroqClient(settings)

    @staticmethod
    def _format_retrieval_context(matches: list[dict[str, Any]]) -> str:
        if not matches:
            return "No retrieved astrology notes were matched."
        return "\n".join(f"- {match['title']}: {match['excerpt']}" for match in matches)

    @staticmethod
    def _format_tool_context(tool_outputs: list[dict[str, Any]]) -> str:
        if not tool_outputs:
            return "No tool output used."
        lines: list[str] = []
        for output in tool_outputs:
            lines.append(f"Tool: {output['tool']}")
            lines.append(output["summary"])
        return "\n".join(lines)

    @staticmethod
    def _build_local_reply(
        intent: IntentResult,
        emotion_label: str,
        kundali_summary: str | None,
        retrieval_matches: list[dict[str, Any]],
        tool_outputs: list[dict[str, Any]],
    ) -> str:
        sections = [
            "Groq is not configured, so this is the local fallback response.",
            f"Detected intent: {intent.name} (confidence {intent.confidence:.2f}).",
            f"Detected emotional tone: {emotion_label}.",
        ]
        if kundali_summary:
            sections.append(f"Kundali summary: {kundali_summary}")
        if tool_outputs:
            sections.append(ChatService._format_tool_context(tool_outputs))
        if retrieval_matches:
            sections.append(
                "Relevant knowledge:\n" + ChatService._format_retrieval_context(retrieval_matches)
            )
        sections.append(
            "Connect a GROQ_API_KEY to replace this deterministic fallback with a live LLM answer."
        )
        return "\n\n".join(sections)

    async def _prepare_reply_context(
        self,
        *,
        session_id: str,
        message: str,
        birth_details: dict[str, Any] | None = None,
        current_user: AuthenticatedUser | None = None,
    ) -> dict[str, Any]:
        intent = self.intent_classifier.classify(message)
        route = pick_model_route(intent)
        emotion = detect_emotion(message)

        if birth_details is not None:
            self.memory_service.remember_birth_details(session_id, birth_details)

        recent_messages = self.memory_service.recent_messages(
            session_id,
            self.settings.MEMORY_WINDOW_SIZE,
        )
        long_term_context = self.memory_service.long_term_context(session_id)
        retrieval_matches = self.rag_service.retrieve(message, self.settings.RAG_TOP_K)

        kundali_summary: str | None = None
        kundali_chart: dict[str, Any] | None = None
        if birth_details is not None or intent.name == "show_kundali":
            if birth_details:
                kundali_chart = await compute_full_chart(birth_details)
                kundali_summary = show_kundali(kundali_chart)

        tool_outputs: list[dict[str, Any]] = []
        if intent.name == "recommend_product":
            tool_outputs.append(recommend_product(message, kundali_summary=kundali_summary))
        if intent.name == "suggest_consultant":
            tool_outputs.append(suggest_consultant(message, kundali_summary=kundali_summary))
        if kundali_summary:
            tool_outputs.append({"tool": "show_kundali", "summary": kundali_summary})

        persona_prompt = build_persona_prompt(
            long_term_context=long_term_context,
            retrieval_context=self._format_retrieval_context(retrieval_matches),
            tool_context=self._format_tool_context(tool_outputs),
        )
        messages = [{"role": "system", "content": persona_prompt}]
        messages.extend(recent_messages)
        messages.append({"role": "user", "content": message})

        metadata_json = None
        if current_user is not None:
            metadata_json = str({"user_id": current_user.user_id, "role": current_user.role})

        return {
            "intent": intent,
            "route": route,
            "emotion": emotion,
            "messages": messages,
            "tool_outputs": tool_outputs,
            "retrieval_matches": retrieval_matches,
            "kundali_chart": kundali_chart,
            "kundali_summary": kundali_summary,
            "metadata_json": metadata_json,
            "message": message,
            "session_id": session_id,
            "current_user": current_user,
        }

    def _persist_chat_turns(self, context: dict[str, Any], reply: str) -> None:
        intent = context["intent"]
        route = context["route"]

        self.memory_service.repository.add_turn(
            context["session_id"],
            "user",
            context["message"],
            provider="client",
            intent=intent.name,
            metadata_json=context["metadata_json"],
        )
        self.memory_service.repository.add_turn(
            context["session_id"],
            "assistant",
            reply,
            provider=route.provider,
            model=route.model,
            intent=intent.name,
            metadata_json=str({"reasoning_profile": route.reasoning_profile}),
        )

    async def generate_reply(
        self,
        *,
        session_id: str,
        message: str,
        birth_details: dict[str, Any] | None = None,
        current_user: AuthenticatedUser | None = None,
    ) -> dict[str, Any]:
        context = await self._prepare_reply_context(
            session_id=session_id,
            message=message,
            birth_details=birth_details,
            current_user=current_user,
        )
        intent = context["intent"]
        route = context["route"]

        if self.groq_client.is_configured:
            reply = await self.groq_client.generate(
                context["messages"],
                session_id=session_id,
                user_id=current_user.user_id if current_user is not None else None,
            )
        else:
            reply = self._build_local_reply(
                intent,
                context["emotion"].label,
                context["kundali_summary"],
                context["retrieval_matches"],
                context["tool_outputs"],
            )

        self._persist_chat_turns(context, reply)

        return {
            "reply": reply,
            "intent": intent.name,
            "tool_outputs": context["tool_outputs"],
            "retrieval_matches": context["retrieval_matches"],
            "kundali_chart": context["kundali_chart"],
            "kundali_summary": context["kundali_summary"],
            "emotion": context["emotion"].label,
            "route": route,
        }

    async def stream_reply_events(
        self,
        *,
        session_id: str,
        message: str,
        birth_details: dict[str, Any] | None = None,
        current_user: AuthenticatedUser | None = None,
    ) -> AsyncGenerator[tuple[str, dict[str, Any]], None]:
        context = await self._prepare_reply_context(
            session_id=session_id,
            message=message,
            birth_details=birth_details,
            current_user=current_user,
        )
        intent = context["intent"]

        yield (
            "meta",
            {
                "resolved_session_id": session_id,
                "intent": intent.name,
                "tool_count": len(context["tool_outputs"]),
            },
        )

        reply_parts: list[str] = []
        if self.groq_client.is_configured:
            async for delta in self.groq_client.stream_generate(
                context["messages"],
                session_id=session_id,
                user_id=current_user.user_id if current_user is not None else None,
            ):
                reply_parts.append(delta)
                yield ("message", {"delta": delta})
        else:
            fallback_reply = self._build_local_reply(
                intent,
                context["emotion"].label,
                context["kundali_summary"],
                context["retrieval_matches"],
                context["tool_outputs"],
            )
            reply_parts.append(fallback_reply)
            for chunk in chunk_text(fallback_reply):
                yield ("message", {"delta": chunk})

        reply = "".join(reply_parts)
        self._persist_chat_turns(context, reply)

        yield (
            "done",
            {
                "resolved_session_id": session_id,
                "reply": reply,
                "intent": intent.name,
            },
        )
