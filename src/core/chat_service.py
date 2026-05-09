from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

from sqlalchemy.orm import Session

from src.auth.jwt import AuthenticatedUser
from src.astro.kundli import compute_full_chart
from src.core.core_service import CoreServiceClient
from src.core.emotion import detect_emotion
from src.core.intent import IntentClassifier, IntentResult
from src.core.llm import GroqClient
from src.core.memory import MemoryService
from src.core.persona import build_persona_prompt
from src.core.rag import RAGService
from src.core.router import pick_model_route
from src.core.streaming import chunk_text
from src.tools.show_matchmaking import show_matchmaking
from src.tools.show_kundali import show_kundali


class ChatService:
    def __init__(self, db: Session, settings: Any) -> None:
        self.db = db
        self.settings = settings
        self.memory_service = MemoryService(db)
        self.rag_service = RAGService()
        self.intent_classifier = IntentClassifier()
        self.groq_client = GroqClient(settings)
        self.core_service_client = CoreServiceClient(settings)

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

    @staticmethod
    def _build_product_tool_output(
        products: list[dict[str, Any]],
        kundali_summary: str | None = None,
    ) -> dict[str, Any] | None:
        if not products:
            return None

        items: list[dict[str, Any]] = []
        names: list[str] = []
        for product in products[:3]:
            name = str(product.get("name") or "Product")
            names.append(name)
            primary_media = product.get("primary_media") or {}
            items.append(
                {
                    "id": str(product.get("id", "")),
                    "slug": str(product.get("slug", "")),
                    "name": name,
                    "starting_price_paise": product.get("starting_price_paise"),
                    "image_url": primary_media.get("url") if isinstance(primary_media, dict) else None,
                }
            )

        summary = "Relevant products from the Digveda catalog: " + ", ".join(names) + "."
        if kundali_summary:
            summary += f" Kundali context considered: {kundali_summary}"

        return {
            "tool": "recommend_product",
            "event_name": "suggestion_product",
            "summary": summary,
            "items": items,
            "source": "core-service",
        }

    @staticmethod
    def _build_consultant_tool_output(
        consultants: list[dict[str, Any]],
        kundali_summary: str | None = None,
    ) -> dict[str, Any] | None:
        if not consultants:
            return None

        items: list[dict[str, Any]] = []
        names: list[str] = []
        for consultant in consultants[:3]:
            name = str(consultant.get("name") or "Pandit")
            names.append(name)
            items.append(
                {
                    "id": str(consultant.get("id", "")),
                    "provider_handle": str(consultant.get("provider_handle", "")),
                    "name": name,
                    "specialties": consultant.get("specialties"),
                    "languages": consultant.get("languages"),
                    "consultation_fee_per_min": consultant.get("consultation_fee_per_min"),
                    "default_photo_url": consultant.get("default_photo_url"),
                }
            )

        summary = "Relevant pandits from the Digveda network: " + ", ".join(names) + "."
        if kundali_summary:
            summary += f" Kundali context considered: {kundali_summary}"

        return {
            "tool": "suggest_consultant",
            "event_name": "suggestion_consultant",
            "summary": summary,
            "items": items,
            "source": "core-service",
        }

    @staticmethod
    def _build_matchmaking_tool_output(matchmaking: dict[str, Any]) -> dict[str, Any] | None:
        if not matchmaking:
            return None

        return {
            "tool": "matchmaking",
            "event_name": "suggestion_matchmaking",
            "summary": show_matchmaking(matchmaking),
            "matchmaking": matchmaking,
            "source": "core-service",
        }

    @staticmethod
    def _build_booking_tool_output(
        home_puja_services: list[dict[str, Any]],
        temple_services: list[dict[str, Any]],
        pandits: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if not home_puja_services and not temple_services and not pandits:
            return None

        home_items = [
            {
                "id": str(service.get("id", "")),
                "name": str(service.get("name") or "Service"),
                "description": service.get("description"),
                "price_range_min_rupees": service.get("price_range_min_rupees"),
                "price_range_max_rupees": service.get("price_range_max_rupees"),
                "tiers": service.get("tiers"),
                "images": service.get("images"),
            }
            for service in home_puja_services[:3]
        ]
        temple_items = [
            {
                "id": str(service.get("id", "")),
                "name": str(service.get("name") or "Temple Service"),
                "description": service.get("description"),
                "service_mode": service.get("service_mode"),
                "temple": service.get("temple"),
                "min_price_paise": service.get("min_price_paise"),
                "max_price_paise": service.get("max_price_paise"),
                "tiers": service.get("tiers"),
                "images": service.get("images"),
                "primary_image_variants": service.get("primary_image_variants"),
            }
            for service in temple_services[:3]
        ]
        pandit_items = [
            {
                "id": str(pandit.get("id", "")),
                "name": str(pandit.get("name") or "Pandit"),
                "provider_handle": pandit.get("provider_handle"),
                "photo_url": pandit.get("photo_url"),
                "experience_years": pandit.get("experience_years"),
                "languages": pandit.get("languages"),
                "specialties": pandit.get("specialties"),
                "bio": pandit.get("bio"),
                "average_rating": pandit.get("average_rating"),
                "total_reviews": pandit.get("total_reviews"),
                "offered_services": pandit.get("offered_services"),
                "city": pandit.get("city"),
                "state": pandit.get("state"),
            }
            for pandit in pandits[:3]
        ]

        summary_parts: list[str] = []
        if home_items:
            summary_parts.append(
                "Home puja services: " + ", ".join(item["name"] for item in home_items) + "."
            )
        if temple_items:
            summary_parts.append(
                "Temple services: " + ", ".join(item["name"] for item in temple_items) + "."
            )
        if pandit_items:
            summary_parts.append(
                "Available pandits: " + ", ".join(item["name"] for item in pandit_items) + "."
            )

        return {
            "tool": "book_pooja",
            "event_name": "suggestion_booking",
            "summary": " ".join(summary_parts) if summary_parts else "Booking suggestions are available.",
            "home_puja_services": home_items,
            "temple_services": temple_items,
            "pandits": pandit_items,
            "source": "core-service",
        }

    async def _prepare_reply_context(
        self,
        *,
        session_id: str,
        message: str,
        birth_details: dict[str, Any] | None = None,
        matchmaking_details: dict[str, Any] | None = None,
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
                kundali_chart = await self.core_service_client.generate_kundli(
                    birth_details,
                    current_user,
                )
                if kundali_chart is None:
                    kundali_chart = await compute_full_chart(birth_details)
                kundali_summary = show_kundali(kundali_chart)

        matchmaking_result: dict[str, Any] | None = None
        if matchmaking_details is not None:
            matchmaking_result = await self.core_service_client.generate_matchmaking(
                matchmaking_details,
                current_user,
            )

        tool_outputs: list[dict[str, Any]] = []
        matchmaking_output = self._build_matchmaking_tool_output(matchmaking_result or {})
        if matchmaking_output is not None:
            tool_outputs.append(matchmaking_output)

        if intent.name == "book_pooja":
            home_puja_services, temple_services, public_pandits = await asyncio.gather(
                self.core_service_client.list_home_puja_services(message),
                self.core_service_client.list_temple_services(message),
                self.core_service_client.list_public_pandits(message),
            )
            booking_output = self._build_booking_tool_output(
                home_puja_services,
                temple_services,
                public_pandits,
            )
            if booking_output is not None:
                tool_outputs.append(booking_output)
        if intent.name == "recommend_product":
            product_output = self._build_product_tool_output(
                await self.core_service_client.search_products(message),
                kundali_summary=kundali_summary,
            )
            if product_output is not None:
                tool_outputs.append(product_output)
        if intent.name == "suggest_consultant":
            consultant_output = self._build_consultant_tool_output(
                await self.core_service_client.search_pandits(message, current_user),
                kundali_summary=kundali_summary,
            )
            if consultant_output is not None:
                tool_outputs.append(consultant_output)
        if kundali_summary:
            tool_outputs.append(
                {
                    "tool": "show_kundali",
                    "event_name": "suggestion_kundali",
                    "summary": kundali_summary,
                    "chart": kundali_chart,
                }
            )

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
            "matchmaking_result": matchmaking_result,
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
        matchmaking_details: dict[str, Any] | None = None,
        current_user: AuthenticatedUser | None = None,
    ) -> dict[str, Any]:
        context = await self._prepare_reply_context(
            session_id=session_id,
            message=message,
            birth_details=birth_details,
            matchmaking_details=matchmaking_details,
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
            "matchmaking_result": context["matchmaking_result"],
            "emotion": context["emotion"].label,
            "route": route,
        }

    async def stream_reply_events(
        self,
        *,
        session_id: str,
        message: str,
        birth_details: dict[str, Any] | None = None,
        matchmaking_details: dict[str, Any] | None = None,
        current_user: AuthenticatedUser | None = None,
    ) -> AsyncGenerator[tuple[str, dict[str, Any]], None]:
        context = await self._prepare_reply_context(
            session_id=session_id,
            message=message,
            birth_details=birth_details,
            matchmaking_details=matchmaking_details,
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

        for output in context["tool_outputs"]:
            event_name = output.get("event_name")
            if not isinstance(event_name, str):
                continue

            payload = {
                key: value
                for key, value in output.items()
                if key not in {"tool", "event_name"}
            }
            yield (event_name, payload)

        yield (
            "done",
            {
                "resolved_session_id": session_id,
                "reply": reply,
                "intent": intent.name,
            },
        )
