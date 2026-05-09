from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

from sqlalchemy.orm import Session

from src.auth.jwt import AuthenticatedUser
from src.astro.kundli import compute_full_chart
from src.core.core_service import CoreServiceClient
from src.core.emotion import detect_emotion
from src.core.llm import GroqClient
from src.core.memory import MemoryService
from src.core.planner import ConversationPlanner, PlannerResult
from src.core.persona import build_persona_prompt
from src.core.rag import RAGService
from src.core.router import pick_model_route
from src.core.streaming import chunk_text
from src.tools.show_matchmaking import show_matchmaking
from src.tools.show_kundali import show_kundali


class ChatService:
    TOOL_CONFIDENCE_THRESHOLD = 0.75

    def __init__(self, db: Session, settings: Any) -> None:
        self.db = db
        self.settings = settings
        self.memory_service = MemoryService(db)
        self.rag_service = RAGService()
        self.groq_client = GroqClient(settings)
        self.planner = ConversationPlanner(self.groq_client, settings.GROQ_PLANNER_MODEL)
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
        plan: PlannerResult,
        emotion_label: str,
        kundali_summary: str | None,
        retrieval_matches: list[dict[str, Any]],
        tool_outputs: list[dict[str, Any]],
    ) -> str:
        sections = [
            "Groq is not configured, so this is the local fallback response.",
            f"Planned action: {plan.action} (confidence {plan.confidence:.2f}).",
            f"Detected emotional tone: {emotion_label}.",
        ]
        if plan.missing_information:
            sections.append("Missing information: " + ", ".join(plan.missing_information))
        sections.append(f"Planner reasoning: {plan.reasoning}")
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

    @staticmethod
    def _planner_search_query(plan: PlannerResult) -> str | None:
        query = plan.arguments.get("search_query")
        if not isinstance(query, str):
            return None
        normalized = query.strip()
        return normalized or None

    @classmethod
    def _has_required_fields_for_action(
        cls,
        plan: PlannerResult,
        *,
        birth_details: dict[str, Any] | None,
        matchmaking_details: dict[str, Any] | None,
    ) -> bool:
        if plan.action == "show_kundali":
            return birth_details is not None
        if plan.action == "matchmaking":
            return matchmaking_details is not None
        if plan.action in {"book_pooja", "recommend_product", "suggest_consultant"}:
            return cls._planner_search_query(plan) is not None
        return True

    @classmethod
    def _should_execute_tool(
        cls,
        plan: PlannerResult,
        *,
        birth_details: dict[str, Any] | None,
        matchmaking_details: dict[str, Any] | None,
    ) -> bool:
        if not plan.should_call_tool:
            return False
        if plan.confidence <= cls.TOOL_CONFIDENCE_THRESHOLD:
            return False
        if plan.action not in {
            "show_kundali",
            "matchmaking",
            "book_pooja",
            "recommend_product",
            "suggest_consultant",
        }:
            return False
        return cls._has_required_fields_for_action(
            plan,
            birth_details=birth_details,
            matchmaking_details=matchmaking_details,
        )

    @staticmethod
    def _format_planner_context(plan: PlannerResult, tool_allowed: bool) -> str:
        lines = [
            f"Planner action: {plan.action}",
            f"Planner confidence: {plan.confidence:.2f}",
            f"Tool execution allowed: {str(tool_allowed).lower()}",
            f"Planner reasoning: {plan.reasoning}",
        ]
        if plan.arguments:
            lines.append(f"Planner arguments: {plan.arguments}")
        if plan.missing_information:
            lines.append("Missing information: " + ", ".join(plan.missing_information))
        return "\n".join(lines)

    async def _prepare_reply_context(
        self,
        *,
        session_id: str,
        message: str,
        birth_details: dict[str, Any] | None = None,
        matchmaking_details: dict[str, Any] | None = None,
        current_user: AuthenticatedUser | None = None,
    ) -> dict[str, Any]:
        plan = await self.planner.plan(
            message=message,
            has_birth_details=birth_details is not None,
            has_matchmaking_details=matchmaking_details is not None,
            is_authenticated=current_user is not None,
        )
        tool_execution_allowed = self._should_execute_tool(
            plan,
            birth_details=birth_details,
            matchmaking_details=matchmaking_details,
        )
        route = pick_model_route(
            plan.model_copy(update={"should_call_tool": tool_execution_allowed})
        )
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
        if tool_execution_allowed and plan.action == "show_kundali" and birth_details is not None:
            kundali_chart = await self.core_service_client.generate_kundli(
                birth_details,
                current_user,
            )
            if kundali_chart is None:
                kundali_chart = await compute_full_chart(birth_details)
            kundali_summary = show_kundali(kundali_chart)

        matchmaking_result: dict[str, Any] | None = None
        if tool_execution_allowed and plan.action == "matchmaking" and matchmaking_details is not None:
            matchmaking_result = await self.core_service_client.generate_matchmaking(
                matchmaking_details,
                current_user,
            )

        tool_outputs: list[dict[str, Any]] = []
        matchmaking_output = self._build_matchmaking_tool_output(matchmaking_result or {})
        if matchmaking_output is not None:
            tool_outputs.append(matchmaking_output)

        search_query = self._planner_search_query(plan)
        if tool_execution_allowed and plan.action == "book_pooja" and search_query is not None:
            home_puja_services, temple_services, public_pandits = await asyncio.gather(
                self.core_service_client.list_home_puja_services(search_query),
                self.core_service_client.list_temple_services(search_query),
                self.core_service_client.list_public_pandits(search_query),
            )
            booking_output = self._build_booking_tool_output(
                home_puja_services,
                temple_services,
                public_pandits,
            )
            if booking_output is not None:
                tool_outputs.append(booking_output)
        if tool_execution_allowed and plan.action == "recommend_product" and search_query is not None:
            product_output = self._build_product_tool_output(
                await self.core_service_client.search_products(search_query),
                kundali_summary=kundali_summary,
            )
            if product_output is not None:
                tool_outputs.append(product_output)
        if tool_execution_allowed and plan.action == "suggest_consultant" and search_query is not None:
            consultant_output = self._build_consultant_tool_output(
                await self.core_service_client.search_pandits(search_query, current_user),
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
        messages.append(
            {
                "role": "system",
                "content": self._format_planner_context(plan, tool_execution_allowed),
            }
        )
        messages.extend(recent_messages)
        messages.append({"role": "user", "content": message})

        metadata_json = None
        if current_user is not None:
            metadata_json = str({"user_id": current_user.user_id, "role": current_user.role})

        return {
            "plan": plan,
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
        }

    def _persist_chat_turns(self, context: dict[str, Any], reply: str) -> None:
        plan = context["plan"]
        route = context["route"]

        self.memory_service.repository.add_turn(
            context["session_id"],
            "user",
            context["message"],
            provider="client",
            intent=plan.action,
            metadata_json=context["metadata_json"],
        )
        self.memory_service.repository.add_turn(
            context["session_id"],
            "assistant",
            reply,
            provider=route.provider,
            model=route.model,
            intent=plan.action,
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
        plan = context["plan"]
        route = context["route"]

        if self.groq_client.is_configured:
            reply = await self.groq_client.generate(
                context["messages"],
                session_id=session_id,
                user_id=current_user.user_id if current_user is not None else None,
            )
        else:
            reply = self._build_local_reply(
                plan,
                context["emotion"].label,
                context["kundali_summary"],
                context["retrieval_matches"],
                context["tool_outputs"],
            )

        self._persist_chat_turns(context, reply)

        return {
            "reply": reply,
            "intent": plan.action,
            "planner_confidence": plan.confidence,
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
        plan = context["plan"]

        yield (
            "meta",
            {
                "resolved_session_id": session_id,
                "intent": plan.action,
                "planner_confidence": plan.confidence,
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
                plan,
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
                "intent": plan.action,
            },
        )
