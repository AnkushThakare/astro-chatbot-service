from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
import re
from typing import Any

from sqlalchemy.orm import Session

from src.auth.jwt import AuthenticatedUser
from src.astro.kundli import compute_full_chart
from src.core.core_service import CoreServiceClient
from src.core.emotion import detect_emotion
from src.core.logging import get_logger
from src.core.llm import GroqClient
from src.core.memory import MemoryService
from src.core.planner import ConversationPlanner, PlannerResult
from src.core.persona import build_persona_prompt
from src.core.product_policy import validate_product_search_query
from src.core.rag import RAGService
from src.core.router import pick_model_route
from src.core.streaming import chunk_text
from src.db.repositories.users import UserRepository
from src.tools.show_matchmaking import show_matchmaking
from src.tools.show_kundali import show_kundali

logger = get_logger(__name__)


class ChatService:
    TOOL_CONFIDENCE_THRESHOLD = 0.75
    ENGLISH_HINT_WORDS = {
        "and",
        "because",
        "career",
        "feel",
        "for",
        "guidance",
        "help",
        "i",
        "is",
        "it",
        "my",
        "not",
        "progress",
        "really",
        "should",
        "the",
        "what",
        "why",
        "work",
        "would",
    }
    HINGLISH_HINT_WORDS = {
        "aap",
        "batayein",
        "hai",
        "hain",
        "ka",
        "karna",
        "kya",
        "kyun",
        "mein",
        "mujhe",
        "nahi",
        "samajh",
        "theek",
        "toh",
        "ya",
    }
    CLARIFICATION_FILLER_PHRASES = (
        "Let's break it down a bit.",
        "Let us break it down a bit.",
    )
    RELATIONSHIP_TOKENS = {
        "love",
        "relationship",
        "partner",
        "marriage",
        "misunderstandings",
        "distance",
        "dating",
    }

    def __init__(self, db: Session, settings: Any) -> None:
        self.db = db
        self.settings = settings
        self.memory_service = MemoryService(db)
        self.user_repository = UserRepository(db)
        self.rag_service = RAGService()
        self.groq_client = GroqClient(settings)
        self.planner = ConversationPlanner(self.groq_client, settings.GROQ_PLANNER_MODEL)
        self.core_service_client = CoreServiceClient(settings)

    def _resolve_internal_user_id(self, current_user: AuthenticatedUser | None) -> int | None:
        """Resolve external user ID from JWT to internal DB user.id.

        Creates the user row if it doesn't exist yet. Returns None for
        anonymous (unauthenticated) users.
        """
        if current_user is None:
            return None
        user = self.user_repository.get_or_create(current_user.user_id)
        return user.id

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
            "policy_note": "Only mention products that appear in these catalog results.",
            "items": items,
            "source": "core-service",
        }

    @staticmethod
    def _build_empty_product_tool_output(search_query: str) -> dict[str, Any]:
        return {
            "tool": "recommend_product",
            "summary": (
                "No matching product catalog results were found for "
                f"'{search_query}'. Do not mention any specific rudraksha, mukhi count, mala, "
                "or bracelet variant unless it appears in actual catalog results. Give only "
                "general guidance or suggest a pandit consultation."
            ),
            "policy_note": "No matching catalog items were found for this request.",
            "items": [],
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
                    "experience_years": consultant.get("experience_years"),
                    "average_rating": consultant.get("average_rating"),
                    "total_reviews": consultant.get("total_reviews"),
                    "bio": consultant.get("bio"),
                    "offered_services": consultant.get("offered_services"),
                    "city": consultant.get("city"),
                    "state": consultant.get("state"),
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

    async def _find_consultant_results(
        self,
        query: str,
        current_user: AuthenticatedUser | None,
    ) -> list[dict[str, Any]]:
        consultant_results = await self.core_service_client.search_pandits(query, current_user)
        if consultant_results:
            return consultant_results

        candidate_queries = [query]
        lowered = query.lower()
        if "relationship" in lowered or "love" in lowered or "marriage" in lowered:
            candidate_queries.extend(["relationship", "marriage", "pandit"])
        elif "career" in lowered:
            candidate_queries.extend(["career", "pandit"])
        else:
            candidate_queries.append("pandit")

        seen: set[str] = set()
        for candidate in candidate_queries:
            normalized = candidate.strip().lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            public_results = await self.core_service_client.list_public_pandits(candidate)
            if public_results:
                return public_results
        return []

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
    def _tool_guardrail_decision(
        cls,
        plan: PlannerResult,
        *,
        birth_details: dict[str, Any] | None,
        matchmaking_details: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not plan.should_call_tool:
            return {"allowed": False, "reason": "planner_declined_tool"}
        if plan.confidence <= cls.TOOL_CONFIDENCE_THRESHOLD:
            return {
                "allowed": False,
                "reason": "low_confidence",
                "threshold": cls.TOOL_CONFIDENCE_THRESHOLD,
                "confidence": plan.confidence,
            }
        if plan.action not in {
            "show_kundali",
            "matchmaking",
            "book_pooja",
            "recommend_product",
            "suggest_consultant",
        }:
            return {"allowed": False, "reason": "action_not_toolable", "action": plan.action}
        if not cls._has_required_fields_for_action(
            plan,
            birth_details=birth_details,
            matchmaking_details=matchmaking_details,
        ):
            return {
                "allowed": False,
                "reason": "missing_required_fields",
                "missing_information": plan.missing_information,
            }
        decision: dict[str, Any] = {"allowed": True, "reason": "passed"}
        search_query = cls._planner_search_query(plan)
        if search_query is not None:
            decision["search_query"] = search_query
        return decision

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
        return cls._tool_guardrail_decision(
            plan,
            birth_details=birth_details,
            matchmaking_details=matchmaking_details,
        )["allowed"]

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

    @classmethod
    def _infer_response_language(cls, message: str) -> str:
        tokens = re.findall(r"[a-zA-Z']+", message.lower())
        if not tokens:
            return "english"

        english_score = sum(token in cls.ENGLISH_HINT_WORDS for token in tokens)
        hinglish_score = sum(token in cls.HINGLISH_HINT_WORDS for token in tokens)

        if hinglish_score >= 2 and hinglish_score >= english_score:
            return "hinglish"
        return "english"

    @staticmethod
    def _message_declines_products(message: str) -> bool:
        lowered = message.lower()
        decline_patterns = (
            "do not want product",
            "don't want product",
            "dont want product",
            "no product",
            "not want product",
            "not right now",
            "without product",
        )
        return any(pattern in lowered for pattern in decline_patterns)

    @staticmethod
    def _message_requests_single_step(message: str) -> bool:
        lowered = message.lower()
        single_step_patterns = (
            "one practical next step",
            "one step",
            "just one step",
            "what should i do first",
            "this week only",
        )
        return any(pattern in lowered for pattern in single_step_patterns)

    @classmethod
    def _build_response_style_context(
        cls,
        *,
        message: str,
        plan: PlannerResult,
        tool_outputs: list[dict[str, Any]],
    ) -> str:
        lines: list[str] = []
        response_language = cls._infer_response_language(message)
        if response_language == "english":
            lines.append(
                "Reply in plain English only. Do not switch into Hindi or Hinglish unless the user does so."
            )
        else:
            lines.append("Reply in natural Hinglish.")

        lines.append("Do not repeat earlier explanations unless the user explicitly asks for a recap.")
        lines.append("Keep the reply focused, warm, and conversational, not lecture-like.")
        lines.append(
            "Make the reply feel human and engaging: start with a brief natural acknowledgment, then move the conversation forward."
        )
        lines.append(
            "Keep the user engaged with one clear takeaway and at most one relevant follow-up question."
        )
        lines.append("Do not sound robotic, overly polished, or emotionally flat.")

        if plan.action == "ask_clarification":
            lines.append(
                "Use at most 2 short paragraphs. Briefly acknowledge the concern in a human way, then ask exactly one clear question."
            )
            lines.append(
                "Keep clarification replies to 2 or 3 short sentences total. Avoid filler phrases like 'let us break it down' or 'it can be really challenging'."
            )
            lines.append(
                "If you offer choices, keep them in one short sentence with at most 3 options."
            )
            lines.append(
                "Do not mention houses, Venus, Moon, planets, or chart techniques in a clarification reply unless chart data is already present or the user explicitly asks for astrological reasoning."
            )
        elif plan.action == "respond_only":
            lines.append(
                "Use 2 to 4 short sentences by default. Give the plain meaning first, then one practical next step or one engaging follow-up question."
            )
            lines.append(
                "Prefer a reply that sounds like a normal conversation, not a mini article. If the user seems uncertain, gently invite them to say a little more."
            )
        elif plan.action in {"recommend_product", "suggest_consultant", "book_pooja"}:
            lines.append(
                "Keep the reply compact: 2 to 4 short sentences. Briefly explain the recommendation, then point the user to the available options."
            )
            lines.append(
                "Sound advisory and personal, not transactional. The user should feel guided, not pitched to."
            )

        product_items_available = any(
            output.get("tool") == "recommend_product" and bool(output.get("items"))
            for output in tool_outputs
        )
        if plan.action != "recommend_product" and not product_items_available:
            lines.append(
                "Do not introduce remedies, rudraksha, bracelets, or products on your own in this reply."
            )
        if cls._message_declines_products(message):
            lines.append(
                "The user does not want product suggestions right now. Do not mention remedies or products in this reply."
            )
        if cls._message_requests_single_step(message):
            lines.append(
                "The user wants exactly one practical next step for this week. Give one clear action in 2 to 4 sentences and stop."
            )
        return "\n".join(lines)

    @classmethod
    def _postprocess_reply(cls, *, reply: str, plan: PlannerResult, message: str) -> str:
        compacted = " ".join(reply.split())
        lowered_message = message.lower()
        lowered_reply = compacted.lower()

        if plan.action == "ask_clarification":
            for phrase in cls.CLARIFICATION_FILLER_PHRASES:
                compacted = compacted.replace(phrase, "").strip()
            lowered_reply = compacted.lower()

            if (
                "career" in lowered_message
                and "direction" in lowered_reply
                and "confidence" in lowered_reply
                and (
                    "timing" in lowered_reply
                    or "stuck" in lowered_reply
                    or "current situation" in lowered_reply
                    or "current role" in lowered_reply
                )
            ):
                return (
                    "I understand. Career confusion usually comes from direction, confidence, or timing. "
                    "Which of these feels strongest right now?"
                )

            if any(token in lowered_message for token in cls.RELATIONSHIP_TOKENS):
                if "what's been troubling you" in lowered_reply or "current relationship" in lowered_reply:
                    return (
                        "I understand. In love matters, the real issue is usually clarity, trust, or timing. "
                        "Which of these feels most unsettled right now?"
                    )
                if "currently in a relationship" in lowered_reply or "looking for someone special" in lowered_reply:
                    return (
                        "I understand. In love matters, the real issue is usually clarity, trust, or timing. "
                        "Which of these feels most unsettled right now?"
                    )
                if "misunderstand" in lowered_message and ("you, your partner, or a bit of both" in lowered_reply or "you, your partner, or both" in lowered_reply):
                    return (
                        "That sounds more like emotional distance than one big fight. "
                        "Do you feel this is coming more from you, your partner, or both?"
                    )
                if "misunderstand" in lowered_message and ("same page" in lowered_reply or "grown apart" in lowered_reply):
                    return (
                        "That sounds more like emotional distance than one big fight. "
                        "Do you feel this is coming more from you, your partner, or both?"
                    )

        if plan.action == "suggest_consultant" and (
            "would you like me to arrange a consultation" in lowered_reply
            or "would you like me to suggest a consultant" in lowered_reply
            or "recommend consulting with a professional astrologer" in lowered_reply
            or "astrological dynamics at play" in lowered_reply
        ):
            if any(token in lowered_message for token in cls.RELATIONSHIP_TOKENS):
                return (
                    "Yes, speaking to a relationship astrologer would help here. "
                    "I can show you available pandits for relationship guidance."
                )
            return "Yes, speaking to an astrologer would help here. I can show you available pandits."

        return compacted

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
        tool_guardrail = self._tool_guardrail_decision(
            plan,
            birth_details=birth_details,
            matchmaking_details=matchmaking_details,
        )
        tool_execution_allowed = bool(tool_guardrail["allowed"])
        route = pick_model_route(
            plan.model_copy(update={"should_call_tool": tool_execution_allowed})
        )
        emotion = detect_emotion(message)

        logger.info(
            "Planner decision evaluated",
            extra={
                "extra_fields": {
                    "session_id": session_id,
                    "planner_action": plan.action,
                    "planner_confidence": plan.confidence,
                    "planner_should_call_tool": plan.should_call_tool,
                    "planner_arguments": plan.arguments,
                    "planner_missing_information": plan.missing_information,
                    "tool_guardrail": tool_guardrail,
                    "resolved_route_model": route.model,
                    "resolved_reasoning_profile": route.reasoning_profile,
                }
            },
        )

        # Resolve authenticated user to internal DB ID for cross-session memory
        internal_user_id = self._resolve_internal_user_id(current_user)

        if birth_details is not None:
            self.memory_service.remember_birth_details(
                session_id, birth_details, user_id=internal_user_id,
            )

        recent_messages = self.memory_service.recent_messages(
            session_id,
            self.settings.MEMORY_WINDOW_SIZE,
        )
        long_term_context = self.memory_service.long_term_context(
            session_id, user_id=internal_user_id,
        )
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
            sanitized_query = validate_product_search_query(search_query)
            product_results = await self.core_service_client.search_products(sanitized_query)
            product_output = self._build_product_tool_output(
                product_results,
                kundali_summary=kundali_summary,
            )
            if product_output is not None:
                tool_outputs.append(product_output)
            else:
                tool_outputs.append(self._build_empty_product_tool_output(sanitized_query))
        if tool_execution_allowed and plan.action == "suggest_consultant" and search_query is not None:
            consultant_results = await self._find_consultant_results(
                search_query,
                current_user,
            )
            consultant_output = self._build_consultant_tool_output(
                consultant_results,
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
                "content": self._build_response_style_context(
                    message=message,
                    plan=plan,
                    tool_outputs=tool_outputs,
                ),
            }
        )
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
            "tool_guardrail": tool_guardrail,
            "internal_user_id": internal_user_id,
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

    async def _background_memory_extraction(
        self,
        session_id: str,
        *,
        user_id: int | None = None,
    ) -> None:
        """Extract facts from conversation in the background (non-blocking)."""
        if not self.groq_client.is_configured:
            return
        try:
            await self.memory_service.extract_and_store_facts(
                session_id, self.groq_client, user_id=user_id,
            )
        except Exception as exc:
            logger.warning("Background memory extraction failed for %s: %s", session_id, exc)

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

        reply = self._postprocess_reply(reply=reply, plan=plan, message=message)

        self._persist_chat_turns(context, reply)

        # Fire-and-forget memory extraction every few turns
        recent_count = len(self.memory_service.recent_messages(session_id, limit=20))
        if recent_count >= 4 and recent_count % 4 == 0:
            asyncio.ensure_future(self._background_memory_extraction(
                session_id, user_id=context.get("internal_user_id"),
            ))

        return {
            "reply": reply,
            "intent": plan.action,
            "planner_confidence": plan.confidence,
            "planner_arguments": plan.arguments,
            "tool_guardrail": context["tool_guardrail"],
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
                "planner_arguments": plan.arguments,
                "tool_guardrail": context["tool_guardrail"],
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
        reply = self._postprocess_reply(reply=reply, plan=plan, message=message)
        self._persist_chat_turns(context, reply)

        # Fire-and-forget memory extraction every few turns
        session_id = context["session_id"]
        recent_count = len(self.memory_service.recent_messages(session_id, limit=20))
        if recent_count >= 4 and recent_count % 4 == 0:
            asyncio.ensure_future(self._background_memory_extraction(
                session_id, user_id=context.get("internal_user_id"),
            ))

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
