from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from datetime import datetime
import json
import re
import time
from typing import Any

from sqlalchemy.orm import Session

from src.astro.geocode import geocode_place
from src.auth.jwt import AuthenticatedUser
from src.astro.kundli import compute_full_chart
from src.astro.transits import compute_current_transits, format_transits_for_prompt
from src.astro.predictions import generate_predictive_insights, format_predictions_for_prompt
from src.core.config import settings
from src.core.confidence_policy import get_threshold, is_confident
from src.core.core_service import CoreServiceClient
from src.core.emotion import detect_emotion
from src.core.guardrails import final_response_guardrail, pre_scope_guardrail, sanitize_user_input, tool_specific_guardrail
from src.core.logging import get_logger
from src.core.llm import GroqClient
from src.core.memory import MemoryService
from src.core.pattern_engine import build_pattern_summary
from src.core.planner import ConversationPlanner, PlannerResult, ToolCallPlanner
from src.core.persona import build_persona_prompt
from src.core.prompt_registry import current_prompt_metadata
from src.core.product_policy import enrich_product_query, validate_product_search_query
from src.core.rag import RAGService
from src.core.response_composer import build_cards, build_style_instruction, compose_blocked_reply, compose_clarification_reply
from src.core.router import ChatRouteDecision, pick_model_route
from src.core.streaming import chunk_text
from src.db.repositories.users import UserRepository
from src.tools.show_matchmaking import show_matchmaking
from src.tools.show_kundali import show_kundali

logger = get_logger(__name__)

# ── Session-level birth detail cache ──────────────────────────────
# Keeps birth details in memory so they persist across messages
# within the same session. Entries expire after 2 hours.
_SESSION_BIRTH_CACHE: dict[str, tuple[dict[str, Any], float]] = {}
_SESSION_BIRTH_PARTIAL_CACHE: dict[str, tuple[dict[str, Any], float]] = {}
_SESSION_MATCHMAKING_CACHE: dict[str, tuple[dict[str, Any], float]] = {}
_SESSION_CONTEXT_CACHE: dict[str, tuple[dict[str, Any], float]] = {}
_BIRTH_CACHE_TTL = 7200  # 2 hours


def _get_cached_birth_details(session_id: str) -> dict[str, Any] | None:
    entry = _SESSION_BIRTH_CACHE.get(session_id)
    if entry is None:
        return None
    data, ts = entry
    if time.time() - ts > _BIRTH_CACHE_TTL:
        _SESSION_BIRTH_CACHE.pop(session_id, None)
        return None
    return data


def _set_cached_birth_details(session_id: str, details: dict[str, Any]) -> None:
    _SESSION_BIRTH_CACHE[session_id] = (details, time.time())
    # Evict old entries to prevent memory leak (keep last 500 sessions)
    if len(_SESSION_BIRTH_CACHE) > 500:
        oldest = sorted(_SESSION_BIRTH_CACHE, key=lambda k: _SESSION_BIRTH_CACHE[k][1])
        for key in oldest[:100]:
            _SESSION_BIRTH_CACHE.pop(key, None)


def _get_cached_partial_birth_details(session_id: str) -> dict[str, Any] | None:
    entry = _SESSION_BIRTH_PARTIAL_CACHE.get(session_id)
    if entry is None:
        return None
    data, ts = entry
    if time.time() - ts > _BIRTH_CACHE_TTL:
        _SESSION_BIRTH_PARTIAL_CACHE.pop(session_id, None)
        return None
    return data


def _set_cached_partial_birth_details(session_id: str, details: dict[str, Any]) -> None:
    _SESSION_BIRTH_PARTIAL_CACHE[session_id] = (details, time.time())
    if len(_SESSION_BIRTH_PARTIAL_CACHE) > 500:
        oldest = sorted(
            _SESSION_BIRTH_PARTIAL_CACHE,
            key=lambda k: _SESSION_BIRTH_PARTIAL_CACHE[k][1],
        )
        for key in oldest[:100]:
            _SESSION_BIRTH_PARTIAL_CACHE.pop(key, None)


def _clear_cached_partial_birth_details(session_id: str) -> None:
    _SESSION_BIRTH_PARTIAL_CACHE.pop(session_id, None)


def _get_cached_matchmaking_details(session_id: str) -> dict[str, Any] | None:
    entry = _SESSION_MATCHMAKING_CACHE.get(session_id)
    if entry is None:
        return None
    data, ts = entry
    if time.time() - ts > _BIRTH_CACHE_TTL:
        _SESSION_MATCHMAKING_CACHE.pop(session_id, None)
        return None
    return data


def _set_cached_matchmaking_details(session_id: str, details: dict[str, Any]) -> None:
    _SESSION_MATCHMAKING_CACHE[session_id] = (details, time.time())
    if len(_SESSION_MATCHMAKING_CACHE) > 500:
        oldest = sorted(
            _SESSION_MATCHMAKING_CACHE,
            key=lambda k: _SESSION_MATCHMAKING_CACHE[k][1],
        )
        for key in oldest[:100]:
            _SESSION_MATCHMAKING_CACHE.pop(key, None)


def _get_cached_session_context(session_id: str) -> dict[str, Any] | None:
    entry = _SESSION_CONTEXT_CACHE.get(session_id)
    if entry is None:
        return None
    data, ts = entry
    if time.time() - ts > _BIRTH_CACHE_TTL:
        _SESSION_CONTEXT_CACHE.pop(session_id, None)
        return None
    return data


def _set_cached_session_context(session_id: str, details: dict[str, Any]) -> None:
    _SESSION_CONTEXT_CACHE[session_id] = (details, time.time())
    if len(_SESSION_CONTEXT_CACHE) > 500:
        oldest = sorted(
            _SESSION_CONTEXT_CACHE,
            key=lambda k: _SESSION_CONTEXT_CACHE[k][1],
        )
        for key in oldest[:100]:
            _SESSION_CONTEXT_CACHE.pop(key, None)


class ChatService:
    COMPACT_HISTORY_WINDOW = 4
    POLICY_DOMAINS = {"product_policy", "booking_guidance", "remedy_guidance"}
    GREETING_TOKENS = {"hello", "hey", "hi", "namaste", "namaskar", "hii"}
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
    PERSONAL_TOKENS = {
        "i",
        "me",
        "my",
        "mine",
        "myself",
        "mujhe",
        "mera",
        "meri",
        "mere",
        "main",
        "hum",
        "hamara",
    }
    LIFE_GUIDANCE_TOKENS = {
        "career",
        "clarity",
        "decision",
        "destiny",
        "family",
        "finance",
        "future",
        "guidance",
        "health",
        "job",
        "love",
        "marriage",
        "money",
        "partner",
        "peace",
        "problem",
        "relationship",
        "spiritual",
        "spirituality",
        "stress",
        "studies",
        "study",
        "timing",
        "work",
    }
    ASTROLOGY_SCOPE_TOKENS = {
        "astrology",
        "astrologer",
        "birth",
        "bracelet",
        "chart",
        "compatibility",
        "consultant",
        "graha",
        "horoscope",
        "house",
        "houses",
        "jyotish",
        "kundali",
        "kundli",
        "lagna",
        "matchmaking",
        "moon",
        "nakshatra",
        "pandit",
        "planet",
        "planets",
        "pooja",
        "puja",
        "rashi",
        "remedies",
        "remedy",
        "rudraksha",
        "saturn",
        "transit",
        "venus",
        "zodiac",
    }
    PUBLIC_FIGURE_TOKENS = {
        "actor",
        "actress",
        "celebrity",
        "celebrities",
        "cricketer",
        "election",
        "minister",
        "ministers",
        "opponent",
        "opponents",
        "political",
        "politician",
        "politicians",
        "president",
        "prime",
    }
    POLITICS_TOKENS = {
        "election",
        "government",
        "minister",
        "political",
        "politician",
        "politicians",
        "president",
        "prime",
    }
    SPORTS_TOKENS = {
        "cricket",
        "football",
        "match",
        "opponent",
        "score",
        "team",
        "winner",
        "win",
    }
    TECH_TOKENS = {
        "app",
        "code",
        "coding",
        "laptop",
        "mobile",
        "phone",
        "software",
        "tech",
        "technology",
    }
    FOLLOW_UP_HINT_TOKENS = {
        "also",
        "and",
        "else",
        "explain",
        "more",
        "next",
        "now",
        "then",
        "what",
        "why",
    }
    NON_LOCATION_TOKENS = {
        "again",
        "already",
        "book",
        "check",
        "details",
        "gave",
        "given",
        "give",
        "know",
        "please",
        "provided",
        "recommend",
        "see",
        "send",
        "sent",
        "share",
        "shared",
        "show",
        "suggest",
        "tell",
        "told",
        "want",
    }
    ENTERTAINMENT_TOKENS = {
        "bollywood",
        "movie",
        "movies",
        "music",
        "show",
        "star",
    }
    MATCHMAKING_TOKENS = {
        "compatibility",
        "guna",
        "kundali",
        "kundli",
        "match",
        "matching",
        "matchmaking",
        "milan",
    }
    MATCHMAKING_FOLLOWUP_TOKENS = {
        "already",
        "check",
        "done",
        "go",
        "now",
        "ok",
        "okay",
        "please",
        "proceed",
        "see",
    }
    KUNDALI_TOKENS = {
        "birth",
        "chart",
        "horoscope",
        "kundali",
        "kundli",
    }
    KUNDALI_FOLLOWUP_TOKENS = {
        "about",
        "again",
        "career",
        "check",
        "continue",
        "deeper",
        "details",
        "explain",
        "finance",
        "future",
        "job",
        "love",
        "marriage",
        "more",
        "next",
        "now",
        "please",
        "proceed",
        "read",
        "relationship",
        "show",
        "timing",
        "what",
        "why",
    }

    def __init__(self, db: Session, settings: Any) -> None:
        self.db = db
        self.settings = settings
        self.memory_service = MemoryService(db)
        self.user_repository = UserRepository(db)
        self.rag_service = RAGService(db)
        self.groq_client = GroqClient(settings)
        self.planner = ToolCallPlanner(self.groq_client)
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
    def _normalize_user_message(message: str) -> str:
        return " ".join((message or "").strip().split())

    @staticmethod
    def _house_for_sign_index(
        sign_index: int | None,
        house_sign_indices: dict[int, int] | dict[str, int] | None,
    ) -> int | None:
        if sign_index is None or not house_sign_indices:
            return None
        normalized: dict[int, int] = {}
        for house, value in house_sign_indices.items():
            if not isinstance(house, (int, str)) or not isinstance(value, int):
                continue
            try:
                normalized[int(house)] = int(value)
            except ValueError:
                continue
        for house, house_sign_index in normalized.items():
            if house_sign_index == sign_index:
                return house
        return None

    @staticmethod
    def _format_chart_context_for_prompt(rag_chart_context: dict[str, Any] | None) -> str | None:
        if not rag_chart_context:
            return None
        parts: list[str] = []
        asc = rag_chart_context.get("ascendant_sign")
        if asc:
            parts.append(f"Ascendant: {asc}")
        moon = rag_chart_context.get("moon_sign")
        moon_nak = rag_chart_context.get("moon_nakshatra")
        if moon:
            moon_str = f"Moon sign: {moon}"
            if moon_nak:
                moon_str += f" (Nakshatra: {moon_nak})"
            parts.append(moon_str)
        maha = rag_chart_context.get("current_mahadasha")
        antara = rag_chart_context.get("current_antardasha")
        if maha:
            dasha_str = f"Current Mahadasha: {maha}"
            maha_period = rag_chart_context.get("mahadasha_period", "").strip()
            if maha_period and maha_period != "to":
                dasha_str += f" ({maha_period})"
            if antara:
                dasha_str += f"\nCurrent Antardasha: {antara}"
                antara_period = rag_chart_context.get("antardasha_period", "").strip()
                if antara_period and antara_period != "to":
                    dasha_str += f" ({antara_period})"
            parts.append(dasha_str)
        upcoming = rag_chart_context.get("upcoming_dashas") or []
        if upcoming:
            upcoming_strs = [f"{d['planet']} from {d['starts']} ({d['years']}yr)" for d in upcoming[:2]]
            parts.append("Upcoming dashas: " + ", ".join(upcoming_strs))
        placements = rag_chart_context.get("placements") or []
        placement_strs = []
        for p in placements[:7]:
            planet = p.get("planet", "").title()
            house = p.get("house")
            sign = p.get("sign", "").title() if p.get("sign") else ""
            if planet and house:
                entry = f"{planet} in {house}th house"
                if sign:
                    entry += f" ({sign})"
                placement_strs.append(entry)
        if placement_strs:
            parts.append("Placements: " + ", ".join(placement_strs))
        yogas = rag_chart_context.get("yogas") or []
        if yogas:
            yoga_strs = [f"{y['name']}: {y['description']}" for y in yogas[:4]]
            parts.append("Yogas detected:\n- " + "\n- ".join(yoga_strs))
        return "\n".join(parts) if parts else None

    @classmethod
    def _build_chart_rag_context(cls, chart: dict[str, Any] | None) -> dict[str, Any] | None:
        if not chart:
            return None

        ascendant_sign = chart.get("ascendant_sign_name")
        planets_sign_names = chart.get("planets_sign_names") or {}
        planets_sign_indices = chart.get("planets_sign_indices") or {}
        house_sign_indices = chart.get("house_sign_indices") or {}
        d1_planets = (((chart.get("charts") or {}).get("D1") or {}).get("planets")) or []
        placements: list[dict[str, Any]] = []

        if isinstance(d1_planets, list) and d1_planets:
            for planet in d1_planets:
                if not isinstance(planet, dict):
                    continue
                name = planet.get("name") or planet.get("id")
                house = planet.get("house_num")
                sign = planet.get("sign_name")
                if isinstance(name, str) and isinstance(house, int):
                    placements.append(
                        {
                            "planet": name.lower(),
                            "house": house,
                            "sign": str(sign).lower() if isinstance(sign, str) else None,
                        }
                    )
        else:
            for name, sign_index in planets_sign_indices.items():
                if not isinstance(name, str) or not isinstance(sign_index, int):
                    continue
                house = cls._house_for_sign_index(sign_index, house_sign_indices)
                placements.append(
                    {
                        "planet": name.lower(),
                        "house": house,
                        "sign": str(planets_sign_names.get(name) or "").lower() or None,
                    }
                )

        dasha = chart.get("dasha") or {}
        maha = dasha.get("mahadasha")
        antara = dasha.get("antardasha")
        moon_sign = planets_sign_names.get("Moon")
        significant_placements = [
            f"{placement['planet']} in {placement['house']}th house"
            for placement in placements
            if placement.get("planet") and placement.get("house")
        ][:5]

        # Extract nakshatras from enriched planet data
        nakshatras: list[str] = []
        if isinstance(d1_planets, list):
            for planet in d1_planets:
                if isinstance(planet, dict):
                    nak = planet.get("nakshatra_name")
                    if isinstance(nak, str) and nak:
                        name = planet.get("name") or planet.get("id") or ""
                        nakshatras.append(f"{name}: {nak}")
        # Moon nakshatra from dasha calculation
        moon_nakshatra = dasha.get("nakshatra")

        astro_entities = {
            "planets": sorted(
                {
                    str(placement.get("planet"))
                    for placement in placements
                    if isinstance(placement.get("planet"), str) and placement.get("planet")
                }
            ),
            "houses": sorted(
                {
                    int(placement.get("house"))
                    for placement in placements
                    if isinstance(placement.get("house"), int)
                }
            ),
            "signs": sorted(
                {
                    str(placement.get("sign"))
                    for placement in placements
                    if isinstance(placement.get("sign"), str) and placement.get("sign")
                }
            ),
            "nakshatras": nakshatras,
            "dashas": [
                str(value).lower()
                for value in [maha, antara]
                if isinstance(value, str) and value
            ],
        }

        # Yogas from chart
        yogas = chart.get("yogas") or []

        return {
            "ascendant_sign": ascendant_sign,
            "moon_sign": moon_sign,
            "moon_nakshatra": moon_nakshatra,
            "current_mahadasha": maha,
            "current_antardasha": antara,
            "mahadasha_period": f"{dasha.get('mahadasha_start', '')} to {dasha.get('mahadasha_end', '')}".strip(),
            "antardasha_period": f"{dasha.get('antardasha_start', '')} to {dasha.get('antardasha_end', '')}".strip(),
            "upcoming_dashas": dasha.get("upcoming_dashas") or [],
            "placements": placements,
            "significant_placements": significant_placements,
            "yogas": yogas,
            "astro_entities": astro_entities,
        }

    async def _compute_rag_chart_context(
        self,
        birth_details: dict[str, Any],
    ) -> dict[str, Any]:
        chart = await compute_full_chart(birth_details)

        # Compute transits and predictions in parallel
        transit_data: dict[str, Any] | None = None
        try:
            transit_data = await compute_current_transits(chart, birth_details)
        except Exception:
            logger.debug("Transit computation failed — skipping")

        predictions = generate_predictive_insights(chart, transit_data)

        return {
            "chart": chart,
            "rag_context": self._build_chart_rag_context(chart),
            "transit_data": transit_data,
            "predictions": predictions,
        }

    @staticmethod
    def _plan_from_route(route_decision: ChatRouteDecision) -> PlannerResult:
        action = "ask_clarification" if route_decision.route == "CLARIFICATION" else route_decision.intent
        return PlannerResult(
            action=action,  # type: ignore[arg-type]
            confidence=route_decision.confidence,
            arguments=dict(route_decision.normalized_args),
            missing_information=list(route_decision.missing_slots),
            should_call_tool=route_decision.should_call_tool,
            reasoning=route_decision.reason,
        )

    @staticmethod
    def _route_name_for_plan(plan: PlannerResult) -> str:
        if plan.action == "ask_clarification" or plan.missing_information:
            return "CLARIFICATION"
        if plan.action in {
            "show_kundali",
            "matchmaking",
            "book_pooja",
            "recommend_product",
            "suggest_consultant",
        } and plan.should_call_tool:
            return "TOOL_FLOW"
        return "FAST_CHAT"

    @classmethod
    def _route_decision_from_plan(
        cls,
        route_decision: ChatRouteDecision,
        plan: PlannerResult,
    ) -> ChatRouteDecision:
        return ChatRouteDecision(
            route=cls._route_name_for_plan(plan),
            intent=plan.action,
            confidence=plan.confidence,
            risk_level=route_decision.risk_level,
            reason="planner_resolved_route",
            missing_slots=list(plan.missing_information),
            should_call_tool=plan.should_call_tool,
            needs_planner=False,
            normalized_args=dict(plan.arguments),
        )

    @staticmethod
    def _safe_json_dumps(payload: dict[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii=True, default=str)

    @staticmethod
    def _derive_memory_facts(
        *,
        message: str,
        emotion_label: str,
        birth_details: dict[str, Any] | None,
    ) -> dict[str, str]:
        lowered = message.lower()
        concern = "general"
        if any(term in lowered for term in ("career", "job", "work", "money", "finance")):
            concern = "career delay" if "delay" in lowered or "stuck" in lowered else "career"
        elif any(term in lowered for term in ("relationship", "marriage", "partner", "love")):
            concern = "relationship"
        elif any(term in lowered for term in ("health", "stress", "peace", "anxiety")):
            concern = "peace and wellbeing"

        preferred_style = "simple English calm tone"
        if any(term in lowered for term in ("mantra", "shiv", "hanuman", "pooja", "puja")):
            preferred_style = "simple Hindi-English devotional tone"

        return {
            "last_concern": concern,
            "emotion_trend": emotion_label,
            "preferred_style": preferred_style,
            "last_topic": concern,
            "has_birth_details": "true" if birth_details is not None else "false",
        }

    def _persist_lightweight_memory(
        self,
        *,
        session_id: str,
        message: str,
        emotion_label: str,
        birth_details: dict[str, Any] | None,
        user_id: int | None,
    ) -> None:
        for fact_key, fact_value in self._derive_memory_facts(
            message=message,
            emotion_label=emotion_label,
            birth_details=birth_details,
        ).items():
            self.memory_service.repository.upsert_fact(
                session_id=session_id,
                fact_key=fact_key,
                fact_value=fact_value,
                user_id=user_id,
            )

    @staticmethod
    def _compact_recent_messages(recent_messages: list[dict[str, str]] | None) -> list[dict[str, str]]:
        if not recent_messages:
            return []
        return list(recent_messages[-ChatService.COMPACT_HISTORY_WINDOW :])

    @classmethod
    def _pending_birth_slots(cls, parts: dict[str, Any] | None) -> list[str]:
        if parts is None:
            return []
        pending: list[str] = []
        if parts.get("date_parts") is None:
            pending.append("birth_date")
        if parts.get("time_parts") is None:
            pending.append("birth_time")
        if parts.get("place") is None:
            pending.append("birth_place")
        return pending

    @staticmethod
    def _summarize_matchmaking_details(matchmaking_details: dict[str, Any] | None) -> str | None:
        if not matchmaking_details:
            return None
        primary = matchmaking_details.get("primary") or {}
        partner = matchmaking_details.get("partner") or {}
        primary_gender = primary.get("gender")
        partner_gender = partner.get("gender")
        primary_birth = primary.get("birth_details") or {}
        partner_birth = partner.get("birth_details") or {}
        primary_birth_dt = primary_birth.get("birth_datetime")
        partner_birth_dt = partner_birth.get("birth_datetime")
        pieces: list[str] = []
        if isinstance(primary_gender, str) and isinstance(partner_gender, str):
            pieces.append(f"primary={primary_gender}, partner={partner_gender}")
        if primary_birth_dt and partner_birth_dt:
            pieces.append("both birth charts provided")
        return ", ".join(pieces) if pieces else "pair details available"

    @staticmethod
    def _last_tool_summary(tool_outputs: list[dict[str, Any]]) -> tuple[str | None, str | None]:
        if not tool_outputs:
            return None, None
        last_output = tool_outputs[-1]
        tool_name = last_output.get("tool")
        summary = last_output.get("summary")
        return (
            tool_name if isinstance(tool_name, str) else None,
            summary if isinstance(summary, str) else None,
        )

    @classmethod
    def _build_compact_session_state(
        cls,
        *,
        context: dict[str, Any],
        reply: str,
        response_metadata: dict[str, Any] | None = None,
        partial: bool = False,
    ) -> dict[str, Any]:
        del response_metadata
        active_intent = context["plan"].action
        effective_birth_details = context.get("effective_birth_details")
        partial_birth_details = context.get("partial_birth_details")
        matchmaking_details = context.get("matchmaking_details")
        pending_slots: list[str] = []
        if effective_birth_details is None and cls._has_birth_detail_fragment(partial_birth_details):
            pending_slots.extend(cls._pending_birth_slots(partial_birth_details))
        elif active_intent == "show_kundali" and effective_birth_details is None:
            pending_slots.extend(["birth_details"])
        if active_intent == "matchmaking" and matchmaking_details is None:
            pending_slots.append("matchmaking_details")
        last_tool, last_tool_summary = cls._last_tool_summary(list(context.get("tool_outputs") or []))
        state = {
            "active_intent": active_intent,
            "birth_details": effective_birth_details,
            "partial_birth_details": partial_birth_details if effective_birth_details is None else None,
            "matchmaking_details": matchmaking_details,
            "pending_slots": pending_slots,
            "last_tool": last_tool,
            "last_tool_summary": last_tool_summary,
            "last_user_goal": context.get("message"),
            "last_reply_summary": reply,
            "last_updated_at": datetime.utcnow().isoformat(),
            "partial_reply": partial,
        }
        return state

    @classmethod
    def _format_compact_session_context(cls, session_state: dict[str, Any] | None) -> str | None:
        if not session_state:
            return None
        lines: list[str] = []
        active_intent = session_state.get("active_intent")
        if isinstance(active_intent, str) and active_intent:
            lines.append(f"Active intent: {active_intent}")
        pending_slots = session_state.get("pending_slots")
        if isinstance(pending_slots, list) and pending_slots:
            lines.append("Pending slots: " + ", ".join(str(slot) for slot in pending_slots))
        birth_details = session_state.get("birth_details")
        if isinstance(birth_details, dict) and birth_details:
            lines.append("Birth details: available")
        partial_birth_details = session_state.get("partial_birth_details")
        if isinstance(partial_birth_details, dict) and cls._has_birth_detail_fragment(partial_birth_details):
            lines.append(
                "Partial birth details: "
                + ", ".join(
                    label
                    for label, value in (
                        ("date", partial_birth_details.get("date_parts")),
                        ("time", partial_birth_details.get("time_parts")),
                        ("place", partial_birth_details.get("place")),
                    )
                    if value is not None
                )
            )
        matchmaking_details = session_state.get("matchmaking_details")
        matchmaking_summary = cls._summarize_matchmaking_details(matchmaking_details)
        if matchmaking_summary:
            lines.append("Matchmaking context: " + matchmaking_summary)
        last_tool = session_state.get("last_tool")
        last_tool_summary = session_state.get("last_tool_summary")
        if isinstance(last_tool, str) and last_tool:
            if isinstance(last_tool_summary, str) and last_tool_summary:
                lines.append(f"Last tool: {last_tool} | {last_tool_summary}")
            else:
                lines.append(f"Last tool: {last_tool}")
        last_user_goal = session_state.get("last_user_goal")
        if isinstance(last_user_goal, str) and last_user_goal:
            lines.append(f"Last user goal: {last_user_goal}")
        if not lines:
            return None
        return "Compact session state:\n" + "\n".join(f"- {line}" for line in lines)

    @staticmethod
    def _compact_retrieval_match(match: dict[str, Any]) -> dict[str, Any]:
        metadata = match.get("metadata") if isinstance(match.get("metadata"), dict) else {}
        compact: dict[str, Any] = {}

        bucket = match.get("bucket")
        if isinstance(bucket, str) and bucket:
            compact["bucket"] = bucket

        source = match.get("title") or match.get("source")
        if isinstance(source, str) and source:
            compact["source"] = source

        path = match.get("path")
        if isinstance(path, str) and path:
            compact["path"] = path

        domain = metadata.get("domain") or match.get("type")
        if isinstance(domain, str) and domain:
            compact["domain"] = domain

        score = match.get("score")
        if isinstance(score, (int, float)):
            compact["score"] = round(float(score), 4)

        risk = match.get("risk") or metadata.get("risk")
        if isinstance(risk, str) and risk:
            compact["risk"] = risk

        allowed_actions = match.get("allowed_actions") or metadata.get("allowed_actions")
        if isinstance(allowed_actions, list):
            compact["allowed_actions"] = [
                item for item in allowed_actions if isinstance(item, str) and item.strip()
            ]

        chunk_id = metadata.get("chunk_id") or metadata.get("id")
        if isinstance(chunk_id, (int, str)):
            compact["chunk_id"] = chunk_id

        return compact

    @classmethod
    def _build_retrieval_trace(
        cls,
        *,
        retrieval_matches: list[dict[str, Any]] | None = None,
        retrieval_knowledge_matches: list[dict[str, Any]] | None = None,
        retrieval_policy_matches: list[dict[str, Any]] | None = None,
        retrieval_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        combined_matches = list(retrieval_matches or [])
        knowledge_matches = list(retrieval_knowledge_matches or [])
        policy_matches = list(retrieval_policy_matches or [])

        if not knowledge_matches and not policy_matches and combined_matches:
            for match in combined_matches:
                if cls._is_policy_match(match):
                    policy_matches.append(match)
                else:
                    knowledge_matches.append(match)

        trace = {
            "match_count": len(combined_matches) or (len(knowledge_matches) + len(policy_matches)),
            "knowledge_match_count": len(knowledge_matches),
            "policy_match_count": len(policy_matches),
            "knowledge": [cls._compact_retrieval_match(match) for match in knowledge_matches],
            "policy": [cls._compact_retrieval_match(match) for match in policy_matches],
        }
        if isinstance(retrieval_metadata, dict):
            provider = retrieval_metadata.get("provider")
            strategy = retrieval_metadata.get("retrieval_strategy")
            embedding_provider = retrieval_metadata.get("embedding_provider")
            embedding_model = retrieval_metadata.get("embedding_model")
            vector_backend = retrieval_metadata.get("vector_backend")
            keyword_backend = retrieval_metadata.get("keyword_backend")
            reranker_provider = retrieval_metadata.get("reranker_provider")
            reranker_model = retrieval_metadata.get("reranker_model")
            chart_context_used = retrieval_metadata.get("chart_context_used")
            document_count = retrieval_metadata.get("document_count")
            if isinstance(provider, str) and provider:
                trace["provider"] = provider
            if isinstance(strategy, str) and strategy:
                trace["strategy"] = strategy
            if isinstance(embedding_provider, str) and embedding_provider:
                trace["embedding_provider"] = embedding_provider
            if isinstance(embedding_model, str) and embedding_model:
                trace["embedding_model"] = embedding_model
            if isinstance(vector_backend, str) and vector_backend:
                trace["vector_backend"] = vector_backend
            if isinstance(keyword_backend, str) and keyword_backend:
                trace["keyword_backend"] = keyword_backend
            if isinstance(reranker_provider, str) and reranker_provider:
                trace["reranker_provider"] = reranker_provider
            if isinstance(reranker_model, str) and reranker_model:
                trace["reranker_model"] = reranker_model
            if isinstance(chart_context_used, bool):
                trace["chart_context_used"] = chart_context_used
            if isinstance(document_count, int):
                trace["document_count"] = document_count
        return trace

    @classmethod
    def _response_metadata(
        cls,
        *,
        reply: str,
        route_decision: ChatRouteDecision,
        plan: PlannerResult,
        message: str,
        tool_outputs: list[dict[str, Any]],
        latency_ms: int,
        model: str | None = None,
        total_tokens_input: int | None = None,
        total_tokens_output: int | None = None,
        needs_birth_details: bool = False,
        retrieval_matches: list[dict[str, Any]] | None = None,
        retrieval_knowledge_matches: list[dict[str, Any]] | None = None,
        retrieval_policy_matches: list[dict[str, Any]] | None = None,
        retrieval_metadata: dict[str, Any] | None = None,
        recommendation_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        tool_names = [output.get("tool") for output in tool_outputs if isinstance(output.get("tool"), str)]
        return {
            "text": reply,
            "metadata": {
                "cards": build_cards(tool_outputs),
                "intent": plan.action,
                "route": route_decision.route,
                "risk_level": route_decision.risk_level,
                "tool_called": tool_names[0] if len(tool_names) == 1 else tool_names,
                "prompt_versions": current_prompt_metadata(),
                "model_used": model,
                "route_taken": route_decision.route,
                "variant_id": settings.RESPONSE_VARIANT_ID,
                "total_tokens_input": total_tokens_input,
                "total_tokens_output": total_tokens_output,
                "latency_ms": latency_ms,
                "needs_birth_details": needs_birth_details,
                "retrieval_trace": cls._build_retrieval_trace(
                    retrieval_matches=retrieval_matches,
                    retrieval_knowledge_matches=retrieval_knowledge_matches,
                    retrieval_policy_matches=retrieval_policy_matches,
                    retrieval_metadata=retrieval_metadata,
                ),
                "product_recommendation_trace": cls._build_product_recommendation_trace(
                    message=message,
                    plan=plan,
                    tool_outputs=tool_outputs,
                    recommendation_context=recommendation_context,
                ),
            },
        }

    @classmethod
    def _llm_trace_metadata(cls, context: dict[str, Any]) -> dict[str, Any]:
        plan: PlannerResult = context["plan"]
        route_decision: ChatRouteDecision = context["route_decision"]
        return {
            "intent": plan.action,
            "route": route_decision.route,
            "risk_level": route_decision.risk_level,
            "tool_execution_allowed": bool(context.get("tool_execution_allowed")),
            "tool_count": len(context.get("tool_outputs") or []),
            "retrieval_trace": cls._build_retrieval_trace(
                retrieval_matches=context.get("retrieval_matches"),
                retrieval_knowledge_matches=context.get("retrieval_knowledge_matches"),
                retrieval_policy_matches=context.get("retrieval_policy_matches"),
                retrieval_metadata=context.get("retrieval_metadata"),
            ),
            "product_recommendation_trace": cls._build_product_recommendation_trace(
                message=str(context.get("message") or ""),
                plan=plan,
                tool_outputs=list(context.get("tool_outputs") or []),
                recommendation_context=context.get("recommendation_context"),
            ),
        }

    @staticmethod
    async def _run_with_timeout(awaitable: Any, timeout_seconds: int) -> Any:
        return await asyncio.wait_for(awaitable, timeout=timeout_seconds)

    @classmethod
    async def _safe_tool_result(
        cls,
        awaitable: Any,
        *,
        timeout_seconds: int,
        default: Any,
        tool_name: str = "unknown",
    ) -> Any:
        try:
            return await cls._run_with_timeout(awaitable, timeout_seconds)
        except asyncio.TimeoutError:
            logger.warning("Tool '%s' timed out after %ds", tool_name, timeout_seconds)
            return default
        except Exception as exc:
            logger.warning("Tool '%s' failed: %s", tool_name, exc)
            return default

    @staticmethod
    def _needs_rag(route_decision: ChatRouteDecision, plan: PlannerResult) -> bool:
        if route_decision.route == "FAST_CHAT":
            return True
        return plan.action in {
            "show_kundali",
            "matchmaking",
            "respond_only",
            "book_pooja",
            "recommend_product",
            "suggest_consultant",
        }

    @staticmethod
    def _format_retrieval_context(matches: list[dict[str, Any]]) -> str:
        if not matches:
            return "No retrieved astrology notes were matched."
        lines: list[str] = []
        for match in matches:
            metadata = match.get("metadata") or {}
            domain = metadata.get("domain") or match.get("type")
            title = match.get("title") or match.get("source") or "Knowledge"
            excerpt = match.get("excerpt") or match.get("text") or ""
            if isinstance(domain, str) and domain:
                lines.append(f"- {title} [{domain}]: {excerpt}")
            else:
                lines.append(f"- {title}: {excerpt}")
        return "\n".join(lines)

    @classmethod
    def _is_policy_match(cls, match: dict[str, Any]) -> bool:
        metadata = match.get("metadata") or {}
        domain = metadata.get("domain") or match.get("type")
        return isinstance(domain, str) and domain in cls.POLICY_DOMAINS

    @classmethod
    def _format_retrieval_knowledge_context(cls, matches: list[dict[str, Any]]) -> str:
        knowledge_matches = [match for match in matches if not cls._is_policy_match(match)]
        return cls._format_retrieval_context(knowledge_matches)

    @classmethod
    def _format_retrieval_policy_context(cls, matches: list[dict[str, Any]]) -> str | None:
        policy_matches = [match for match in matches if cls._is_policy_match(match)]
        if not policy_matches:
            return None
        return cls._format_retrieval_context(policy_matches)

    @staticmethod
    def _format_tool_context(tool_outputs: list[dict[str, Any]]) -> str:
        if not tool_outputs:
            return "No tool output used."
        lines: list[str] = []
        for output in tool_outputs:
            lines.append(f"Tool: {output['tool']}")
            lines.append(output["summary"])
            policy_note = output.get("policy_note")
            if isinstance(policy_note, str) and policy_note.strip():
                lines.append("Tool policy: " + policy_note.strip())
            display_names = ChatService._tool_output_display_names(output)
            if display_names:
                lines.append("Items shown to user:")
                lines.extend(f"- {name}" for name in display_names)
                lines.append(
                    "Your reply MUST reference these items by name or clearly say that options are shown above."
                )
            elif output.get("tool") in {"recommend_product", "suggest_consultant", "book_pooja"}:
                lines.append(
                    "No items were returned by this tool. Do NOT imply that a specific product, pandit, "
                    "or puja option is currently available."
                )
        return "\n".join(lines)

    @staticmethod
    def _tool_output_display_names(output: dict[str, Any]) -> list[str]:
        names: list[str] = []
        for key in ("items", "home_puja_services", "temple_services", "pandits"):
            values = output.get(key)
            if not isinstance(values, list):
                continue
            for item in values[:3]:
                if not isinstance(item, dict):
                    continue
                name = item.get("name")
                if isinstance(name, str) and name.strip():
                    names.append(name.strip())
        return names

    @classmethod
    def _verify_card_text_consistency(
        cls,
        *,
        reply: str,
        tool_outputs: list[dict[str, Any]],
        conversation_id: str,
    ) -> None:
        expected_names: list[str] = []
        for output in tool_outputs:
            expected_names.extend(cls._tool_output_display_names(output))
        if not expected_names:
            return

        lowered_reply = reply.lower()
        generic_reference_present = (
            "shown above" in lowered_reply
            or "options below" in lowered_reply
            or "options above" in lowered_reply
        )
        if generic_reference_present:
            return

        if any(name.lower() in lowered_reply for name in expected_names):
            return

        logger.warning(
            "card_text_mismatch",
            extra={
                "extra_fields": {
                    "conversation_id": conversation_id,
                    "expected_items": expected_names,
                }
            },
        )

    @staticmethod
    def _usage_token_counts(usage: dict[str, Any] | None) -> tuple[int | None, int | None]:
        if not isinstance(usage, dict):
            return None, None
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        return (
            prompt_tokens if isinstance(prompt_tokens, int) else None,
            completion_tokens if isinstance(completion_tokens, int) else None,
        )

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
        *,
        search_query: str | None = None,
        soft_recommendation: bool = False,
    ) -> dict[str, Any] | None:
        if not products:
            return None

        items: list[dict[str, Any]] = []
        names: list[str] = []
        for product in products[:3]:
            name = str(product.get("name") or "Product")
            names.append(name)
            primary_media = product.get("primary_media") or {}
            image_variants = product.get("primary_image_variants") or {}
            # Prefer small variant for cards, fall back to primary media URL
            image_url = (
                image_variants.get("small")
                or (primary_media.get("url") if isinstance(primary_media, dict) else None)
            )
            items.append(
                {
                    "id": str(product.get("id", "")),
                    "slug": str(product.get("slug", "")),
                    "name": name,
                    "starting_price_paise": product.get("starting_price_paise"),
                    "starting_mrp_paise": product.get("starting_mrp_paise"),
                    "image_url": image_url,
                    "category": (product.get("category") or {}).get("name"),
                }
            )

        if soft_recommendation:
            summary = (
                "A supportive item is quietly available if the user wants it: "
                + ", ".join(names)
                + ". Do NOT lead with this or make it the focus. Mention only as a brief aside after fully answering the concern."
            )
        else:
            summary = "Relevant products from the Digveda catalog: " + ", ".join(names) + "."
        if kundali_summary:
            summary += f" Kundali context considered: {kundali_summary}"

        return {
            "tool": "recommend_product",
            "event_name": "suggestion_product",
            "summary": summary,
            "policy_note": (
                "This product is optional background context. Mention it in at most one brief sentence after "
                "fully addressing the user's concern. Do not describe features or pricing. Do not lead with it."
                if soft_recommendation
                else "Only mention products that appear in these catalog results."
            ),
            "items": items,
            "search_query": search_query,
            "soft_recommendation": soft_recommendation,
            "source": "core-service",
        }

    @staticmethod
    def _build_empty_product_tool_output(search_query: str) -> dict[str, Any]:
        return {
            "tool": "recommend_product",
            "event_name": "suggestion_product",
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
    def _policy_allows_product_recommendation(matches: list[dict[str, Any]]) -> bool:
        for match in matches:
            metadata = match.get("metadata") or {}
            allowed_actions = metadata.get("allowed_actions") or []
            if not isinstance(allowed_actions, list):
                continue
            normalized = {str(action) for action in allowed_actions}
            if {"can_recommend", "recommend_product"} & normalized:
                return True
        return False

    PLANET_REMEDY_MAP: dict[str, dict[str, str]] = {
        "sun": {"query": "1 mukhi rudraksha", "reason": "strengthen Sun energy"},
        "moon": {"query": "2 mukhi rudraksha", "reason": "emotional balance and Moon strength"},
        "mars": {"query": "3 mukhi rudraksha", "reason": "Mars energy and courage"},
        "mercury": {"query": "4 mukhi rudraksha", "reason": "communication and intellect"},
        "jupiter": {"query": "5 mukhi rudraksha", "reason": "Jupiter blessings and wisdom"},
        "venus": {"query": "6 mukhi rudraksha", "reason": "Venus harmony and relationships"},
        "saturn": {"query": "7 mukhi rudraksha", "reason": "Saturn discipline and stability"},
        "rahu": {"query": "rudraksha bracelet protection", "reason": "Rahu balance and clarity"},
        "ketu": {"query": "9 mukhi rudraksha", "reason": "Ketu spiritual grounding"},
    }
    DUSTHANA_HOUSES = {6, 8, 12}

    @classmethod
    def _identify_afflicted_planets(cls, chart_context: dict[str, Any]) -> list[str]:
        afflicted: list[str] = []
        placements = chart_context.get("placements") or []
        for p in placements:
            planet = p.get("planet", "").lower()
            house = p.get("house")
            if isinstance(house, int) and house in cls.DUSTHANA_HOUSES and planet:
                afflicted.append(planet)
        maha = (chart_context.get("current_mahadasha") or "").lower()
        if maha and maha not in afflicted:
            afflicted.append(maha)
        return afflicted

    @classmethod
    def _infer_soft_product_query(
        cls,
        *,
        message: str,
        kundali_summary: str | None = None,
        chart_context: dict[str, Any] | None = None,
    ) -> str | None:
        direct_query = ConversationPlanner._extract_product_query(message)
        if direct_query:
            direct_tokens = set(direct_query.split())
            if direct_tokens & {"rudraksha", "bracelet", "mala", "mukhi"}:
                return direct_query

        combined_text = message if not kundali_summary else f"{message} {kundali_summary}"
        tokens = cls._message_tokens(combined_text)

        remedy_seeking = tokens & {"remedy", "remedies", "upay", "solution", "wear", "support", "protection"}
        if not remedy_seeking:
            return None

        # Chart-aware: map afflicted planet → specific product
        if chart_context:
            afflicted = cls._identify_afflicted_planets(chart_context)
            for planet in afflicted:
                mapping = cls.PLANET_REMEDY_MAP.get(planet)
                if mapping:
                    return mapping["query"]

        if tokens & {"saturn", "shani", "delay", "stuck", "obstacle", "obstacles"}:
            return "rudraksha career"
        if tokens & {"stress", "anxiety", "peace", "sleep", "restless", "moon", "calm"}:
            return "rudraksha peace"
        if tokens & {"rahu", "ketu", "fear", "confusion", "negative"}:
            return "bracelet protection"
        if tokens & {"spiritual", "spirituality", "meditation", "mantra"}:
            return "rudraksha mala"
        if tokens & (cls.RELATIONSHIP_TOKENS | {"venus", "harmony"}):
            return "rudraksha relationship"
        return None

    @classmethod
    def _soft_product_decision(
        cls,
        *,
        message: str,
        plan: PlannerResult,
        retrieval_policy_matches: list[dict[str, Any]],
        kundali_summary: str | None = None,
        chart_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not settings.SOFT_PRODUCT_RECOMMENDATIONS_ENABLED:
            return {"allowed": False, "reason": "feature_disabled", "query": None}
        if plan.action in {"ask_clarification", "book_pooja", "recommend_product", "suggest_consultant"}:
            return {"allowed": False, "reason": "plan_ineligible", "query": None}
        if cls._message_declines_products(message):
            return {"allowed": False, "reason": "user_declined_products", "query": None}
        if cls._message_requests_single_step(message):
            return {"allowed": False, "reason": "single_step_requested", "query": None}
        if not cls._policy_allows_product_recommendation(retrieval_policy_matches):
            return {"allowed": False, "reason": "policy_disallows_product", "query": None}

        query = cls._infer_soft_product_query(message=message, kundali_summary=kundali_summary, chart_context=chart_context)
        if query is None:
            return {"allowed": False, "reason": "no_product_mapping", "query": None}
        return {"allowed": True, "reason": "policy_and_context_match", "query": query}

    @classmethod
    def _should_offer_soft_product(
        cls,
        *,
        message: str,
        plan: PlannerResult,
        retrieval_policy_matches: list[dict[str, Any]],
        kundali_summary: str | None = None,
    ) -> bool:
        return cls._soft_product_decision(
            message=message,
            plan=plan,
            retrieval_policy_matches=retrieval_policy_matches,
            kundali_summary=kundali_summary,
        )["allowed"]

    _product_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}
    _PRODUCT_CACHE_TTL = 90  # seconds

    @classmethod
    def _get_cached_products(cls, query: str) -> list[dict[str, Any]] | None:
        entry = cls._product_cache.get(query)
        if entry is None:
            return None
        cached_at, results = entry
        if time.time() - cached_at > cls._PRODUCT_CACHE_TTL:
            del cls._product_cache[query]
            return None
        return results

    @classmethod
    def _set_cached_products(cls, query: str, results: list[dict[str, Any]]) -> None:
        cls._product_cache[query] = (time.time(), results)
        # Evict stale entries if cache grows
        if len(cls._product_cache) > 50:
            now = time.time()
            cls._product_cache = {
                k: v for k, v in cls._product_cache.items()
                if now - v[0] <= cls._PRODUCT_CACHE_TTL
            }

    async def _lookup_product_tool_output(
        self,
        *,
        search_query: str,
        kundali_summary: str | None = None,
        include_empty: bool,
        soft_recommendation: bool,
        afflicted_planets: list[str] | None = None,
        current_dasha: str | None = None,
    ) -> dict[str, Any] | None:
        sanitized_query = enrich_product_query(
            search_query,
            afflicted_planets=afflicted_planets,
            current_dasha=current_dasha,
        )
        generic_terms = {"product", "products", "item", "items", "some", "any", "something"}
        if all(token in generic_terms for token in sanitized_query.split()):
            sanitized_query = "rudraksha"

        product_results = self._get_cached_products(sanitized_query)
        if product_results is None:
            product_results = await self._safe_tool_result(
                self.core_service_client.search_products(sanitized_query),
                timeout_seconds=self.settings.TOOL_TIMEOUT_SECONDS,
                default=[],
                tool_name="search_products",
            )
            if product_results:
                self._set_cached_products(sanitized_query, product_results)

        if not product_results and " " in sanitized_query:
            product_cores = {"rudraksha", "bracelet", "mala", "mukhi"}
            fallback_token = next(
                (token for token in sanitized_query.split() if token in product_cores),
                "rudraksha",
            )
            product_results = self._get_cached_products(fallback_token)
            if product_results is None:
                product_results = await self._safe_tool_result(
                    self.core_service_client.search_products(fallback_token),
                    timeout_seconds=self.settings.TOOL_TIMEOUT_SECONDS,
                    default=[],
                    tool_name="search_products_fallback",
                )
                if product_results:
                    self._set_cached_products(fallback_token, product_results)
        product_output = self._build_product_tool_output(
            product_results,
            kundali_summary=kundali_summary,
            search_query=sanitized_query,
            soft_recommendation=soft_recommendation,
        )
        if product_output is not None:
            return product_output
        if include_empty:
            return self._build_empty_product_tool_output(sanitized_query)
        return None

    @classmethod
    def _build_product_recommendation_trace(
        cls,
        *,
        message: str,
        plan: PlannerResult,
        tool_outputs: list[dict[str, Any]],
        recommendation_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        product_outputs = [
            output for output in tool_outputs
            if output.get("tool") == "recommend_product"
        ]
        soft_outputs = [output for output in product_outputs if output.get("soft_recommendation")]
        explicit_outputs = [output for output in product_outputs if not output.get("soft_recommendation")]
        primary_output = explicit_outputs[0] if explicit_outputs else (soft_outputs[0] if soft_outputs else None)
        recommendation_context = recommendation_context or {}
        soft_context = recommendation_context.get("soft_product") or {}

        trace: dict[str, Any] = {
            "mode": (
                "explicit"
                if explicit_outputs
                else "soft"
                if soft_outputs
                else "none"
            ),
            "source_action": plan.action,
            "tool_invoked": bool(product_outputs),
            "presented": False,
            "result_count": 0,
            "user_declined_products": cls._message_declines_products(message),
            "soft_eligible": bool(soft_context.get("eligible")),
            "soft_reason": soft_context.get("reason") if isinstance(soft_context.get("reason"), str) else None,
        }

        decision_query = soft_context.get("query")
        if isinstance(decision_query, str) and decision_query.strip():
            trace["soft_query"] = decision_query

        if primary_output is None:
            return trace

        items = primary_output.get("items")
        item_names = cls._tool_output_display_names(primary_output)
        result_count = len(items) if isinstance(items, list) else 0
        trace["presented"] = result_count > 0
        trace["result_count"] = result_count
        if item_names:
            trace["item_names"] = item_names

        search_query = primary_output.get("search_query")
        if isinstance(search_query, str) and search_query.strip():
            trace["search_query"] = search_query

        policy_note = primary_output.get("policy_note")
        if isinstance(policy_note, str) and policy_note.strip():
            trace["policy_note"] = policy_note

        return trace

    @staticmethod
    def _log_product_recommendation_trace(
        *,
        session_id: str,
        route_decision: ChatRouteDecision,
        plan: PlannerResult,
        trace: dict[str, Any],
    ) -> None:
        logger.info(
            "Product recommendation evaluated",
            extra={
                "extra_fields": {
                    "session_id": session_id,
                    "route_name": route_decision.route,
                    "planner_action": plan.action,
                    "recommendation_mode": trace.get("mode"),
                    "recommendation_presented": trace.get("presented"),
                    "recommendation_result_count": trace.get("result_count"),
                    "soft_eligible": trace.get("soft_eligible"),
                    "soft_reason": trace.get("soft_reason"),
                    "search_query": trace.get("search_query"),
                    "soft_query": trace.get("soft_query"),
                    "item_names": trace.get("item_names"),
                }
            },
        )

    @staticmethod
    def _build_empty_consultant_tool_output(search_query: str) -> dict[str, Any]:
        return {
            "tool": "suggest_consultant",
            "event_name": "suggestion_consultant",
            "summary": (
                "No consultant matches were returned for "
                f"'{search_query}'. Do not invent pandit names, ratings, prices, or availability. "
                "Offer broader guidance and ask whether the user wants a broader category or another support path."
            ),
            "policy_note": "No consultant matches were found for this request.",
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

        # Build unique fallback queries and run them in parallel
        lowered = query.lower()
        fallback_queries: list[str] = []
        if "relationship" in lowered or "love" in lowered or "marriage" in lowered:
            fallback_queries = ["relationship", "marriage", "pandit"]
        elif "career" in lowered:
            fallback_queries = ["career", "pandit"]
        else:
            fallback_queries = ["pandit"]

        # Deduplicate against original query
        seen = {query.strip().lower()}
        unique_queries = [q for q in fallback_queries if q not in seen]

        if not unique_queries:
            return []

        # Run all fallback queries in parallel
        results = await asyncio.gather(
            *(self.core_service_client.list_public_pandits(q) for q in unique_queries)
        )
        for result in results:
            if result:
                return result
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
    def _build_empty_booking_tool_output(search_query: str) -> dict[str, Any]:
        return {
            "tool": "book_pooja",
            "event_name": "suggestion_booking",
            "summary": (
                "No booking suggestions were returned for "
                f"'{search_query}'. Do not invent puja services, temple services, temple names, or pandit names. "
                "Ask the user whether they want a broader home puja search, a temple service, or a pandit consultation."
            ),
            "policy_note": "No booking suggestions were found for this request.",
            "home_puja_services": [],
            "temple_services": [],
            "pandits": [],
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
        message: str | None = None,
        birth_details: dict[str, Any] | None,
        matchmaking_details: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not plan.should_call_tool:
            return {"allowed": False, "reason": "planner_declined_tool"}
        if not is_confident(plan):
            threshold = get_threshold(plan.action)
            logger.info(
                "plan_rejected_by_threshold",
                extra={
                    "extra_fields": {
                        "action": plan.action,
                        "confidence": plan.confidence,
                        "threshold": threshold,
                    }
                },
            )
            return {
                "allowed": False,
                "reason": "low_confidence",
                "threshold": threshold,
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
        if message is not None:
            tool_guard = tool_specific_guardrail(plan.action, message, decision)
            if not tool_guard.allowed:
                return {
                    "allowed": False,
                    "reason": tool_guard.reason,
                    "risk_level": tool_guard.risk_level,
                    "safe_reply": tool_guard.safe_reply,
                }
            decision.update(tool_guard.normalized_args)
        return decision

    @staticmethod
    def _fallback_plan_for_threshold_rejection(
        plan: PlannerResult,
        tool_guardrail: dict[str, Any],
    ) -> PlannerResult:
        if tool_guardrail.get("reason") != "low_confidence":
            return plan
        return plan.model_copy(
            update={
                "action": "respond_only",
                "arguments": {},
                "missing_information": [],
                "should_call_tool": False,
                "reasoning": f"{plan.reasoning} Tool execution was rejected by the confidence policy.",
            }
        )

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

    @staticmethod
    def _message_requests_detail(message: str) -> bool:
        lowered = message.lower()
        detail_patterns = (
            "in detail",
            "in depth",
            "detailed",
            "deeply",
            "elaborate",
            "full analysis",
            "step by step",
            "tell me more",
            "explain more",
        )
        return any(pattern in lowered for pattern in detail_patterns)

    @staticmethod
    def _message_tokens(message: str) -> set[str]:
        return set(re.findall(r"[a-z0-9']+", message.lower()))

    @classmethod
    def _is_greeting_only(cls, message: str) -> bool:
        tokens = cls._message_tokens(message)
        return bool(tokens) and tokens.issubset(cls.GREETING_TOKENS | cls.PERSONAL_TOKENS)

    _MONTH_NAMES: dict[str, int] = {
        "jan": 1, "january": 1, "feb": 2, "february": 2,
        "mar": 3, "march": 3, "apr": 4, "april": 4,
        "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
        "aug": 8, "august": 8, "sep": 9, "september": 9,
        "oct": 10, "october": 10, "nov": 11, "november": 11,
        "dec": 12, "december": 12,
    }

    @classmethod
    def _extract_birth_date_parts(cls, message: str) -> tuple[int, int, int] | None:
        # Numeric: 20/06/2001, 20.06.2001, 20-06-2001
        numeric_match = re.search(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{4})\b", message)
        if numeric_match is not None:
            day = int(numeric_match.group(1))
            month = int(numeric_match.group(2))
            year = int(numeric_match.group(3))
            try:
                datetime(year, month, day)
            except ValueError:
                return None
            return day, month, year

        # Text month: "20 June 2001", "June 20 2001", "20 Jun, 2001"
        month_pattern = "|".join(cls._MONTH_NAMES.keys())
        text_match = re.search(
            rf"\b(\d{{1,2}})\s*[,]?\s*({month_pattern})\s*[,]?\s*(\d{{4}})\b",
            message, re.IGNORECASE,
        )
        if text_match is not None:
            day = int(text_match.group(1))
            month = cls._MONTH_NAMES[text_match.group(2).lower()]
            year = int(text_match.group(3))
            try:
                datetime(year, month, day)
            except ValueError:
                return None
            return day, month, year

        # "June 20, 2001" format
        text_match2 = re.search(
            rf"\b({month_pattern})\s+(\d{{1,2}})\s*[,]?\s*(\d{{4}})\b",
            message, re.IGNORECASE,
        )
        if text_match2 is not None:
            month = cls._MONTH_NAMES[text_match2.group(1).lower()]
            day = int(text_match2.group(2))
            year = int(text_match2.group(3))
            try:
                datetime(year, month, day)
            except ValueError:
                return None
            return day, month, year

        return None

    @staticmethod
    def _extract_birth_time_parts(message: str) -> tuple[int, int] | None:
        lowered = message.lower()
        meridiem_match = re.search(
            r"\b(?:time\s*)?(\d{1,2})(?:[:.](\d{2}))?\s*(am|pm)\b",
            lowered,
        )
        if meridiem_match is not None:
            hour = int(meridiem_match.group(1))
            minute = int(meridiem_match.group(2) or 0)
            meridiem = meridiem_match.group(3)
            if hour < 1 or hour > 12 or minute > 59:
                return None
            if meridiem == "pm" and hour != 12:
                hour += 12
            if meridiem == "am" and hour == 12:
                hour = 0
            return hour, minute

        time_match = re.search(r"\b(?:time\s*)?(\d{1,2})[:.](\d{2})\b", lowered)
        if time_match is None:
            return None

        hour = int(time_match.group(1))
        minute = int(time_match.group(2))
        if hour > 23 or minute > 59:
            return None
        return hour, minute

    @classmethod
    def _looks_like_bare_birth_place_text(cls, place: str | None) -> bool:
        if not place:
            return False
        if "?" in place or "!" in place:
            return False
        place_tokens = cls._message_tokens(place)
        if not place_tokens:
            return False
        if len(place_tokens) > 4:
            return False
        disallowed_tokens = (
            cls.GREETING_TOKENS
            | cls.PERSONAL_TOKENS
            | cls.FOLLOW_UP_HINT_TOKENS
            | cls.LIFE_GUIDANCE_TOKENS
            | cls.ASTROLOGY_SCOPE_TOKENS
            | cls.ENGLISH_HINT_WORDS
            | cls.HINGLISH_HINT_WORDS
            | cls.NON_LOCATION_TOKENS
            | {"the", "is"}
        )
        return not bool(place_tokens & disallowed_tokens)

    @classmethod
    def _extract_birth_place_text(
        cls,
        message: str,
        *,
        allow_bare_place: bool = False,
        allow_embedded_place: bool = False,
    ) -> str | None:
        lowered = message.lower()
        place_match = re.search(
            r"\b(?:place|city|birth place|birthplace|pob)\s*[:=-]?\s*([a-z][a-z\s,.-]{1,80})$",
            lowered,
        )
        if place_match is not None:
            place = re.sub(r"\s+", " ", place_match.group(1)).strip(" ,.-")
            return place.title() if place else None

        if not allow_bare_place:
            return None

        stripped = re.sub(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{4}\b", " ", message)
        stripped = re.sub(r"\b(?:time\s*)?\d{1,2}[:.]\d{2}\b", " ", stripped, flags=re.IGNORECASE)
        stripped = re.sub(
            r"\b(?:time\s*)?\d{1,2}(?:[:.]\d{2})?\s*(?:am|pm)\b",
            " ",
            stripped,
            flags=re.IGNORECASE,
        )
        stripped = re.sub(r"\b(?:dob|date|time|place|birth|birthplace|city|pob|of)\b", " ", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"[^A-Za-z,\s-]", " ", stripped)
        stripped = re.sub(r"\s+", " ", stripped).strip(" ,.-")
        if not stripped:
            return None
        if not allow_embedded_place:
            if len(stripped.split()) > 4:
                return None
            if not cls._looks_like_bare_birth_place_text(stripped):
                return None
            return stripped.title()

        tokens = re.findall(r"[A-Za-z]+", stripped)
        if not tokens:
            return None
        max_window = min(4, len(tokens))
        for width in range(max_window, 0, -1):
            for start in range(0, len(tokens) - width + 1):
                candidate = " ".join(tokens[start : start + width])
                if cls._looks_like_bare_birth_place_text(candidate):
                    return candidate.title()
        return None

    @classmethod
    def _extract_birth_detail_parts(
        cls,
        message: str,
        *,
        allow_bare_place: bool = False,
    ) -> dict[str, Any]:
        date_parts = cls._extract_birth_date_parts(message)
        time_parts = cls._extract_birth_time_parts(message)
        allow_embedded_place = date_parts is not None or time_parts is not None
        allow_bare_place = allow_bare_place or allow_embedded_place
        return {
            "date_parts": date_parts,
            "time_parts": time_parts,
            "place": cls._extract_birth_place_text(
                message,
                allow_bare_place=allow_bare_place,
                allow_embedded_place=allow_embedded_place,
            ),
        }

    @staticmethod
    def _has_birth_detail_fragment(parts: dict[str, Any] | None) -> bool:
        if not parts:
            return False
        return any(parts.get(key) is not None for key in ("date_parts", "time_parts", "place"))

    @staticmethod
    def _has_complete_birth_details(parts: dict[str, Any] | None) -> bool:
        if not parts:
            return False
        return all(parts.get(key) is not None for key in ("date_parts", "time_parts", "place"))

    @staticmethod
    def _merge_birth_detail_parts(
        existing: dict[str, Any] | None,
        incoming: dict[str, Any] | None,
    ) -> dict[str, Any]:
        merged = {
            "date_parts": None,
            "time_parts": None,
            "place": None,
        }
        for source in (existing or {}, incoming or {}):
            for key in merged:
                value = source.get(key)
                if value is not None:
                    merged[key] = value
        return merged

    @classmethod
    def _looks_like_birth_details_message(cls, message: str) -> bool:
        return cls._has_complete_birth_details(cls._extract_birth_detail_parts(message))

    @staticmethod
    def _assistant_requested_birth_details(recent_messages: list[dict[str, str]] | None) -> bool:
        if not recent_messages:
            return False
        prompt_markers = (
            "birth details",
            "date, time, and place of birth",
            "share your date, time, and place",
            "share your birth details",
            "share your date of birth",
            "share your birth time",
            "share your birthplace",
            "send the birthplace",
            "birthplace a bit more specifically",
            "exact chart-based answer",
            "exact chart",
            "personal astrology answer",
        )
        for item in recent_messages[-4:]:
            if item.get("role") != "assistant":
                continue
            content = item.get("content", "").lower()
            if any(marker in content for marker in prompt_markers):
                return True
        return False

    @classmethod
    def _is_birth_details_followup(
        cls,
        message: str,
        recent_messages: list[dict[str, str]] | None,
    ) -> bool:
        if not cls._assistant_requested_birth_details(recent_messages):
            return False
        return cls._has_birth_detail_fragment(
            cls._extract_birth_detail_parts(message, allow_bare_place=True)
        )

    @classmethod
    def _message_acknowledges_shared_birth_details(cls, message: str) -> bool:
        lowered = message.lower()
        acknowledgement_patterns = (
            "already given",
            "already gave",
            "already shared",
            "already told",
            "already provided",
            "gave already",
            "shared already",
            "told you already",
            "provided already",
            "i gave",
            "i shared",
            "i told you",
            "i provided",
        )
        return any(pattern in lowered for pattern in acknowledgement_patterns)

    @staticmethod
    def _assistant_requested_matchmaking_details(recent_messages: list[dict[str, str]] | None) -> bool:
        if not recent_messages:
            return False
        prompt_markers = (
            "birth details of both",
            "birth dates, times, and places of both",
            "compatibility of two people",
            "matchmaking",
            "match my kundali with partner",
        )
        for item in recent_messages[-4:]:
            if item.get("role") != "assistant":
                continue
            content = item.get("content", "").lower()
            if any(marker in content for marker in prompt_markers):
                return True
        return False

    @staticmethod
    def _session_state_prefers_intent(session_state: dict[str, Any] | None, intent: str) -> bool:
        if not session_state:
            return False
        active_intent = session_state.get("active_intent")
        last_tool = session_state.get("last_tool")
        return active_intent == intent or last_tool == intent

    @staticmethod
    def _session_state_pending_slots(session_state: dict[str, Any] | None) -> list[str]:
        if not session_state:
            return []
        pending_slots = session_state.get("pending_slots")
        if not isinstance(pending_slots, list):
            return []
        return [slot for slot in pending_slots if isinstance(slot, str)]

    @classmethod
    def _should_resume_matchmaking_from_context(
        cls,
        *,
        message: str,
        recent_messages: list[dict[str, str]] | None,
        matchmaking_details: dict[str, Any] | None,
        route_decision: ChatRouteDecision,
        session_state: dict[str, Any] | None = None,
    ) -> bool:
        if matchmaking_details is None:
            return False
        if route_decision.intent not in {"respond_only", "ask_clarification"}:
            return False
        tokens = cls._message_tokens(message)
        if tokens & cls.MATCHMAKING_TOKENS:
            return True
        if (
            cls._session_state_prefers_intent(session_state, "matchmaking")
            and len(tokens) <= 6
            and bool(tokens & cls.MATCHMAKING_FOLLOWUP_TOKENS)
        ):
            return True
        if not cls._assistant_requested_matchmaking_details(recent_messages):
            return False
        return len(tokens) <= 6 and bool(tokens & cls.MATCHMAKING_FOLLOWUP_TOKENS)

    @classmethod
    def _should_resume_kundali_from_context(
        cls,
        *,
        message: str,
        birth_details: dict[str, Any] | None,
        route_decision: ChatRouteDecision,
        session_state: dict[str, Any] | None,
    ) -> bool:
        if birth_details is None:
            return False
        if route_decision.intent not in {"respond_only", "ask_clarification"}:
            return False
        if not cls._session_state_prefers_intent(session_state, "show_kundali"):
            return False
        if cls._is_greeting_only(message) or cls._message_acknowledges_shared_birth_details(message):
            return False
        tokens = cls._message_tokens(message)
        if not tokens:
            return False
        if tokens & cls.KUNDALI_TOKENS:
            return True
        if tokens & cls.LIFE_GUIDANCE_TOKENS:
            return True
        return len(tokens) <= 6 and bool(tokens & cls.KUNDALI_FOLLOWUP_TOKENS)

    @classmethod
    def _should_keep_kundali_clarification_from_context(
        cls,
        *,
        message: str,
        partial_birth_details: dict[str, Any] | None,
        route_decision: ChatRouteDecision,
        session_state: dict[str, Any] | None,
    ) -> bool:
        if route_decision.intent not in {"respond_only", "ask_clarification"}:
            return False
        if not cls._session_state_prefers_intent(session_state, "show_kundali"):
            return False
        if not cls._has_birth_detail_fragment(partial_birth_details):
            return False
        if cls._message_acknowledges_shared_birth_details(message):
            return True
        if cls._is_greeting_only(message):
            return False
        tokens = cls._message_tokens(message)
        if tokens & cls.KUNDALI_TOKENS:
            return True
        if tokens & cls.LIFE_GUIDANCE_TOKENS:
            return True
        return len(tokens) <= 6 and bool(tokens & cls.FOLLOW_UP_HINT_TOKENS)

    @classmethod
    def _override_route_from_session_state(
        cls,
        *,
        message: str,
        recent_messages: list[dict[str, str]] | None,
        route_decision: ChatRouteDecision,
        session_state: dict[str, Any] | None,
        effective_birth_details: dict[str, Any] | None,
        partial_birth_details: dict[str, Any] | None,
        matchmaking_details: dict[str, Any] | None,
        provided_matchmaking_details: bool,
    ) -> ChatRouteDecision:
        if route_decision.route == "BLOCKED":
            return route_decision

        pending_slots = cls._session_state_pending_slots(session_state)

        if cls._should_keep_kundali_clarification_from_context(
            message=message,
            partial_birth_details=partial_birth_details,
            route_decision=route_decision,
            session_state=session_state,
        ):
            missing_slots = cls._pending_birth_slots(partial_birth_details) or [
                slot for slot in pending_slots if slot.startswith("birth_")
            ] or ["birth_details"]
            return ChatRouteDecision(
                route="CLARIFICATION",
                intent="show_kundali",
                confidence=0.95,
                risk_level="low",
                reason="cached_kundali_birth_context",
                missing_slots=missing_slots,
                should_call_tool=False,
            )

        if (
            effective_birth_details is None
            and matchmaking_details is None
            and route_decision.intent in {"respond_only", "ask_clarification"}
            and cls._session_state_prefers_intent(session_state, "matchmaking")
            and "matchmaking_details" in pending_slots
        ):
            tokens = cls._message_tokens(message)
            if tokens & cls.MATCHMAKING_TOKENS or (
                len(tokens) <= 6 and bool(tokens & cls.MATCHMAKING_FOLLOWUP_TOKENS)
            ):
                return ChatRouteDecision(
                    route="CLARIFICATION",
                    intent="matchmaking",
                    confidence=0.95,
                    risk_level="low",
                    reason="cached_matchmaking_context",
                    missing_slots=["matchmaking_details"],
                    should_call_tool=False,
                )

        if cls._should_resume_kundali_from_context(
            message=message,
            birth_details=effective_birth_details,
            route_decision=route_decision,
            session_state=session_state,
        ):
            return ChatRouteDecision(
                route="TOOL_FLOW",
                intent="show_kundali",
                confidence=0.96,
                risk_level="low",
                reason="cached_kundali_context",
                should_call_tool=True,
            )

        if (
            provided_matchmaking_details
            and matchmaking_details is not None
            and route_decision.intent in {"respond_only", "ask_clarification"}
        ) or cls._should_resume_matchmaking_from_context(
            message=message,
            recent_messages=recent_messages,
            matchmaking_details=matchmaking_details,
            route_decision=route_decision,
            session_state=session_state,
        ):
            return ChatRouteDecision(
                route="TOOL_FLOW",
                intent="matchmaking",
                confidence=0.96,
                risk_level="low",
                reason="cached_matchmaking_context",
                should_call_tool=True,
            )

        return route_decision

    @staticmethod
    async def _infer_birth_details_from_parts(parts: dict[str, Any] | None) -> dict[str, Any] | None:
        if not ChatService._has_complete_birth_details(parts):
            return None

        date_parts = parts["date_parts"]
        time_parts = parts["time_parts"]
        place = parts["place"]
        geocoded = await geocode_place(place)
        birth_datetime = datetime(
            date_parts[2],
            date_parts[1],
            date_parts[0],
            time_parts[0],
            time_parts[1],
        )
        return {
            "name": None,
            "latitude": geocoded["latitude"],
            "longitude": geocoded["longitude"],
            "birth_datetime": birth_datetime.isoformat(),
            "timezone_str": geocoded.get("timezone"),
        }

    async def _infer_birth_details_from_message(
        self,
        message: str,
    ) -> dict[str, Any] | None:
        return await self._infer_birth_details_from_parts(
            self._extract_birth_detail_parts(message)
        )

    @classmethod
    def _quick_scope_block_reason(cls, message: str) -> str | None:
        tokens = cls._message_tokens(message)
        if not tokens or cls._is_greeting_only(message):
            return None
        if tokens & cls.PUBLIC_FIGURE_TOKENS:
            return "public_figure_or_competitor_topic"
        if tokens & cls.TECH_TOKENS and not (tokens & cls.ASTROLOGY_SCOPE_TOKENS):
            return "out_of_astrology_scope"
        return None

    @classmethod
    def _scope_guardrail_decision(
        cls,
        *,
        message: str,
        plan: PlannerResult,
        recent_messages: list[dict[str, str]] | None = None,
        birth_details: dict[str, Any] | None = None,
        matchmaking_details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        tokens = cls._message_tokens(message)
        if cls._is_greeting_only(message):
            return {"allowed": True, "reason": "greeting_only"}

        if birth_details is not None or matchmaking_details is not None:
            return {"allowed": True, "reason": "structured_astrology_context"}

        if cls._is_birth_details_followup(message, recent_messages):
            return {"allowed": True, "reason": "birth_details_followup"}

        has_public_figure_cue = bool(tokens & cls.PUBLIC_FIGURE_TOKENS)
        has_astrology_cue = bool(tokens & cls.ASTROLOGY_SCOPE_TOKENS)
        has_personal_guidance_cue = bool(tokens & cls.LIFE_GUIDANCE_TOKENS)
        prior_astrology_context = False
        if recent_messages:
            prior_text = " ".join(
                item.get("content", "")
                for item in recent_messages[-4:]
                if isinstance(item, dict)
            ).lower()
            prior_tokens = cls._message_tokens(prior_text)
            prior_astrology_context = bool(prior_tokens & cls.ASTROLOGY_SCOPE_TOKENS) or bool(
                prior_tokens & cls.LIFE_GUIDANCE_TOKENS
            )
        short_follow_up = (
            len(tokens) <= 8
            and bool(tokens & (cls.FOLLOW_UP_HINT_TOKENS | cls.LIFE_GUIDANCE_TOKENS | cls.PERSONAL_TOKENS))
        )

        if has_public_figure_cue:
            return {"allowed": False, "reason": "public_figure_or_competitor_topic"}

        if has_astrology_cue or has_personal_guidance_cue:
            return {"allowed": True, "reason": "astrology_domain"}

        if prior_astrology_context and short_follow_up:
            return {"allowed": True, "reason": "astrology_followup_context"}

        if plan.action in {"show_kundali", "matchmaking", "book_pooja", "recommend_product", "suggest_consultant"}:
            return {"allowed": True, "reason": "astrology_service_action"}

        return {"allowed": False, "reason": "out_of_astrology_scope"}

    @classmethod
    def _build_scope_fallback_reply(cls, scope_reason: str, message: str = "") -> str:
        tokens = cls._message_tokens(message)
        if scope_reason == "public_figure_or_competitor_topic":
            if tokens & cls.POLITICS_TOKENS:
                return (
                    "I do not cover politics or public-figure predictions here. "
                    "I can help with your own Vedic astrology questions."
                )
            if tokens & cls.SPORTS_TOKENS:
                return (
                    "I am not the right assistant for sports or opponent predictions. "
                    "I can help with personal astrology around career, timing, or decisions."
                )
            if tokens & cls.ENTERTAINMENT_TOKENS:
                return (
                    "I do not cover celebrity or entertainment gossip here. "
                    "I can help with your own Vedic astrology questions."
                )
            return (
                "I do not cover politics, celebrities, or public-figure predictions here. "
                "I can help with your own Vedic astrology questions."
            )
        if tokens & cls.TECH_TOKENS:
            return (
                "I am here for astrology, not tech or product advice. "
                "Ask me about your chart, career, marriage, health, or spiritual guidance."
            )
        return (
            "I am here only for Vedic astrology guidance. "
            "Ask me about your chart, career, marriage, health, or spiritual guidance."
        )

    @classmethod
    def _build_fast_greeting_reply(cls, message: str) -> str | None:
        if not cls._is_greeting_only(message):
            return None
        if cls._infer_response_language(message) == "hinglish":
            return "Namaste. Batayein, aap kis baat par margdarshan chahte hain?"
        return "Hello. What would you like guidance on?"

    @staticmethod
    def _compact_to_sentence_limit(reply: str, max_sentences: int) -> str:
        parts = re.split(r"(?<=[.!?])\s+", " ".join(reply.split()))
        compact = [part.strip() for part in parts if part.strip()]
        if len(compact) <= max_sentences:
            return " ".join(compact)
        return " ".join(compact[:max_sentences]).strip()

    @staticmethod
    def _trim_to_word_limit(reply: str, max_words: int) -> str:
        words = reply.split()
        if len(words) <= max_words:
            return reply
        return " ".join(words[:max_words]).rstrip(",.;:") + "."

    @staticmethod
    def _limit_question_count(reply: str, max_questions: int = 1) -> str:
        if reply.count("?") <= max_questions:
            return reply

        trimmed: list[str] = []
        seen_questions = 0
        for char in reply:
            if char == "?":
                seen_questions += 1
                if seen_questions > max_questions:
                    break
            trimmed.append(char)
        return "".join(trimmed).rstrip(" ,.;:") + "?"

    @classmethod
    def _humanize_reply(cls, *, reply: str, message: str) -> str:
        humanized = " ".join(reply.split())
        replacements = {
            "Based on the provided context, ": "",
            "Based on the context, ": "",
            "If you want, I can ": "I can ",
            "If you want, we can ": "We can ",
            "If you want I can ": "I can ",
            "If you want we can ": "We can ",
        }
        for source, target in replacements.items():
            humanized = humanized.replace(source, target)
        return cls._limit_question_count(humanized, 1)

    @classmethod
    def _reply_sentence_limit(cls, plan: PlannerResult, message: str) -> int:
        if cls._message_requests_single_step(message):
            return 2
        if cls._message_requests_detail(message):
            return 5
        if plan.action in {"ask_clarification", "recommend_product", "suggest_consultant", "book_pooja"}:
            return 2
        if plan.action in {"show_kundali", "matchmaking"}:
            return 3
        return 3

    @classmethod
    def _reply_word_limit(cls, message: str) -> int:
        if cls._message_requests_detail(message):
            return 140
        if cls._message_requests_single_step(message):
            return 60
        return 90

    @classmethod
    def _enforce_reply_shape(cls, *, reply: str, plan: PlannerResult, message: str) -> str:
        compacted = cls._humanize_reply(reply=reply, message=message)
        compacted = cls._compact_to_sentence_limit(
            compacted,
            cls._reply_sentence_limit(plan, message),
        )
        return cls._trim_to_word_limit(compacted, cls._reply_word_limit(message))

    @classmethod
    def _build_fast_astrology_reply(
        cls,
        *,
        message: str,
        plan: PlannerResult,
        birth_details: dict[str, Any] | None,
        matchmaking_details: dict[str, Any] | None,
    ) -> str | None:
        if birth_details is not None or matchmaking_details is not None:
            return None
        greeting_reply = cls._build_fast_greeting_reply(message)
        if greeting_reply is not None:
            return greeting_reply
        if plan.action != "respond_only":
            return None

        tokens = cls._message_tokens(message)
        has_personal_guidance = bool(tokens & cls.LIFE_GUIDANCE_TOKENS) and bool(
            tokens & cls.PERSONAL_TOKENS
        )
        if not has_personal_guidance:
            return None

        response_language = cls._infer_response_language(message)
        if tokens & {"career", "job", "work", "finance", "money"}:
            if response_language == "hinglish":
                return (
                    "Career pressure aksar delay, direction, ya confidence ke phase ko dikhata hai, permanent blockage ko nahi. "
                    "Astrology mein iske liye 10th house, Saturn, aur current timing dekhe jaate hain. "
                    "Agar aap exact chart-based answer chahte hain, to date, time, aur place of birth share kijiye."
                )
            return (
                "Career pressure usually shows a phase of delay, unclear direction, or low confidence rather than a permanent block. "
                "In astrology, the 10th house, Saturn, and current timing are usually checked for this. "
                "If you want an exact chart-based answer, share your date, time, and place of birth."
            )
        if tokens & {"relationship", "marriage", "love", "partner"}:
            if response_language == "hinglish":
                return (
                    "Relationship tension aksar clarity, trust, ya timing ke imbalance ko dikhati hai. "
                    "Astrology mein Venus, Moon, 7th house, aur current period dekhe jaate hain. "
                    "Agar aap exact chart-based answer chahte hain, to date, time, aur place of birth share kijiye."
                )
            return (
                "Relationship tension usually points to imbalance in clarity, trust, or timing. "
                "In astrology, Venus, the Moon, the 7th house, and the current period are usually checked for this. "
                "If you want an exact chart-based answer, share your date, time, and place of birth."
            )
        if tokens & {"timing", "future"}:
            if response_language == "hinglish":
                return (
                    "Future ya timing ka answer general level par trend batata hai, final certainty nahi. "
                    "Astrology mein current period aur transit ka effect sabse pehle dekha jaata hai. "
                    "Agar aap exact chart-based answer chahte hain, to date, time, aur place of birth share kijiye."
                )
            return (
                "Questions about future timing usually show trends, not absolute certainty. "
                "In astrology, the current period and transits are checked first for this. "
                "If you want an exact chart-based answer, share your date, time, and place of birth."
            )

        if response_language == "hinglish":
            return (
                "Main aapko general astrology guidance de sakta hoon, lekin personal chart reading ke liye birth details chahiye hongi. "
                "Agar aap exact answer chahte hain, to date, time, aur place of birth share kijiye."
            )
        return (
            "I can give you general astrology guidance, but for a personal chart reading I would need your birth details. "
            "If you want an exact answer, share your date, time, and place of birth."
        )

    @classmethod
    def _build_birth_details_capture_reply(
        cls,
        message: str,
        birth_detail_parts: dict[str, Any] | None = None,
    ) -> str | None:
        parts = birth_detail_parts or cls._extract_birth_detail_parts(message)
        if not cls._has_birth_detail_fragment(parts):
            return None

        place = parts.get("place")
        has_date = parts.get("date_parts") is not None
        has_time = parts.get("time_parts") is not None
        has_place = place is not None
        missing_fields: list[str] = []
        if not has_date:
            missing_fields.append("date")
        if not has_time:
            missing_fields.append("time")
        if not has_place:
            missing_fields.append("place")

        if cls._infer_response_language(message) == "hinglish":
            if not missing_fields:
                return (
                    f"Maine aapki birth details le li hain, including {place}. "
                    "Agar app city ko exact match nahi kar pa rahi ho, to place ko thoda aur specific likhiye, jaise city, state, country."
                )
            if missing_fields == ["place"]:
                return (
                    "Maine aapki date aur time le liya hai. "
                    "Ab exact chart ke liye birthplace ko thoda aur specific likhiye, jaise city, state, country."
                )
            if missing_fields == ["time"]:
                return "Maine aapki date aur place le liya hai. Ab birth time bhi bhej dijiye."
            if missing_fields == ["date"]:
                return "Maine aapka time aur place note kar liya hai. Ab date of birth bhi bhej dijiye."
            if missing_fields == ["time", "place"]:
                return "Maine aapki date of birth le li hai. Ab birth time aur place bhi bhej dijiye."
            if missing_fields == ["date", "place"]:
                return "Maine aapka birth time le liya hai. Ab date of birth aur place bhi bhej dijiye."
            if missing_fields == ["date", "time"]:
                return "Maine aapka birthplace note kar liya hai. Ab date of birth aur birth time bhi bhej dijiye."
            return (
                "Maine aapki details ka jo hissa mila hai wo note kar liya hai. "
                "Ab date, time, aur place of birth complete bhej dijiye."
            )

        if not missing_fields:
            return (
                f"I have your birth details, including {place}. "
                "If the app cannot resolve the birthplace exactly, send the place a bit more specifically, like city, state, country."
            )
        if missing_fields == ["place"]:
            return (
                "I have your date and time. "
                "For an exact chart, send the birthplace a bit more specifically, like city, state, country."
            )
        if missing_fields == ["time"]:
            return "I have your date and birthplace. Please share your birth time as well."
        if missing_fields == ["date"]:
            return "I have your birth time and birthplace. Please share your date of birth as well."
        if missing_fields == ["time", "place"]:
            return "I have your date of birth. Please share your birth time and birthplace as well."
        if missing_fields == ["date", "place"]:
            return "I have your birth time. Please share your date of birth and birthplace as well."
        if missing_fields == ["date", "time"]:
            return "I have your birthplace. Please share your date of birth and birth time as well."
        return "Please share your date, time, and place of birth so I can check the chart properly."

    @classmethod
    def _build_tool_backed_reply(
        cls,
        *,
        message: str,
        plan: PlannerResult,
        tool_outputs: list[dict[str, Any]],
    ) -> str | None:
        if not tool_outputs:
            return None

        response_language = cls._infer_response_language(message)

        def english(available: str, unavailable: str, *, has_results: bool) -> str:
            return available if has_results else unavailable

        def hinglish(available: str, unavailable: str, *, has_results: bool) -> str:
            return available if has_results else unavailable

        for output in tool_outputs:
            tool_name = output.get("tool")
            if tool_name == "show_kundali" and plan.action == "show_kundali":
                if response_language == "hinglish":
                    return (
                        "Aapki kundli summary neeche hai. "
                        "Main ab ise career, marriage, ya timing ke hisaab se seedhe samjha deta hoon."
                    )
                return (
                    "Your kundali summary is below. "
                    "I can now explain it simply for career, marriage, or timing."
                )

            if tool_name == "matchmaking" and plan.action == "matchmaking":
                if response_language == "hinglish":
                    return (
                        "Aapka matchmaking result neeche hai. "
                        "Main ab strongest aur weakest compatibility ko simple words mein samjha deta hoon."
                    )
                return (
                    "Your matchmaking result is below. "
                    "I can also explain the strongest and weakest compatibility in simple terms."
                )

            if tool_name == "book_pooja" and plan.action == "book_pooja":
                has_results = bool(
                    output.get("home_puja_services")
                    or output.get("temple_services")
                    or output.get("pandits")
                )
                if response_language == "hinglish":
                    return hinglish(
                        (
                            "Neeche jo puja options dikh rahe hain, ye is need ke sabse kareeb hain. "
                            "Main inhe budget, location, ya purpose ke hisaab se aur narrow kar sakta hoon."
                        ),
                        (
                            "Abhi exact puja match nahi mila. "
                            "Main ise home puja, temple service, ya pandit consultation tak broaden kar sakta hoon."
                        ),
                        has_results=has_results,
                    )
                return english(
                    (
                        "The puja options below are the closest fit for this need. "
                        "I can narrow them further by budget, location, or purpose."
                    ),
                    (
                        "I could not find an exact puja match yet. "
                        "I can broaden this to home puja, temple services, or pandit consultation."
                    ),
                    has_results=has_results,
                )

            if tool_name == "recommend_product" and plan.action == "recommend_product":
                has_results = bool(output.get("items"))
                if response_language == "hinglish":
                    return hinglish(
                        (
                            "Neeche jo options dikh rahe hain, ye is need ke sabse relevant hain. "
                            "Main inhe purpose ya budget ke hisaab se aur narrow kar sakta hoon."
                        ),
                        (
                            "Abhi exact catalog match nahi mila. "
                            "Main search ko broader kar sakta hoon, ya pehle pandit consultation suggest kar sakta hoon."
                        ),
                        has_results=has_results,
                    )
                return english(
                    (
                        "The options below are the closest fit for this need. "
                        "I can narrow them further by purpose or budget."
                    ),
                    (
                        "I could not find an exact catalog match yet. "
                        "I can broaden the search or suggest a pandit consultation first."
                        ),
                        has_results=has_results,
                    )
            if tool_name == "recommend_product" and output.get("soft_recommendation"):
                # Soft products should not generate their own intro text.
                # The LLM will weave it in as a brief aside. Return None to skip.
                return None

            if tool_name == "suggest_consultant" and plan.action == "suggest_consultant":
                has_results = bool(output.get("items"))
                if response_language == "hinglish":
                    return hinglish(
                        (
                            "Neeche jo pandits dikh rahe hain, ye is baat ke liye sabse relevant lagte hain. "
                            "Main inhe specialty, language, ya budget ke hisaab se aur narrow kar sakta hoon."
                        ),
                        (
                            "Abhi exact pandit match nahi mila. "
                            "Main broader specialty ya general consultation options dikha sakta hoon."
                        ),
                        has_results=has_results,
                    )
                return english(
                    (
                        "The pandits below are the closest fit for this need. "
                        "I can narrow the list further by specialty, language, or budget."
                    ),
                    (
                        "I could not find an exact pandit match yet. "
                        "I can broaden the specialty or show general consultation options."
                    ),
                    has_results=has_results,
                )

        return None

    @classmethod
    def _build_response_style_context(
        cls,
        *,
        message: str,
        plan: PlannerResult,
        tool_outputs: list[dict[str, Any]],
        route_name: str | None = None,
        emotion: Any | None = None,
    ) -> str:
        composer_emotion = emotion or detect_emotion(message)
        route_value = route_name or ("CLARIFICATION" if plan.action == "ask_clarification" else "FAST_CHAT")
        style = build_style_instruction(
            message=message,
            emotion=composer_emotion,
            route=route_value,
            intent=plan.action,
            tool_outputs=tool_outputs,
        )
        extra_lines: list[str] = []
        response_language = cls._infer_response_language(message)
        extra_lines.append(
            "Reply in plain English only." if response_language == "english" else "Reply in natural Hinglish."
        )
        extra_lines.append(
            "If you can clearly see a repeating life loop, name it directly in one line before the advice. "
            "Use grounded phrasing like 'I keep seeing the same pattern here' or "
            "'This looks like a repeating cycle, not a random bad day.'"
        )
        has_soft_product_output = any(
            output.get("tool") == "recommend_product" and output.get("soft_recommendation")
            for output in tool_outputs
        )
        empty_explicit_product_output = any(
            output.get("tool") == "recommend_product"
            and not output.get("soft_recommendation")
            and not output.get("items")
            for output in tool_outputs
        )
        empty_consultant_output = any(
            output.get("tool") == "suggest_consultant" and not output.get("items")
            for output in tool_outputs
        )
        empty_booking_output = any(
            output.get("tool") == "book_pooja"
            and not output.get("home_puja_services")
            and not output.get("temple_services")
            and not output.get("pandits")
            for output in tool_outputs
        )
        if has_soft_product_output:
            extra_lines.append(
                "A supportive product option is available but do NOT make it the focus. "
                "Address the user's concern fully first. Only mention the product in one brief aside sentence at most, "
                "framed as 'if you feel drawn to extra support, this is available' — never as a recommendation or call to action."
            )
        elif plan.action != "recommend_product":
            extra_lines.append("Do not introduce rudraksha, bracelets, or catalog products on your own.")
        if empty_explicit_product_output:
            extra_lines.append(
                "The catalog returned no exact product items. Say that plainly. Do not imply that a specific "
                "rudraksha, bracelet, or mala is available right now. Give the astrological reasoning first, "
                "then offer to broaden the search or suggest a pandit consultation."
            )
        if empty_consultant_output:
            extra_lines.append(
                "No consultant matches were returned. Say that plainly. Do not invent pandit names, experience, "
                "or availability. Offer a broader consultant category or general guidance."
            )
        if empty_booking_output:
            extra_lines.append(
                "No booking matches were returned. Say that plainly. Do not invent puja services, temple names, "
                "or pandit availability. Offer to broaden the search."
            )
        if cls._message_declines_products(message):
            extra_lines.append("The user does not want product suggestions right now.")
        if cls._message_requests_single_step(message):
            extra_lines.append("Give exactly one practical next step.")
        return "\n".join([style, *extra_lines])

    @classmethod
    def _postprocess_reply(cls, *, reply: str, plan: PlannerResult, message: str) -> str:
        compacted = " ".join(reply.split())
        lowered_message = message.lower()
        lowered_reply = compacted.lower()

        if plan.action == "respond_only":
            compacted = compacted.replace("Let us understand this step by step.", "").strip()
            compacted = compacted.replace("Let's understand this step by step.", "").strip()
            lowered_reply = compacted.lower()

            compacted = cls._compact_to_sentence_limit(compacted, 2)
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

            compacted = cls._compact_to_sentence_limit(compacted, 2)

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

        if plan.action in {"recommend_product", "suggest_consultant", "book_pooja"}:
            compacted = cls._compact_to_sentence_limit(compacted, 2)

        return compacted

    @classmethod
    def _finalize_reply_text(cls, *, reply: str, plan: PlannerResult, message: str) -> str:
        return final_response_guardrail(
            cls._enforce_reply_shape(
                reply=cls._postprocess_reply(reply=reply, plan=plan, message=message),
                plan=plan,
                message=message,
            )
        )

    async def _prepare_base_reply_context(
        self,
        *,
        session_id: str,
        message: str,
        birth_details: dict[str, Any] | None = None,
        matchmaking_details: dict[str, Any] | None = None,
        current_user: AuthenticatedUser | None = None,
    ) -> dict[str, Any]:
        recent_messages = await asyncio.to_thread(
            self.memory_service.recent_messages,
            session_id,
            self.settings.MEMORY_WINDOW_SIZE,
        )
        message = self._normalize_user_message(message)
        session_state = _get_cached_session_context(session_id)
        if session_state is None:
            session_state = await asyncio.to_thread(
                self.memory_service.repository.get_session_state,
                session_id,
            ) or {}
            if session_state:
                _set_cached_session_context(session_id, session_state)
        assistant_requested_birth_details = self._assistant_requested_birth_details(recent_messages)
        provided_matchmaking_details = matchmaking_details is not None
        if matchmaking_details is not None:
            _set_cached_matchmaking_details(session_id, matchmaking_details)
        else:
            matchmaking_details = _get_cached_matchmaking_details(session_id)
        effective_birth_details = birth_details
        partial_birth_details = _get_cached_partial_birth_details(session_id)
        if effective_birth_details is None:
            cached_birth_details = session_state.get("birth_details")
            if isinstance(cached_birth_details, dict) and cached_birth_details:
                effective_birth_details = cached_birth_details
        if partial_birth_details is None:
            cached_partial_birth_details = session_state.get("partial_birth_details")
            if isinstance(cached_partial_birth_details, dict) and cached_partial_birth_details:
                partial_birth_details = cached_partial_birth_details
        if matchmaking_details is None:
            cached_matchmaking_details = session_state.get("matchmaking_details")
            if isinstance(cached_matchmaking_details, dict) and cached_matchmaking_details:
                matchmaking_details = cached_matchmaking_details
        if partial_birth_details and not self._looks_like_bare_birth_place_text(partial_birth_details.get("place")):
            partial_birth_details = {
                **partial_birth_details,
                "place": None,
            }
            if self._has_birth_detail_fragment(partial_birth_details):
                _set_cached_partial_birth_details(session_id, partial_birth_details)
            else:
                _clear_cached_partial_birth_details(session_id)
        profile_birth_parts: dict[str, Any] | None = None
        if current_user is not None:
            profile_birth_parts = await self.core_service_client.get_user_birth_profile(
                current_user.user_id,
                current_user,
            )
            if profile_birth_parts is not None:
                partial_birth_details = self._merge_birth_detail_parts(
                    profile_birth_parts,
                    partial_birth_details,
                )
        message_birth_parts = self._extract_birth_detail_parts(
            message,
            allow_bare_place=assistant_requested_birth_details,
        )
        has_message_birth_details = self._has_birth_detail_fragment(message_birth_parts)
        birth_details_followup = assistant_requested_birth_details and has_message_birth_details
        if (
            not birth_details_followup
            and assistant_requested_birth_details
            and self._message_acknowledges_shared_birth_details(message)
            and self._has_birth_detail_fragment(partial_birth_details)
        ):
            birth_details_followup = True
        if effective_birth_details is None and (birth_details_followup or has_message_birth_details):
            try:
                merged_birth_parts = self._merge_birth_detail_parts(
                    partial_birth_details,
                    message_birth_parts,
                )
                effective_birth_details = await self._infer_birth_details_from_parts(
                    merged_birth_parts
                )
                if effective_birth_details is not None:
                    _clear_cached_partial_birth_details(session_id)
                    partial_birth_details = None
                elif self._has_birth_detail_fragment(merged_birth_parts):
                    partial_birth_details = merged_birth_parts
                    _set_cached_partial_birth_details(session_id, merged_birth_parts)
            except Exception as exc:
                logger.info("Birth detail inference failed for session %s: %s", session_id, exc)
        if current_user is not None and has_message_birth_details:
            partial_birth_payload = self._merge_birth_detail_parts(
                profile_birth_parts,
                self._merge_birth_detail_parts(partial_birth_details, message_birth_parts),
            )
            saved_partial_birth_details = await self.core_service_client.save_user_birth_profile(
                current_user.user_id,
                partial_birth_payload,
                current_user,
            )
            if saved_partial_birth_details is not None:
                partial_birth_details = self._merge_birth_detail_parts(
                    saved_partial_birth_details,
                    partial_birth_details,
                )
                if self._has_birth_detail_fragment(partial_birth_details):
                    _set_cached_partial_birth_details(session_id, partial_birth_details)
        # Check session-level cache for previously shared birth details
        if effective_birth_details is None:
            effective_birth_details = _get_cached_birth_details(session_id)
        if effective_birth_details is None and current_user is not None:
            effective_birth_details = await self.core_service_client.get_user_birth_details(
                current_user.user_id,
                current_user,
            )
        # Persist birth details: save to session cache + core-service
        if effective_birth_details is not None:
            _set_cached_birth_details(session_id, effective_birth_details)
            _clear_cached_partial_birth_details(session_id)
            if current_user is not None and (
                birth_details_followup
                or self._has_birth_detail_fragment(message_birth_parts)
                or birth_details is not None
            ):
                saved_birth_details = await self.core_service_client.save_user_birth_details(
                    current_user.user_id,
                    effective_birth_details,
                    current_user,
                )
                if saved_birth_details is not None:
                    effective_birth_details = saved_birth_details
        message = sanitize_user_input(message)
        pre_guardrail = pre_scope_guardrail(message)
        # Lightweight initial route: guardrail check + always defer to ToolCallPlanner
        if not pre_guardrail.allowed:
            route_decision = ChatRouteDecision(
                route="BLOCKED",
                intent="respond_only",
                confidence=0.99,
                risk_level=pre_guardrail.risk_level,
                reason=pre_guardrail.reason,
                should_call_tool=False,
            )
        else:
            route_decision = ChatRouteDecision(
                route="FAST_CHAT",
                intent="respond_only",
                confidence=0.5,
                risk_level="low",
                reason="awaiting_planner",
                should_call_tool=False,
                needs_planner=True,
            )
        route_decision = self._override_route_from_session_state(
            message=message,
            recent_messages=recent_messages,
            route_decision=route_decision,
            session_state=session_state,
            effective_birth_details=effective_birth_details,
            partial_birth_details=partial_birth_details,
            matchmaking_details=matchmaking_details,
            provided_matchmaking_details=provided_matchmaking_details,
        )
        needs_birth_details = (
            current_user is not None
            and effective_birth_details is None
            and route_decision.intent == "show_kundali"
        )
        birth_details_capture_pending = (
            effective_birth_details is None
            and self._has_birth_detail_fragment(partial_birth_details)
            and (birth_details_followup or route_decision.intent == "show_kundali")
        )
        metadata_json = None
        if current_user is not None:
            metadata_json = self._safe_json_dumps({"user_id": current_user.user_id, "role": current_user.role})
        internal_user_id = self._resolve_internal_user_id(current_user)
        emotion = detect_emotion(message)

        if birth_details_followup and effective_birth_details is not None:
            route_decision = ChatRouteDecision(
                route="TOOL_FLOW",
                intent="show_kundali",
                confidence=0.99,
                risk_level="low",
                reason="birth_details_followup",
                should_call_tool=True,
            )

        if route_decision.route == "BLOCKED":
            plan = self._plan_from_route(route_decision)
            scope_guardrail = {"allowed": False, "reason": route_decision.reason}
            tool_guardrail = {"allowed": False, "reason": "route_blocked"}
            route = pick_model_route(plan)
            return {
                "plan": plan,
                "route": route,
                "emotion": emotion,
                "messages": [],
                "tool_outputs": [],
                "retrieval_matches": [],
                "kundali_chart": None,
                "kundali_summary": None,
                "matchmaking_result": None,
                "metadata_json": metadata_json,
                "message": message,
                "session_id": session_id,
                "scope_guardrail": scope_guardrail,
                "tool_guardrail": tool_guardrail,
                "tool_execution_allowed": False,
                "birth_details_followup": birth_details_followup,
                "birth_details_capture_pending": birth_details_capture_pending,
                "partial_birth_details": partial_birth_details,
                "effective_birth_details": effective_birth_details,
                "session_state": session_state,
                "recent_messages": recent_messages,
                "internal_user_id": internal_user_id,
                "route_decision": route_decision,
                "normalized_message": message,
                "deferred_planner": False,
                "needs_birth_details": needs_birth_details,
                "matchmaking_details": matchmaking_details,
                "current_user": current_user,
            }

        plan = self._plan_from_route(route_decision)
        # Defer planner LLM call to _complete_reply_context where it runs
        # in parallel with RAG + memory for lower latency
        deferred_planner = route_decision.needs_planner
        scope_guardrail = {"allowed": True, "reason": route_decision.reason}
        tool_guardrail = self._tool_guardrail_decision(
            plan,
            message=message,
            birth_details=effective_birth_details,
            matchmaking_details=matchmaking_details,
        )
        if route_decision.route != "TOOL_FLOW":
            tool_guardrail = {"allowed": False, "reason": "route_not_tool_flow"}
        plan = self._fallback_plan_for_threshold_rejection(plan, tool_guardrail)
        tool_execution_allowed = bool(tool_guardrail.get("allowed"))
        route = pick_model_route(plan.model_copy(update={"should_call_tool": tool_execution_allowed}))

        logger.info(
            "Planner decision evaluated",
            extra={
                "extra_fields": {
                    "session_id": session_id,
                    "route_name": route_decision.route,
                    "planner_action": plan.action,
                    "planner_confidence": plan.confidence,
                    "planner_should_call_tool": plan.should_call_tool,
                    "planner_arguments": plan.arguments,
                    "planner_missing_information": plan.missing_information,
                    "scope_guardrail": scope_guardrail,
                    "tool_guardrail": tool_guardrail,
                    "resolved_route_model": route.model,
                    "resolved_reasoning_profile": route.reasoning_profile,
                }
            },
        )

        return {
            "plan": plan,
            "route": route,
            "emotion": emotion,
            "messages": [],
            "tool_outputs": [],
            "retrieval_matches": [],
            "kundali_chart": None,
            "kundali_summary": None,
            "matchmaking_result": None,
            "metadata_json": metadata_json,
            "message": message,
            "session_id": session_id,
            "scope_guardrail": scope_guardrail,
            "tool_guardrail": tool_guardrail,
            "tool_execution_allowed": tool_execution_allowed,
            "birth_details_followup": birth_details_followup,
            "birth_details_capture_pending": birth_details_capture_pending,
            "partial_birth_details": partial_birth_details,
            "effective_birth_details": effective_birth_details,
            "session_state": session_state,
            "recent_messages": recent_messages,
            "internal_user_id": internal_user_id,
            "route_decision": route_decision,
            "normalized_message": message,
            "deferred_planner": deferred_planner,
            "needs_birth_details": needs_birth_details,
            "matchmaking_details": matchmaking_details,
            "current_user": current_user,
        }

    async def _complete_reply_context(
        self,
        context: dict[str, Any],
        *,
        birth_details: dict[str, Any] | None = None,
        matchmaking_details: dict[str, Any] | None = None,
        current_user: AuthenticatedUser | None = None,
    ) -> dict[str, Any]:
        if not context["scope_guardrail"]["allowed"]:
            return context

        session_id = context["session_id"]
        message = context["message"]
        plan = context["plan"]
        recent_messages = context["recent_messages"]
        compact_recent_messages = self._compact_recent_messages(recent_messages)
        compact_session_context = self._format_compact_session_context(context.get("session_state"))
        internal_user_id = context["internal_user_id"]
        tool_execution_allowed = bool(context["tool_execution_allowed"])
        route_decision: ChatRouteDecision = context["route_decision"]
        normalized_message = context["normalized_message"]
        deferred_planner = bool(context.get("deferred_planner"))
        effective_birth_details = context.get("effective_birth_details")
        planner_query = self._planner_search_query(plan)
        needs_rag = self._needs_rag(route_decision, plan)
        top_k = self.settings.FAST_RAG_TOP_K if route_decision.route == "FAST_CHAT" else self.settings.RAG_TOP_K
        empty_rag_payload = {"chunks": [], "knowledge_chunks": [], "policy_chunks": [], "retrieval_metadata": {}}

        # Run planner + memory + optional chart-context work in parallel.
        parallel_tasks: list[Any] = [
            asyncio.to_thread(
                self.memory_service.long_term_context,
                session_id,
                user_id=internal_user_id,
            ),
        ]
        rag_task_index: int | None = None
        chart_context_task_index: int | None = None
        planner_task_index: int | None = None

        if needs_rag and effective_birth_details is not None:
            chart_context_task_index = len(parallel_tasks)
            parallel_tasks.append(self._compute_rag_chart_context(effective_birth_details))
        elif needs_rag:
            rag_task_index = len(parallel_tasks)
            parallel_tasks.append(
                asyncio.to_thread(
                    self.rag_service.retrieve_context_bundle,
                    normalized_message,
                    top_k,
                    action=plan.action,
                    planner_query=planner_query,
                    chart_context=None,
                )
            )
        if deferred_planner:
            planner_task_index = len(parallel_tasks)
            parallel_tasks.append(
                self.planner.plan(
                    message=message,
                    has_birth_details=effective_birth_details is not None,
                    has_matchmaking_details=matchmaking_details is not None,
                    is_authenticated=current_user is not None,
                )
            )

        results = await asyncio.gather(*parallel_tasks)
        long_term_context = results[0]
        rag_payload = results[rag_task_index] if rag_task_index is not None else empty_rag_payload
        rag_chart_context: dict[str, Any] | None = None
        precomputed_kundali_chart: dict[str, Any] | None = None
        transit_data: dict[str, Any] | None = None
        predictive_insights: list[dict[str, str]] | None = None
        if chart_context_task_index is not None:
            chart_context_payload = results[chart_context_task_index] or {}
            if isinstance(chart_context_payload, dict):
                rag_chart_context = chart_context_payload.get("rag_context")
                precomputed_kundali_chart = chart_context_payload.get("chart")
                transit_data = chart_context_payload.get("transit_data")
                predictive_insights = chart_context_payload.get("predictions")

        if needs_rag and rag_task_index is None:
            rag_payload = await asyncio.to_thread(
                self.rag_service.retrieve_context_bundle,
                normalized_message,
                top_k,
                action=plan.action,
                planner_query=planner_query,
                chart_context=rag_chart_context,
            )

        # If planner ran in parallel, update plan and re-evaluate guardrails
        if deferred_planner:
            assert planner_task_index is not None
            plan = results[planner_task_index]
            route_decision = self._route_decision_from_plan(route_decision, plan)
            tool_guardrail = self._tool_guardrail_decision(
                plan,
                message=message,
                birth_details=effective_birth_details,
                matchmaking_details=matchmaking_details,
            )
            if route_decision.route != "TOOL_FLOW":
                tool_guardrail = {"allowed": False, "reason": "route_not_tool_flow"}
            plan = self._fallback_plan_for_threshold_rejection(plan, tool_guardrail)
            tool_execution_allowed = bool(tool_guardrail.get("allowed"))
            route = pick_model_route(plan.model_copy(update={"should_call_tool": tool_execution_allowed}))
            context["plan"] = plan
            context["tool_guardrail"] = tool_guardrail
            context["tool_execution_allowed"] = tool_execution_allowed
            context["route"] = route
            context["route_decision"] = route_decision

            # Re-run RAG with planner search query if available
            planner_query = self._planner_search_query(plan)
            updated_needs_rag = self._needs_rag(route_decision, plan)
            if (
                updated_needs_rag
                and effective_birth_details is not None
                and rag_chart_context is None
            ):
                chart_context_payload = await self._compute_rag_chart_context(effective_birth_details)
                rag_chart_context = chart_context_payload.get("rag_context")
                precomputed_kundali_chart = chart_context_payload.get("chart")
                transit_data = chart_context_payload.get("transit_data")
                predictive_insights = chart_context_payload.get("predictions")
            if updated_needs_rag:
                rag_payload = await asyncio.to_thread(
                    self.rag_service.retrieve_context_bundle,
                    normalized_message,
                    self.settings.FAST_RAG_TOP_K if route_decision.route == "FAST_CHAT" else self.settings.RAG_TOP_K,
                    action=plan.action,
                    planner_query=planner_query,
                    chart_context=rag_chart_context,
                )
        retrieval_matches = list(rag_payload.get("chunks") or [])
        retrieval_knowledge_matches = list(rag_payload.get("knowledge_chunks") or [])
        retrieval_policy_matches = list(rag_payload.get("policy_chunks") or [])
        retrieval_metadata = dict(rag_payload.get("retrieval_metadata") or {})
        recommendation_context: dict[str, Any] = {
            "soft_product": {
                "eligible": False,
                "reason": "not_evaluated",
                "query": None,
            }
        }

        kundali_summary: str | None = None
        kundali_chart: dict[str, Any] | None = None
        if tool_execution_allowed and plan.action == "show_kundali" and birth_details is not None:
            kundali_chart = await self._safe_tool_result(
                self.core_service_client.generate_kundli(
                    birth_details,
                    current_user,
                ),
                timeout_seconds=self.settings.TOOL_TIMEOUT_SECONDS,
                default=None,
                tool_name="generate_kundli",
            )
            if kundali_chart is None:
                kundali_chart = precomputed_kundali_chart or await compute_full_chart(birth_details)
            kundali_summary = show_kundali(kundali_chart)

        matchmaking_result: dict[str, Any] | None = None
        if tool_execution_allowed and plan.action == "matchmaking" and matchmaking_details is not None:
            matchmaking_result = await self._safe_tool_result(
                self.core_service_client.generate_matchmaking(
                    matchmaking_details,
                    current_user,
                ),
                timeout_seconds=self.settings.TOOL_TIMEOUT_SECONDS,
                default=None,
                tool_name="generate_matchmaking",
            )

        tool_outputs: list[dict[str, Any]] = []
        matchmaking_output = self._build_matchmaking_tool_output(matchmaking_result or {})
        if matchmaking_output is not None:
            tool_outputs.append(matchmaking_output)

        search_query = self._planner_search_query(plan)
        if tool_execution_allowed and plan.action == "book_pooja" and search_query is not None:
            home_puja_services, temple_services, public_pandits = await asyncio.gather(
                self._safe_tool_result(
                    self.core_service_client.list_home_puja_services(search_query),
                    timeout_seconds=self.settings.TOOL_TIMEOUT_SECONDS,
                    default=[],
                    tool_name="list_home_puja_services",
                ),
                self._safe_tool_result(
                    self.core_service_client.list_temple_services(search_query),
                    timeout_seconds=self.settings.TOOL_TIMEOUT_SECONDS,
                    default=[],
                    tool_name="list_temple_services",
                ),
                self._safe_tool_result(
                    self.core_service_client.list_public_pandits(search_query),
                    timeout_seconds=self.settings.TOOL_TIMEOUT_SECONDS,
                    default=[],
                    tool_name="list_public_pandits",
                ),
            )
            booking_output = self._build_booking_tool_output(
                home_puja_services,
                temple_services,
                public_pandits,
            )
            if booking_output is not None:
                tool_outputs.append(booking_output)
            else:
                tool_outputs.append(self._build_empty_booking_tool_output(search_query))
        if tool_execution_allowed and plan.action == "recommend_product" and search_query is not None:
            _afflicted = self._identify_afflicted_planets(rag_chart_context) if rag_chart_context else None
            _cur_dasha = (rag_chart_context or {}).get("current_mahadasha")
            product_output = await self._lookup_product_tool_output(
                search_query=search_query,
                kundali_summary=kundali_summary,
                include_empty=True,
                soft_recommendation=False,
                afflicted_planets=_afflicted,
                current_dasha=_cur_dasha,
            )
            if product_output is not None:
                tool_outputs.append(product_output)
        if tool_execution_allowed and plan.action == "suggest_consultant" and search_query is not None:
            consultant_results = await self._safe_tool_result(
                self._find_consultant_results(
                    search_query,
                    current_user,
                ),
                timeout_seconds=self.settings.TOOL_TIMEOUT_SECONDS,
                default=[],
                tool_name="find_consultant",
            )
            consultant_output = self._build_consultant_tool_output(
                consultant_results,
                kundali_summary=kundali_summary,
            )
            if consultant_output is not None:
                tool_outputs.append(consultant_output)
            else:
                tool_outputs.append(self._build_empty_consultant_tool_output(search_query))
        soft_product_decision = self._soft_product_decision(
            message=message,
            plan=plan,
            retrieval_policy_matches=retrieval_policy_matches,
            kundali_summary=kundali_summary,
            chart_context=rag_chart_context,
        )
        recommendation_context["soft_product"] = {
            "eligible": bool(soft_product_decision.get("allowed")),
            "reason": soft_product_decision.get("reason"),
            "query": soft_product_decision.get("query"),
        }
        if soft_product_decision.get("allowed"):
            soft_product_query = soft_product_decision.get("query")
            if isinstance(soft_product_query, str) and soft_product_query.strip():
                soft_product_guard = tool_specific_guardrail(
                    "recommend_product",
                    message,
                    {"search_query": soft_product_query},
                )
                if soft_product_guard.allowed:
                    normalized_query = soft_product_guard.normalized_args.get("search_query")
                    if isinstance(normalized_query, str) and normalized_query.strip():
                        _soft_afflicted = self._identify_afflicted_planets(rag_chart_context) if rag_chart_context else None
                        _soft_dasha = (rag_chart_context or {}).get("current_mahadasha")
                        soft_product_output = await self._lookup_product_tool_output(
                            search_query=normalized_query,
                            kundali_summary=kundali_summary,
                            include_empty=False,
                            soft_recommendation=True,
                            afflicted_planets=_soft_afflicted,
                            current_dasha=_soft_dasha,
                        )
                        if soft_product_output is not None:
                            recommendation_context["soft_product"]["reason"] = "soft_product_added"
                            tool_outputs.append(soft_product_output)
                        else:
                            recommendation_context["soft_product"]["reason"] = "no_catalog_results"
                else:
                    recommendation_context["soft_product"]["reason"] = soft_product_guard.reason
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
            retrieval_context=self._format_retrieval_knowledge_context(
                retrieval_knowledge_matches or retrieval_matches
            ),
            tool_context=self._format_tool_context(tool_outputs),
            retrieval_policy_context=self._format_retrieval_policy_context(
                retrieval_policy_matches or retrieval_matches
            ),
            chart_summary=self._format_chart_context_for_prompt(rag_chart_context),
            transit_summary=format_transits_for_prompt(transit_data) if transit_data else None,
            prediction_summary=format_predictions_for_prompt(predictive_insights) if predictive_insights else None,
            pattern_summary=build_pattern_summary(
                long_term_context=long_term_context,
                recent_messages=recent_messages,
                transit_data=transit_data,
                predictions=predictive_insights,
            ),
        )
        messages = [{"role": "system", "content": persona_prompt}]
        messages.append(
            {
                "role": "system",
                "content": self._build_response_style_context(
                    message=message,
                    plan=plan,
                    tool_outputs=tool_outputs,
                    route_name=route_decision.route,
                    emotion=context["emotion"],
                ),
            }
        )
        messages.append(
            {
                "role": "system",
                "content": self._format_planner_context(plan, tool_execution_allowed),
            }
        )
        if compact_session_context is not None:
            messages.append({"role": "system", "content": compact_session_context})
        messages.extend(compact_recent_messages)
        messages.append({"role": "user", "content": message})

        enriched = dict(context)
        enriched.update(
            {
                "messages": messages,
                "compact_recent_messages": compact_recent_messages,
                "compact_session_context": compact_session_context,
                "tool_outputs": tool_outputs,
                "retrieval_matches": retrieval_matches,
                "retrieval_knowledge_matches": retrieval_knowledge_matches,
                "retrieval_policy_matches": retrieval_policy_matches,
                "retrieval_metadata": retrieval_metadata,
                "recommendation_context": recommendation_context,
                "kundali_chart": kundali_chart,
                "kundali_summary": kundali_summary,
                "matchmaking_result": matchmaking_result,
            }
        )
        return enriched

    async def _prepare_reply_context(
        self,
        *,
        session_id: str,
        message: str,
        birth_details: dict[str, Any] | None = None,
        matchmaking_details: dict[str, Any] | None = None,
        current_user: AuthenticatedUser | None = None,
    ) -> dict[str, Any]:
        context = await self._prepare_base_reply_context(
            session_id=session_id,
            message=message,
            birth_details=birth_details,
            matchmaking_details=matchmaking_details,
            current_user=current_user,
        )
        return await self._complete_reply_context(
            context,
            birth_details=context["effective_birth_details"],
            matchmaking_details=context["matchmaking_details"],
            current_user=current_user,
        )

    def _persist_chat_turns(
        self,
        context: dict[str, Any],
        reply: str,
        response_metadata: dict[str, Any] | None = None,
        *,
        partial: bool = False,
    ) -> None:
        plan = context["plan"]
        route = context["route"]
        route_decision: ChatRouteDecision = context["route_decision"]
        prompt_versions = None
        model_used = None
        route_taken = None
        tool_called = None
        variant_id = None
        total_tokens_input = None
        total_tokens_output = None
        latency_ms = None
        if isinstance(response_metadata, dict):
            prompt_versions = response_metadata.get("prompt_versions")
            model_used_value = response_metadata.get("model_used")
            route_taken_value = response_metadata.get("route_taken")
            tool_called_value = response_metadata.get("tool_called")
            variant_id_value = response_metadata.get("variant_id")
            total_tokens_input_value = response_metadata.get("total_tokens_input")
            total_tokens_output_value = response_metadata.get("total_tokens_output")
            latency_ms_value = response_metadata.get("latency_ms")
            model_used = model_used_value if isinstance(model_used_value, str) else None
            route_taken = route_taken_value if isinstance(route_taken_value, str) else None
            if isinstance(tool_called_value, str):
                tool_called = tool_called_value
            elif isinstance(tool_called_value, list):
                tool_called = ",".join(
                    item for item in tool_called_value if isinstance(item, str) and item.strip()
                ) or None
            variant_id = variant_id_value if isinstance(variant_id_value, str) else None
            total_tokens_input = (
                total_tokens_input_value if isinstance(total_tokens_input_value, int) else None
            )
            total_tokens_output = (
                total_tokens_output_value if isinstance(total_tokens_output_value, int) else None
            )
            latency_ms = latency_ms_value if isinstance(latency_ms_value, int) else None

        # Do NOT persist blocked turns to conversation history.
        # Blocked messages (self-harm, violence, curse, medical, etc.)
        # contaminate the LLM context on subsequent turns, causing the
        # next response to reference the blocked topic or adopt a
        # fearful/protective tone even when the user has moved on.
        if route_decision.route == "BLOCKED":
            return

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
            prompt_versions=prompt_versions,
            model_used=model_used,
            route_taken=route_taken,
            tool_called=tool_called,
            variant_id=variant_id,
            total_tokens_input=total_tokens_input,
            total_tokens_output=total_tokens_output,
            latency_ms=latency_ms,
            partial=partial,
            metadata_json=self._safe_json_dumps(
                {
                    "partial": partial,
                    "reasoning_profile": route.reasoning_profile,
                    "response_metadata": response_metadata or {},
                }
            ),
        )
        compact_session_state = self._build_compact_session_state(
            context=context,
            reply=reply,
            response_metadata=response_metadata,
            partial=partial,
        )
        _set_cached_session_context(context["session_id"], compact_session_state)
        self.memory_service.repository.save_session_state(
            context["session_id"],
            compact_session_state,
            user_id=context.get("internal_user_id"),
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
        context = await self._prepare_base_reply_context(
            session_id=session_id,
            message=message,
            birth_details=birth_details,
            matchmaking_details=matchmaking_details,
            current_user=current_user,
        )
        started_at = time.perf_counter()
        plan = context["plan"]
        route = context["route"]
        route_decision: ChatRouteDecision = context["route_decision"]
        effective_birth_details = context["effective_birth_details"]
        birth_details_followup = bool(context["birth_details_followup"])
        birth_details_capture_pending = bool(context["birth_details_capture_pending"])
        partial_birth_details = context.get("partial_birth_details")
        usage: dict[str, Any] | None = None

        if not context["scope_guardrail"]["allowed"]:
            reply = compose_blocked_reply(route_decision.reason, context["emotion"])
        else:
            birth_details_capture_reply = None
            if birth_details_capture_pending and effective_birth_details is None:
                birth_details_capture_reply = self._build_birth_details_capture_reply(
                    message,
                    partial_birth_details,
                )
            clarification_reply = None
            if route_decision.route == "CLARIFICATION":
                clarification_reply = compose_clarification_reply(
                    route_decision.intent,
                    route_decision.missing_slots,
                    context["emotion"],
                )
            # Keep greeting-only turns fast, but let substantive astrology
            # questions flow through RAG + LLM assembly.
            greeting_reply = self._build_fast_greeting_reply(message)
            if birth_details_capture_reply is not None:
                reply = birth_details_capture_reply
            elif clarification_reply is not None:
                reply = clarification_reply
            elif greeting_reply is not None:
                reply = greeting_reply
            else:
                context = await self._complete_reply_context(
                    context,
                    birth_details=effective_birth_details,
                    matchmaking_details=context["matchmaking_details"],
                    current_user=current_user,
                )
                if self.groq_client.is_configured:
                    try:
                        reply = await self.groq_client.generate(
                            context["messages"],
                            model=route.model,
                            session_id=session_id,
                            user_id=current_user.user_id if current_user is not None else None,
                            trace_metadata=self._llm_trace_metadata(context),
                        )
                    except Exception as llm_exc:
                        logger.error(
                            "llm_generate_failed | session=%s | error=%s",
                            session_id, llm_exc, exc_info=True,
                        )
                        reply = "I understand. Let me try again — could you rephrase your question?"
                    usage = self.groq_client.last_usage
                else:
                    tool_reply = None
                    if not birth_details_followup:
                        tool_reply = self._build_tool_backed_reply(
                            message=message,
                            plan=plan,
                            tool_outputs=context["tool_outputs"],
                        )
                    reply = tool_reply or self._build_local_reply(
                        plan,
                        context["emotion"].label,
                        context["kundali_summary"],
                        context["retrieval_matches"],
                        context["tool_outputs"],
                    )

        reply = self._finalize_reply_text(reply=reply, plan=plan, message=message)
        latency_ms = int((time.perf_counter() - started_at) * 1000)
        total_tokens_input, total_tokens_output = self._usage_token_counts(usage)
        response_payload = self._response_metadata(
            reply=reply,
            route_decision=route_decision,
            plan=plan,
            message=message,
            tool_outputs=context["tool_outputs"],
            latency_ms=latency_ms,
            model=route.model,
            total_tokens_input=total_tokens_input,
            total_tokens_output=total_tokens_output,
            needs_birth_details=bool(context.get("needs_birth_details")),
            retrieval_matches=context.get("retrieval_matches"),
            retrieval_knowledge_matches=context.get("retrieval_knowledge_matches"),
            retrieval_policy_matches=context.get("retrieval_policy_matches"),
            retrieval_metadata=context.get("retrieval_metadata"),
            recommendation_context=context.get("recommendation_context"),
        )
        self._log_product_recommendation_trace(
            session_id=session_id,
            route_decision=route_decision,
            plan=plan,
            trace=response_payload["metadata"]["product_recommendation_trace"],
        )
        self._verify_card_text_consistency(
            reply=reply,
            tool_outputs=context["tool_outputs"],
            conversation_id=session_id,
        )
        # Skip memory persistence for blocked turns — blocked content
        # (curse, violence, medical, etc.) contaminates emotion_trend
        # and last_concern facts, causing the next normal reply to
        # carry forward a fearful/protective tone.
        if route_decision.route != "BLOCKED":
            self._persist_lightweight_memory(
                session_id=session_id,
                message=context["message"],
                emotion_label=context["emotion"].label,
                birth_details=effective_birth_details,
                user_id=context.get("internal_user_id"),
            )

        self._persist_chat_turns(context, reply, response_payload["metadata"])

        # Fire-and-forget memory extraction every few turns
        # Use already-loaded recent_messages count to avoid extra DB query
        recent_count = len(context.get("recent_messages") or [])
        if recent_count >= 4 and recent_count % 2 == 0 and route_decision.route != "BLOCKED":
            asyncio.ensure_future(self._background_memory_extraction(
                session_id, user_id=context.get("internal_user_id"),
            ))

        return {
            "reply": reply,
            "intent": plan.action,
            "planner_confidence": plan.confidence,
            "planner_arguments": plan.arguments,
            "scope_guardrail": context["scope_guardrail"],
            "tool_guardrail": context["tool_guardrail"],
            "tool_outputs": context["tool_outputs"],
            "retrieval_matches": context["retrieval_matches"],
            "kundali_chart": context["kundali_chart"],
            "kundali_summary": context["kundali_summary"],
            "matchmaking_result": context["matchmaking_result"],
            "emotion": context["emotion"].label,
            "route": route,
            "response": response_payload,
        }

    async def stream_reply_events(
        self,
        *,
        session_id: str,
        message: str,
        birth_details: dict[str, Any] | None = None,
        matchmaking_details: dict[str, Any] | None = None,
        current_user: AuthenticatedUser | None = None,
        disconnect_checker: Callable[[], Awaitable[bool]] | None = None,
    ) -> AsyncGenerator[tuple[str, dict[str, Any]], None]:
        async def _is_disconnected() -> bool:
            if disconnect_checker is None:
                return False
            return await disconnect_checker()

        yield (
            "status",
            {
                "resolved_session_id": session_id,
                "stage": "started",
                "message": "checking your chart/context",
            },
        )

        if await _is_disconnected():
            logger.info(
                "client_disconnected_mid_stream",
                extra={"extra_fields": {"conversation_id": session_id}},
            )
            return

        context = await self._prepare_base_reply_context(
            session_id=session_id,
            message=message,
            birth_details=birth_details,
            matchmaking_details=matchmaking_details,
            current_user=current_user,
        )
        started_at = time.perf_counter()
        plan = context["plan"]
        route = context["route"]
        route_decision: ChatRouteDecision = context["route_decision"]
        effective_birth_details = context["effective_birth_details"]
        birth_details_followup = bool(context["birth_details_followup"])
        birth_details_capture_pending = bool(context["birth_details_capture_pending"])
        partial_birth_details = context.get("partial_birth_details")
        usage: dict[str, Any] | None = None

        def _emit_meta() -> tuple[str, dict[str, Any]]:
            current_plan: PlannerResult = context["plan"]
            current_rd: ChatRouteDecision = context["route_decision"]
            return (
                "meta",
                {
                    "resolved_session_id": session_id,
                    "intent": current_plan.action,
                    "planner_confidence": current_plan.confidence,
                    "planner_arguments": current_plan.arguments,
                    "scope_guardrail": context["scope_guardrail"],
                    "tool_guardrail": context["tool_guardrail"],
                    "tool_count": len(context["tool_outputs"]),
                    "tools_pending": bool(context["tool_execution_allowed"]),
                    "route": current_rd.route,
                    "risk_level": current_rd.risk_level,
                    "needs_birth_details": bool(context.get("needs_birth_details")),
                },
            )

        reply_parts: list[str] = []
        stream_interrupted = False
        if not context["scope_guardrail"]["allowed"]:
            yield _emit_meta()
            fallback_reply = self._finalize_reply_text(
                reply=compose_blocked_reply(route_decision.reason, context["emotion"]),
                plan=plan,
                message=message,
            )
            reply_parts.append(fallback_reply)
            for chunk in chunk_text(fallback_reply):
                if await _is_disconnected():
                    stream_interrupted = True
                    break
                yield ("message", {"delta": chunk})
        else:
            birth_details_capture_reply = None
            if birth_details_capture_pending and effective_birth_details is None:
                birth_details_capture_reply = self._build_birth_details_capture_reply(
                    message,
                    partial_birth_details,
                )
            clarification_reply = None
            if route_decision.route == "CLARIFICATION":
                clarification_reply = compose_clarification_reply(
                    route_decision.intent,
                    route_decision.missing_slots,
                    context["emotion"],
                )
            # Keep greeting-only turns fast, but let substantive astrology
            # questions flow through RAG + LLM assembly.
            greeting_reply = self._build_fast_greeting_reply(message)
            if birth_details_capture_reply is not None:
                yield _emit_meta()
                birth_details_capture_reply = self._finalize_reply_text(
                    reply=birth_details_capture_reply,
                    plan=plan,
                    message=message,
                )
                reply_parts.append(birth_details_capture_reply)
                for chunk in chunk_text(birth_details_capture_reply):
                    if await _is_disconnected():
                        stream_interrupted = True
                        break
                    yield ("message", {"delta": chunk})
            elif clarification_reply is not None:
                yield _emit_meta()
                clarification_reply = self._finalize_reply_text(
                    reply=clarification_reply,
                    plan=plan,
                    message=message,
                )
                reply_parts.append(clarification_reply)
                for chunk in chunk_text(clarification_reply):
                    if await _is_disconnected():
                        stream_interrupted = True
                        break
                    yield ("message", {"delta": chunk})
            elif greeting_reply is not None:
                yield _emit_meta()
                greeting_reply = self._finalize_reply_text(
                    reply=greeting_reply,
                    plan=plan,
                    message=message,
                )
                reply_parts.append(greeting_reply)
                for chunk in chunk_text(greeting_reply):
                    if await _is_disconnected():
                        stream_interrupted = True
                        break
                    yield ("message", {"delta": chunk})
            else:
                context = await self._complete_reply_context(
                    context,
                    birth_details=effective_birth_details,
                    matchmaking_details=context["matchmaking_details"],
                    current_user=current_user,
                )
                # Re-read plan/route after deferred planner ran
                plan = context["plan"]
                route = context["route"]
                route_decision = context["route_decision"]
                yield _emit_meta()
                if context["tool_outputs"]:
                    yield (
                        "status",
                        {
                            "resolved_session_id": session_id,
                            "stage": "tool_context_ready",
                            "tool_count": len(context["tool_outputs"]),
                        },
                    )
                    for output in context["tool_outputs"]:
                        if await _is_disconnected():
                            stream_interrupted = True
                            break
                        event_name = output.get("event_name")
                        if not isinstance(event_name, str):
                            continue

                        payload = {
                            key: value
                            for key, value in output.items()
                            if key not in {"tool", "event_name"}
                        }
                        yield (event_name, payload)
                    if stream_interrupted:
                        logger.info(
                            "client_disconnected_mid_stream",
                            extra={"extra_fields": {"conversation_id": session_id}},
                        )
                        return

                yield (
                    "status",
                    {
                        "resolved_session_id": session_id,
                        "stage": "generating",
                    },
                )

                if self.groq_client.is_configured:
                    # Stream LLM tokens directly for real-time response
                    llm_parts: list[str] = []
                    try:
                        async for delta in self.groq_client.stream_generate(
                            context["messages"],
                            model=route.model,
                            session_id=session_id,
                            user_id=current_user.user_id if current_user is not None else None,
                            trace_metadata=self._llm_trace_metadata(context),
                        ):
                            if await _is_disconnected():
                                stream_interrupted = True
                                break
                            llm_parts.append(delta)
                            yield ("message", {"delta": delta})
                    except Exception as llm_exc:
                        logger.error(
                            "llm_stream_failed | session=%s | error=%s",
                            session_id, llm_exc, exc_info=True,
                        )
                        if not llm_parts:
                            fallback = "I understand. Let me try again — could you rephrase your question?"
                            reply_parts.append(fallback)
                            yield ("message", {"delta": fallback})
                    if llm_parts:
                        llm_reply = self._finalize_reply_text(
                            reply="".join(llm_parts),
                            plan=plan,
                            message=message,
                        )
                        reply_parts.append(llm_reply)
                    usage = self.groq_client.last_usage
                else:
                    tool_reply = None
                    if not birth_details_followup:
                        tool_reply = self._build_tool_backed_reply(
                            message=message,
                            plan=plan,
                            tool_outputs=context["tool_outputs"],
                        )
                    fallback_reply = self._finalize_reply_text(
                        reply=tool_reply or self._build_local_reply(
                            plan,
                            context["emotion"].label,
                            context["kundali_summary"],
                            context["retrieval_matches"],
                            context["tool_outputs"],
                        ),
                        plan=plan,
                        message=message,
                    )
                    reply_parts.append(fallback_reply)
                    for chunk in chunk_text(fallback_reply):
                        if await _is_disconnected():
                            stream_interrupted = True
                            break
                        yield ("message", {"delta": chunk})

        reply = "".join(reply_parts)
        latency_ms = int((time.perf_counter() - started_at) * 1000)
        total_tokens_input, total_tokens_output = self._usage_token_counts(usage)
        response_payload = self._response_metadata(
            reply=reply,
            route_decision=route_decision,
            plan=plan,
            message=message,
            tool_outputs=context["tool_outputs"],
            latency_ms=latency_ms,
            model=route.model,
            total_tokens_input=total_tokens_input,
            total_tokens_output=total_tokens_output,
            needs_birth_details=bool(context.get("needs_birth_details")),
            retrieval_matches=context.get("retrieval_matches"),
            retrieval_knowledge_matches=context.get("retrieval_knowledge_matches"),
            retrieval_policy_matches=context.get("retrieval_policy_matches"),
            retrieval_metadata=context.get("retrieval_metadata"),
            recommendation_context=context.get("recommendation_context"),
        )
        self._log_product_recommendation_trace(
            session_id=session_id,
            route_decision=route_decision,
            plan=plan,
            trace=response_payload["metadata"]["product_recommendation_trace"],
        )
        self._verify_card_text_consistency(
            reply=reply,
            tool_outputs=context["tool_outputs"],
            conversation_id=session_id,
        )
        # Skip memory persistence for blocked turns — blocked content
        # contaminates emotion/topic facts and poisons the next response.
        if stream_interrupted:
            logger.info(
                "client_disconnected_mid_stream",
                extra={"extra_fields": {"conversation_id": session_id}},
            )
            if reply and route_decision.route != "BLOCKED":
                self._persist_chat_turns(
                    context,
                    reply,
                    response_payload["metadata"],
                    partial=True,
                )
            return
        if route_decision.route != "BLOCKED":
            self._persist_lightweight_memory(
                session_id=session_id,
                message=context["message"],
                emotion_label=context["emotion"].label,
                birth_details=effective_birth_details,
                user_id=context.get("internal_user_id"),
            )
        self._persist_chat_turns(context, reply, response_payload["metadata"])

        # Fire-and-forget memory extraction every few turns
        session_id = context["session_id"]
        recent_count = len(context.get("recent_messages") or [])
        if recent_count >= 4 and recent_count % 2 == 0 and route_decision.route != "BLOCKED":
            asyncio.ensure_future(self._background_memory_extraction(
                session_id, user_id=context.get("internal_user_id"),
            ))

        yield (
            "done",
            {
                "resolved_session_id": session_id,
                "reply": reply,
                "intent": plan.action,
                "response": response_payload,
                "metadata": response_payload["metadata"],
            },
        )
