from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from src.core.config import settings
from src.core.guardrails import GuardrailDecision
from src.core.planner import PlannerResult


FAST_CHAT_KEYWORDS = {
    "astrology",
    "career",
    "chart",
    "confused",
    "dasha",
    "dosh",
    "future",
    "job",
    "mahadasha",
    "mantra",
    "mangal",
    "marriage",
    "moon",
    "peace",
    "planet",
    "rahu",
    "remedy",
    "saturn",
    "shani",
    "spiritual",
    "stuck",
}
KUNDALI_KEYWORDS = {"birth", "chart", "horoscope", "kundali", "kundli"}
MATCHMAKING_KEYWORDS = {
    "compatibility",
    "guna",
    "kundali",
    "kundli",
    "match",
    "matching",
    "matchmaking",
    "milan",
}
PRODUCT_KEYWORDS = {"bracelet", "mala", "mukhi", "product", "rudraksha"}
CONSULTANT_KEYWORDS = {"astrologer", "consultant", "jyotish", "pandit", "panditji"}
BOOKING_KEYWORDS = {"book", "booking", "havan", "homam", "pooja", "puja", "service", "temple"}


@dataclass
class ModelRoute:
    provider: str
    model: str
    reasoning_profile: str


@dataclass
class ChatRouteDecision:
    route: str
    intent: str
    confidence: float
    risk_level: str
    reason: str
    missing_slots: list[str] = field(default_factory=list)
    should_call_tool: bool = False
    needs_planner: bool = False
    normalized_args: dict[str, Any] = field(default_factory=dict)


def _tokens(message: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", message.lower()))


def _query_from_message(message: str, blocked_words: set[str]) -> str | None:
    tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", message.lower())
        if token not in blocked_words
    ]
    if not tokens:
        return None
    return " ".join(tokens[:6])


def classify_route(
    *,
    message: str,
    birth_details: dict[str, Any] | None,
    matchmaking_details: dict[str, Any] | None,
    pre_guardrail: GuardrailDecision,
) -> ChatRouteDecision:
    if not pre_guardrail.allowed:
        return ChatRouteDecision(
            route="BLOCKED",
            intent="respond_only",
            confidence=0.99,
            risk_level=pre_guardrail.risk_level,
            reason=pre_guardrail.reason,
            should_call_tool=False,
        )

    tokens = _tokens(message)

    matchmaking_strong = {"compatibility", "guna", "match", "milan", "matching", "matchmaking"}
    if tokens & MATCHMAKING_KEYWORDS and tokens & matchmaking_strong:
        if matchmaking_details is None:
            return ChatRouteDecision(
                route="CLARIFICATION",
                intent="matchmaking",
                confidence=0.94,
                risk_level="low",
                reason="missing_matchmaking_details",
                missing_slots=["matchmaking_details"],
                should_call_tool=False,
            )
        return ChatRouteDecision(
            route="TOOL_FLOW",
            intent="matchmaking",
            confidence=0.96,
            risk_level="low",
            reason="explicit_matchmaking_request",
            should_call_tool=True,
        )

    if tokens & KUNDALI_KEYWORDS and (tokens & {"show", "read", "check", "my", "meri", "mera", "dikhao", "dekhna", "dekho", "batao", "banaao"}):
        if birth_details is None:
            return ChatRouteDecision(
                route="CLARIFICATION",
                intent="show_kundali",
                confidence=0.92,
                risk_level="low",
                reason="missing_birth_details",
                missing_slots=["birth_details"],
                should_call_tool=False,
            )
        return ChatRouteDecision(
            route="TOOL_FLOW",
            intent="show_kundali",
            confidence=0.96,
            risk_level="low",
            reason="explicit_kundali_request",
            should_call_tool=True,
        )

    if tokens & PRODUCT_KEYWORDS:
        search_query = _query_from_message(
            message,
            {"a", "an", "and", "for", "i", "me", "my", "please", "recommend", "suggest", "wear", "buy", "want", "get", "purchase", "show", "chahiye", "kharidna", "to", "tell", "about", "ke", "liye", "kya", "hai", "hota", "the", "is", "what", "some", "any", "something", "koi", "kuch"},
        )
        if search_query is None:
            return ChatRouteDecision(
                route="CLARIFICATION",
                intent="recommend_product",
                confidence=0.82,
                risk_level="low",
                reason="missing_product_query",
                missing_slots=["search_query"],
            )
        return ChatRouteDecision(
            route="TOOL_FLOW",
            intent="recommend_product",
            confidence=0.9,
            risk_level="low",
            reason="explicit_product_request",
            should_call_tool=True,
            normalized_args={"search_query": search_query},
        )

    if tokens & CONSULTANT_KEYWORDS and ("connect" in tokens or "consult" in tokens or "suggest" in tokens or "talk" in tokens or "want" in tokens):
        search_query = _query_from_message(
            message,
            {"a", "an", "and", "astrologer", "consultant", "find", "i", "me", "pandit", "please", "suggest", "to", "want"},
        )
        search_query = search_query or "general astrologer"
        return ChatRouteDecision(
            route="TOOL_FLOW",
            intent="suggest_consultant",
            confidence=0.9,
            risk_level="low",
            reason="explicit_consultant_request",
            should_call_tool=True,
            normalized_args={"search_query": search_query},
        )

    if tokens & BOOKING_KEYWORDS and ("book" in tokens or "booking" in tokens or "want" in tokens):
        search_query = _query_from_message(
            message,
            {"a", "an", "and", "at", "book", "booking", "for", "i", "me", "my", "please", "to", "want"},
        )
        if search_query is None:
            return ChatRouteDecision(
                route="CLARIFICATION",
                intent="book_pooja",
                confidence=0.82,
                risk_level="low",
                reason="missing_booking_query",
                missing_slots=["search_query"],
            )
        return ChatRouteDecision(
            route="TOOL_FLOW",
            intent="book_pooja",
            confidence=0.91,
            risk_level="low",
            reason="explicit_booking_request",
            should_call_tool=True,
            normalized_args={"search_query": search_query},
        )

    if tokens & FAST_CHAT_KEYWORDS:
        return ChatRouteDecision(
            route="FAST_CHAT",
            intent="respond_only",
            confidence=0.86,
            risk_level="low",
            reason="astrology_qa",
            should_call_tool=False,
            needs_planner=True,
        )

    return ChatRouteDecision(
        route="FAST_CHAT",
        intent="respond_only",
        confidence=0.64,
        risk_level=pre_guardrail.risk_level,
        reason="default_fast_chat",
        should_call_tool=False,
        needs_planner=True,
    )


def pick_model_route(plan: PlannerResult) -> ModelRoute:
    provider = settings.RESPONSE_LLM_PROVIDER
    model = settings.RESPONSE_LLM_MODEL or settings.GROQ_MODEL
    if plan.action in {"respond_only", "ask_clarification"}:
        return ModelRoute(provider=provider, model=model, reasoning_profile="fast-answer")
    if plan.action in {
        "show_kundali",
        "matchmaking",
        "book_pooja",
        "recommend_product",
        "suggest_consultant",
    } and plan.should_call_tool:
        return ModelRoute(provider=provider, model=model, reasoning_profile="tool-aware")
    return ModelRoute(provider=provider, model=model, reasoning_profile="fallback")
