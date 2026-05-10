from __future__ import annotations

import asyncio
from functools import lru_cache
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from src.core.config import settings
from src.core.llm import GroqClient
from src.core.logging import get_logger

logger = get_logger(__name__)

TOOL_ACTIONS = {
    "show_kundali",
    "matchmaking",
    "book_pooja",
    "recommend_product",
    "suggest_consultant",
}
SEARCH_QUERY_ACTIONS = {"book_pooja", "recommend_product", "suggest_consultant"}
LANGUAGE_TOKENS = {
    "hindi",
    "english",
    "tamil",
    "telugu",
    "kannada",
    "malayalam",
    "marathi",
    "gujarati",
    "bengali",
    "punjabi",
}
STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "book",
    "booking",
    "can",
    "for",
    "help",
    "i",
    "in",
    "me",
    "my",
    "next",
    "please",
    "show",
    "suggest",
    "tell",
    "the",
    "to",
    "want",
    "with",
}
REQUEST_VERBS = {
    "book",
    "buy",
    "connect",
    "consult",
    "find",
    "get",
    "need",
    "price",
    "recommend",
    "show",
    "suggest",
    "talk",
    "wear",
    "want",
    "chahiye",
}
PRODUCT_TOKENS = {
    "bracelet",
    "bracelets",
    "gemstone",
    "gemstones",
    "mala",
    "mukhi",
    "rudraksh",
    "rudraksha",
    "yantra",
}
CONSULTANT_TOKENS = {"astrologer", "astroger", "consultant", "jyotish", "pandit", "panditji"}
BOOKING_TOKENS = {"havan", "homam", "puja", "pooja", "ritual", "service", "temple"}
MATCHMAKING_TOKENS = {
    "compatibility",
    "guna",
    "gund",
    "guna-milan",
    "kundali",
    "kundli",
    "match",
    "matching",
    "milan",
}
KUNDALI_TOKENS = {"birth", "chart", "horoscope", "kundali", "kundli"}
SIDE_EFFECT_TOKENS = {"buy", "create", "ignore", "order", "pay", "payment", "purchase"}


PlannerAction = Literal[
    "respond_only",
    "show_kundali",
    "matchmaking",
    "book_pooja",
    "recommend_product",
    "suggest_consultant",
    "ask_clarification",
]


class PlannerResult(BaseModel):
    action: PlannerAction
    confidence: float = Field(ge=0.0, le=1.0)
    arguments: dict[str, Any] = Field(default_factory=dict)
    missing_information: list[str] = Field(default_factory=list)
    should_call_tool: bool = False
    reasoning: str = Field(min_length=1)


class ConversationPlanner:
    MAX_RETRIES = 3

    def __init__(self, groq_client: GroqClient, planner_model: str | None = None) -> None:
        self.groq_client = groq_client
        self.planner_model = planner_model or settings.GROQ_PLANNER_MODEL

    @staticmethod
    @lru_cache
    def planner_prompt() -> str:
        prompt_path = settings.prompts_dir / "planner.txt"
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8").strip()
        return (
            "You are a routing planner for an astrology assistant. "
            "Return JSON only. Choose one action from: "
            "respond_only, show_kundali, matchmaking, book_pooja, "
            "recommend_product, suggest_consultant, ask_clarification. "
            "Never create bookings or execute side effects from chat. "
            "Only suggest read-only tools. "
            "For booking, product, or consultant suggestions, include a concise search_query "
            "inside arguments when a tool should run. "
            "If required information is missing, put the missing field names into "
            "missing_information and set should_call_tool to false. "
            "Return an object with keys: action, confidence, arguments, "
            "missing_information, should_call_tool, reasoning."
        )

    @staticmethod
    def fallback_result(reason: str) -> PlannerResult:
        return PlannerResult(
            action="respond_only",
            confidence=0.0,
            arguments={},
            missing_information=[],
            should_call_tool=False,
            reasoning=reason,
        )

    @staticmethod
    def _clean_tokens(text: str) -> list[str]:
        normalized = text.lower()
        replacements = (
            ("astroger", "astrologer"),
            ("pooja", "puja"),
            ("rudraksh", "rudraksha"),
        )
        for source, target in replacements:
            normalized = re.sub(rf"\b{re.escape(source)}\b", target, normalized)
        return re.findall(r"[a-z0-9]+", normalized)

    @classmethod
    def _normalize_search_query(cls, query: str) -> str | None:
        tokens = [token for token in cls._clean_tokens(query) if token not in STOPWORDS]
        if not tokens:
            return None
        return " ".join(tokens[:6])

    @classmethod
    def _extract_consultant_query(cls, message: str) -> str | None:
        tokens = cls._clean_tokens(message)
        selected: list[str] = []

        if any(token in {"love", "marriage", "relationship", "compatibility"} for token in tokens):
            selected.extend(["relationship", "astrologer"])
        elif any(token in {"career", "job", "work"} for token in tokens):
            selected.extend(["career", "astrologer"])
        elif any(token in {"health", "stress", "anxiety"} for token in tokens):
            selected.extend(["healing", "astrologer"])
        elif any(token in {"finance", "money", "business"} for token in tokens):
            selected.extend(["finance", "astrologer"])
        elif any(token in {"kundali", "kundli", "chart"} for token in tokens):
            selected.extend(["kundali", "astrologer"])
        else:
            specialty_terms = {
                "career",
                "education",
                "family",
                "finance",
                "health",
                "kundali",
                "remedies",
                "ritual",
            }
            for token in tokens:
                if token in specialty_terms or token in LANGUAGE_TOKENS:
                    selected.append(token)
            if selected or any(token in CONSULTANT_TOKENS for token in tokens):
                selected.append("astrologer")

        deduped: list[str] = []
        for token in selected:
            if token not in deduped:
                deduped.append(token)
        return " ".join(deduped[:6]) or None

    @classmethod
    def _extract_booking_query(cls, message: str) -> str | None:
        tokens = cls._clean_tokens(message)
        selected = [
            token
            for token in tokens
            if token not in STOPWORDS and token not in REQUEST_VERBS and token != "week"
        ]
        if not selected:
            return None
        if "puja" not in selected and any(token in BOOKING_TOKENS for token in tokens):
            selected.append("puja")
        return " ".join(selected[:6]) or None

    @classmethod
    def _extract_product_query(cls, message: str) -> str | None:
        tokens = cls._clean_tokens(message)
        selected: list[str] = []

        if "rudraksha" in tokens:
            if "mala" in tokens:
                selected.extend(["rudraksha", "mala"])
            else:
                selected.append("rudraksha")
        if "bracelet" in tokens or "bracelets" in tokens:
            if "protection" in tokens:
                selected.extend(["protection", "bracelet"])
            elif "energy" in tokens:
                selected.extend(["energy", "bracelet"])
            else:
                selected.append("bracelet")

        descriptive_terms = {
            "5",
            "6",
            "7",
            "career",
            "delay",
            "growth",
            "marriage",
            "meditation",
            "mukhi",
            "positivity",
            "price",
            "protection",
            "saturn",
        }
        for token in tokens:
            if token in descriptive_terms and token not in selected:
                selected.append(token)

        if not selected and any(token in {"gemstone", "yantra"} for token in tokens):
            fallback_terms = [
                token
                for token in tokens
                if token in {"career", "delay", "growth", "marriage", "protection", "saturn"}
            ]
            selected = fallback_terms[:3] + ["remedy"]

        deduped: list[str] = []
        for token in selected:
            if token not in deduped:
                deduped.append(token)
        return " ".join(deduped[:6]) or None

    @classmethod
    def _infer_explicit_action(cls, message: str) -> PlannerAction | None:
        tokens = set(cls._clean_tokens(message))

        if tokens & MATCHMAKING_TOKENS and ("milan" in tokens or "match" in tokens or "matching" in tokens):
            return "matchmaking"
        if tokens & CONSULTANT_TOKENS and (
            tokens & {"call", "chat", "connect", "consult", "find", "talk", "want"}
        ):
            return "suggest_consultant"
        if tokens & BOOKING_TOKENS and (tokens & {"book", "booking", "home", "temple", "want"}):
            return "book_pooja"
        if tokens & PRODUCT_TOKENS and (
            tokens & REQUEST_VERBS or "price" in tokens or len(tokens) <= 5
        ):
            return "recommend_product"
        if tokens & KUNDALI_TOKENS and (
            tokens & {"analyse", "analyze", "read", "show", "stands", "tell"}
        ):
            return "show_kundali"
        return None

    @classmethod
    def _normalized_query_for_action(cls, action: PlannerAction, message: str) -> str | None:
        if action == "recommend_product":
            return cls._extract_product_query(message)
        if action == "suggest_consultant":
            return cls._extract_consultant_query(message)
        if action == "book_pooja":
            return cls._extract_booking_query(message)
        return None

    @classmethod
    def _normalize_plan(
        cls,
        result: PlannerResult,
        *,
        message: str,
        has_birth_details: bool,
        has_matchmaking_details: bool,
    ) -> PlannerResult:
        normalized = result.model_copy(deep=True)

        explicit_action = cls._infer_explicit_action(message)
        if normalized.action in {"respond_only", "ask_clarification"} and explicit_action in TOOL_ACTIONS:
            normalized.action = explicit_action
            normalized.reasoning = (
                f"{normalized.reasoning} Backend fallback matched an explicit user request."
            )
            normalized.confidence = max(normalized.confidence, 0.78)

        if normalized.action == "show_kundali" and not has_birth_details:
            normalized.action = "ask_clarification"
            normalized.should_call_tool = False
            normalized.missing_information = ["birth_details"]
            normalized.arguments = {}
            normalized.reasoning = "Birth details are required before a kundali tool call can run."
            return normalized

        if normalized.action == "matchmaking" and not has_matchmaking_details:
            normalized.action = "ask_clarification"
            normalized.should_call_tool = False
            normalized.missing_information = ["matchmaking_details"]
            normalized.arguments = {}
            normalized.reasoning = "Matchmaking details are required before a matchmaking tool call can run."
            return normalized

        if normalized.action in SEARCH_QUERY_ACTIONS:
            query = normalized.arguments.get("search_query")
            query = query if isinstance(query, str) else ""
            cleaned_query = cls._normalize_search_query(query) if query else None
            fallback_query = cls._normalized_query_for_action(normalized.action, message)
            final_query = cleaned_query or fallback_query
            if normalized.action == "suggest_consultant" and fallback_query:
                final_query = fallback_query
            elif cleaned_query and fallback_query and len(cleaned_query.split()) > len(fallback_query.split()):
                final_query = fallback_query

            if final_query:
                normalized.arguments["search_query"] = final_query
                if explicit_action == normalized.action:
                    normalized.should_call_tool = True
            else:
                normalized.should_call_tool = False
                normalized.missing_information = ["search_query"]

        if normalized.action == "respond_only" and set(cls._clean_tokens(message)) & SIDE_EFFECT_TOKENS:
            normalized.should_call_tool = False

        if normalized.action not in TOOL_ACTIONS:
            normalized.should_call_tool = False

        return normalized

    def _build_messages(
        self,
        *,
        message: str,
        has_birth_details: bool,
        has_matchmaking_details: bool,
        is_authenticated: bool,
    ) -> list[dict[str, str]]:
        context = (
            f"has_birth_details={str(has_birth_details).lower()}\n"
            f"has_matchmaking_details={str(has_matchmaking_details).lower()}\n"
            f"is_authenticated={str(is_authenticated).lower()}"
        )
        return [
            {"role": "system", "content": self.planner_prompt()},
            {
                "role": "user",
                "content": (
                    "Plan the next action for this user message.\n"
                    f"Context:\n{context}\n\n"
                    f"User message:\n{message}"
                ),
            },
        ]

    async def _generate_raw_plan(
        self,
        *,
        message: str,
        has_birth_details: bool,
        has_matchmaking_details: bool,
        is_authenticated: bool,
    ) -> str:
        messages = self._build_messages(
            message=message,
            has_birth_details=has_birth_details,
            has_matchmaking_details=has_matchmaking_details,
            is_authenticated=is_authenticated,
        )
        return await self.groq_client.generate(
            messages,
            model=self.planner_model,
            temperature=0.1,
            response_format={"type": "json_object"},
        )

    async def plan(
        self,
        *,
        message: str,
        has_birth_details: bool,
        has_matchmaking_details: bool,
        is_authenticated: bool,
    ) -> PlannerResult:
        if not self.groq_client.is_configured:
            return self.fallback_result("Planner model is unavailable because GROQ_API_KEY is not configured.")

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                raw = await self._generate_raw_plan(
                    message=message,
                    has_birth_details=has_birth_details,
                    has_matchmaking_details=has_matchmaking_details,
                    is_authenticated=is_authenticated,
                )
                validated = PlannerResult.model_validate_json(raw)
                return self._normalize_plan(
                    validated,
                    message=message,
                    has_birth_details=has_birth_details,
                    has_matchmaking_details=has_matchmaking_details,
                )
            except ValidationError as exc:
                logger.warning("Planner schema validation failed: %s", exc)
                break
            except Exception as exc:
                logger.warning("Planner execution failed: %s", exc)
                if "429" in str(exc) and attempt < self.MAX_RETRIES:
                    await asyncio.sleep(0.75 * attempt)
                    continue
                break

        return self.fallback_result("Planner response was invalid, so the assistant fell back to respond_only.")
