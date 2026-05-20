from __future__ import annotations

import asyncio
import json
from functools import lru_cache
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from src.core.config import settings
from src.core.llm import GroqClient
from src.core.logging import get_logger

logger = get_logger(__name__)

_PLANNER_MSG_MAX_CHARS = 300


def _truncate_for_planner(content: str) -> str:
    if len(content) <= _PLANNER_MSG_MAX_CHARS:
        return content
    return content[:_PLANNER_MSG_MAX_CHARS].rstrip() + "..."


TOOL_ACTIONS = {
    "show_kundali",
    "matchmaking",
    "book_pooja",
    "recommend_product",
    "suggest_consultant",
    "confirm_booking",
    "schedule_consultation",
    "check_booking",
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
    "matchmaking",
    "milan",
}
KUNDALI_TOKENS = {"birth", "chart", "horoscope", "kundali", "kundli"}
SIDE_EFFECT_TOKENS = {"buy", "create", "ignore", "order", "pay", "payment", "purchase"}


PlannerAction = Literal[
    "respond_only",
    "show_kundali",
    "matchmaking",
    "book_pooja",
    "confirm_booking",
    "recommend_product",
    "suggest_consultant",
    "schedule_consultation",
    "check_booking",
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
    RETRY_BACKOFF_BASE_SECONDS = 3 / 4

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

        if "matchmaking" in tokens:
            return "matchmaking"
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
            # Don't trigger product action for general remedy questions
            # that happen to mention a product token alongside broad advice words
            general_remedy_words = {"remedy", "remedies", "help", "effects", "reduce", "problems"}
            if tokens & general_remedy_words and not (tokens & {"buy", "price", "wear", "order", "show"}):
                return None
            return "recommend_product"
        if tokens & KUNDALI_TOKENS and (
            tokens & {"analyse", "analyze", "read", "show", "stands", "tell"}
        ):
            return "show_kundali"
        return None

    @classmethod
    def explicit_tool_action(cls, message: str) -> PlannerAction | None:
        """Expose explicit action shortcuts for orchestration layers."""
        return cls._infer_explicit_action(message)

    @classmethod
    def _normalized_query_for_action(cls, action: PlannerAction, message: str) -> str | None:
        if action == "recommend_product":
            return cls._extract_product_query(message)
        if action == "suggest_consultant":
            return cls._extract_consultant_query(message)
        if action == "book_pooja":
            return cls._extract_booking_query(message)
        # confirm_booking and schedule_consultation don't need a search query —
        # they operate on IDs from the previous suggestion.
        return None

    @classmethod
    def _normalize_tool_arguments(
        cls,
        action: PlannerAction,
        message: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        normalized = dict(arguments)
        if action in SEARCH_QUERY_ACTIONS:
            query = normalized.get("search_query")
            cleaned_query = cls._normalize_search_query(query) if isinstance(query, str) else None
            fallback_query = cls._normalized_query_for_action(action, message)
            # Prefer the LLM-generated query — it has catalog awareness and
            # planetary mapping knowledge. Only fall back to keyword extraction
            # when the LLM didn't provide a query.
            final_query = cleaned_query or fallback_query
            if action == "suggest_consultant" and not cleaned_query and fallback_query:
                final_query = fallback_query
            if final_query:
                normalized["search_query"] = final_query
            else:
                normalized.pop("search_query", None)
        return normalized

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

        if normalized.action == "show_kundali" and has_birth_details and explicit_action == "show_kundali":
            normalized.should_call_tool = True
        if normalized.action == "matchmaking" and has_matchmaking_details and explicit_action == "matchmaking":
            normalized.should_call_tool = True

        if normalized.action in SEARCH_QUERY_ACTIONS:
            normalized.arguments = cls._normalize_tool_arguments(
                normalized.action,
                message,
                normalized.arguments,
            )
            final_query = normalized.arguments.get("search_query")
            if final_query:
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
        recent_messages: list[dict[str, str]] | None = None,
        session_state: str | None = None,
    ) -> list[dict[str, str]]:
        context = (
            f"has_birth_details={str(has_birth_details).lower()}\n"
            f"has_matchmaking_details={str(has_matchmaking_details).lower()}\n"
            f"is_authenticated={str(is_authenticated).lower()}"
        )
        if session_state:
            context += f"\n{session_state}"

        msgs: list[dict[str, str]] = [
            {"role": "system", "content": self.planner_prompt()},
        ]
        if recent_messages:
            for turn in recent_messages:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if role in ("user", "assistant") and content:
                    msgs.append({"role": role, "content": _truncate_for_planner(content)})
        msgs.append({
            "role": "user",
            "content": (
                "Plan the next action for this user message.\n"
                f"Context:\n{context}\n\n"
                f"User message:\n{message}"
            ),
        })
        return msgs

    async def _generate_raw_plan(
        self,
        *,
        message: str,
        has_birth_details: bool,
        has_matchmaking_details: bool,
        is_authenticated: bool,
        recent_messages: list[dict[str, str]] | None = None,
        session_state: str | None = None,
    ) -> str:
        messages = self._build_messages(
            message=message,
            has_birth_details=has_birth_details,
            has_matchmaking_details=has_matchmaking_details,
            is_authenticated=is_authenticated,
            recent_messages=recent_messages,
            session_state=session_state,
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
        recent_messages: list[dict[str, str]] | None = None,
        session_state: str | None = None,
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
                    recent_messages=recent_messages,
                    session_state=session_state,
                )
                # Strip markdown fences if LLM wraps JSON in ```json ... ```
                cleaned = raw.strip()
                if cleaned.startswith("```"):
                    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
                    cleaned = re.sub(r"\s*```$", "", cleaned)
                # Try parsing as JSON first, then as pydantic
                try:
                    parsed = json.loads(cleaned)
                    validated = PlannerResult.model_validate(parsed)
                except (json.JSONDecodeError, ValidationError):
                    validated = PlannerResult.model_validate_json(cleaned)
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
                    await asyncio.sleep(self.RETRY_BACKOFF_BASE_SECONDS * attempt)
                    continue
                break

        return self.fallback_result("Planner response was invalid, so the assistant fell back to respond_only.")


# ---------------------------------------------------------------------------
# Tool-call based planner (Groq function calling)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "show_kundali",
            "description": "Show, read, or analyze the user's birth chart (kundali). Call this when the user asks to see their horoscope, birth chart, kundali, or planetary positions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of why this action was chosen.",
                    },
                },
                "required": ["reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "matchmaking",
            "description": "Perform kundali matching or guna milan for marriage compatibility. Call this when the user asks about compatibility, guna matching, or kundali milan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of why this action was chosen.",
                    },
                },
                "required": ["reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_product",
            "description": "Recommend astrology products like rudraksha, mala, or bracelets. Call this when the user asks about products, wants to buy/wear rudraksha, asks for product recommendations, or mentions specific product names. Catalog carries: Rudraksha (1-14 Mukhi), Rudraksha Mala, Bracelets (energy/protection/planetary), Pendants. NO gemstones or yantras.",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_query": {
                        "type": "string",
                        "description": "Specific product search query. Be precise — include mukhi number when the concern maps to a planet (Sun→1, Moon→2, Mars→3, Mercury→4, Jupiter→5, Venus→6, Saturn→7, Rahu→8, Ketu→9). Examples: '7 mukhi rudraksha', '5 mukhi rudraksha mala', 'protection bracelet'.",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of why this action was chosen.",
                    },
                },
                "required": ["search_query", "reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "suggest_consultant",
            "description": "Suggest or connect the user with an astrologer or consultant. Call this when the user wants to talk to, consult, or find an astrologer or pandit. Best for: personalized chart analysis, complex questions, live readings, expert interpretation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_query": {
                        "type": "string",
                        "description": "Consultant specialty search query. Include the area of expertise. Examples: 'career astrologer', 'relationship consultant', 'kundali reading pandit', 'vaastu expert'.",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of why this action was chosen.",
                    },
                },
                "required": ["search_query", "reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_pooja",
            "description": "Book a puja, havan, homam, or religious ritual/service. Call this when the user wants to book or schedule a pooja or ritual. Best for: dosha removal, planetary shanti, life events (griha pravesh, marriage), urgent relief from difficult periods.",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_query": {
                        "type": "string",
                        "description": "Specific puja/ritual search query. Include the puja type or planetary focus. Examples: 'shani puja', 'navgraha shanti', 'mahamrityunjaya jaap', 'rudrabhishek', 'satyanarayan puja home'.",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of why this action was chosen.",
                    },
                },
                "required": ["search_query", "reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "confirm_booking",
            "description": (
                "Confirm and complete a puja or temple service booking that the user "
                "selected from the suggestions. Call this when the user says 'book it', "
                "'yes confirm', 'I want the first one', or otherwise agrees to book a "
                "specific service shown in the booking suggestions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "service_id": {
                        "type": "string",
                        "description": "The ID of the service the user wants to book (from suggestions).",
                    },
                    "service_type": {
                        "type": "string",
                        "enum": ["home_puja", "temple"],
                        "description": "Whether this is a home puja or temple service booking.",
                    },
                    "tier_id": {
                        "type": "string",
                        "description": "Optional tier/variant ID if the user picked a specific tier.",
                    },
                    "service_name": {
                        "type": "string",
                        "description": "Human-readable name of the service being booked.",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of why this action was chosen.",
                    },
                },
                "required": ["service_id", "service_type", "reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_consultation",
            "description": (
                "Book a 1:1 video call consultation with a pandit or astrologer. "
                "Call this when the user wants to book a call, schedule a consultation, "
                "or connect with a specific consultant shown in suggestions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "consultant_id": {
                        "type": "string",
                        "description": "The ID of the consultant/pandit to book with.",
                    },
                    "consultant_name": {
                        "type": "string",
                        "description": "Human-readable name of the consultant.",
                    },
                    "preferred_date": {
                        "type": "string",
                        "description": "Preferred date for the consultation (YYYY-MM-DD).",
                    },
                    "preferred_time": {
                        "type": "string",
                        "description": "Preferred time for the consultation (HH:MM).",
                    },
                    "concern": {
                        "type": "string",
                        "description": "Brief description of user's concern for the consultation.",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of why this action was chosen.",
                    },
                },
                "required": ["consultant_id", "reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_booking",
            "description": (
                "Show the user's booking history or check a booking status. "
                "Call this when the user asks about their bookings, order status, "
                "booking history, or wants to see past/upcoming puja bookings."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "status_filter": {
                        "type": "string",
                        "enum": ["pending", "confirmed", "completed", "cancelled"],
                        "description": "Optional filter for booking status.",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of why this action was chosen.",
                    },
                },
                "required": ["reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_clarification",
            "description": "Ask the user for missing information needed to fulfill their request. Call this when you cannot determine the user's intent or critical details are missing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "missing_information": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of missing fields, e.g. ['birth_details', 'search_query'].",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of what information is needed and why.",
                    },
                },
                "required": ["missing_information", "reasoning"],
            },
        },
    },
]

# Fixed confidence values for tool-call planner (all above policy thresholds)
_TOOL_CALL_CONFIDENCE: dict[str, float] = {
    "show_kundali": 0.92,
    "matchmaking": 0.92,
    "recommend_product": 0.90,
    "suggest_consultant": 0.90,
    "book_pooja": 0.90,
    "confirm_booking": 0.92,
    "schedule_consultation": 0.90,
    "check_booking": 0.90,
    "ask_clarification": 0.88,
    "respond_only": 0.85,
}


class ToolCallPlanner:
    def __init__(self, groq_client: GroqClient) -> None:
        self.groq_client = groq_client

    @staticmethod
    @lru_cache
    def system_prompt() -> str:
        prompt_path = settings.prompts_dir / "planner_tools.txt"
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8").strip()
        return (
            "You are a routing planner for a Vedic astrology assistant called Digveda. "
            "Based on the user's message, decide the best action by calling the appropriate tool. "
            "If the user is just chatting, asking general astrology questions, or greeting, "
            "do NOT call any tool — simply reply with a short acknowledgment. "
            "Only call a tool when there is a clear actionable intent."
        )

    def _build_messages(
        self,
        *,
        message: str,
        has_birth_details: bool,
        has_matchmaking_details: bool,
        recent_messages: list[dict[str, str]] | None = None,
        session_state: str | None = None,
    ) -> list[dict[str, str]]:
        context_lines = [
            f"has_birth_details={str(has_birth_details).lower()}",
            f"has_matchmaking_details={str(has_matchmaking_details).lower()}",
        ]
        if session_state:
            context_lines.append(session_state)

        msgs: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt()},
        ]
        if recent_messages:
            for turn in recent_messages:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if role in ("user", "assistant") and content:
                    msgs.append({"role": role, "content": _truncate_for_planner(content)})
        msgs.append({
            "role": "user",
            "content": (
                f"Context: {', '.join(context_lines)}\n\n"
                f"User message: {message}"
            ),
        })
        return msgs

    async def plan(
        self,
        *,
        message: str,
        has_birth_details: bool,
        has_matchmaking_details: bool,
        is_authenticated: bool,
        recent_messages: list[dict[str, str]] | None = None,
        session_state: str | None = None,
    ) -> PlannerResult:
        if not self.groq_client.is_configured:
            return ConversationPlanner.fallback_result(
                "Tool-call planner unavailable: LLM API key not configured."
            )

        messages = self._build_messages(
            message=message,
            has_birth_details=has_birth_details,
            has_matchmaking_details=has_matchmaking_details,
            recent_messages=recent_messages,
            session_state=session_state,
        )

        try:
            response_message = await self.groq_client.generate_with_tools(
                messages,
                tools=TOOL_DEFINITIONS,
                temperature=0.1,
                tool_choice="auto",
            )
        except Exception as exc:
            recovered = self._recover_from_failed_generation(exc, message, has_birth_details, has_matchmaking_details)
            if recovered is not None:
                return recovered
            logger.warning("ToolCallPlanner failed: %s | msg_count=%d", exc, len(messages))
            return ConversationPlanner.fallback_result(
                f"Tool-call planner error: {exc}"
            )

        tool_calls = response_message.get("tool_calls")
        if not tool_calls:
            return PlannerResult(
                action="respond_only",
                confidence=_TOOL_CALL_CONFIDENCE["respond_only"],
                arguments={},
                missing_information=[],
                should_call_tool=False,
                reasoning=response_message.get("content", "No tool needed") or "General conversation",
            )

        call = tool_calls[0]
        fn_name = call.get("function", {}).get("name", "respond_only")
        try:
            fn_args = json.loads(call.get("function", {}).get("arguments", "{}"))
        except json.JSONDecodeError:
            fn_args = {}

        reasoning = fn_args.pop("reasoning", fn_name)

        if fn_name not in _TOOL_CALL_CONFIDENCE:
            return PlannerResult(
                action="respond_only",
                confidence=_TOOL_CALL_CONFIDENCE["respond_only"],
                arguments={},
                should_call_tool=False,
                reasoning=f"Unknown tool '{fn_name}', falling back to respond_only.",
            )

        action: PlannerAction = fn_name  # type: ignore[assignment]
        confidence = _TOOL_CALL_CONFIDENCE[action]
        should_call_tool = action in TOOL_ACTIONS
        missing_info = fn_args.get("missing_information", [])
        arguments = {k: v for k, v in fn_args.items() if k != "missing_information"}

        result = PlannerResult(
            action=action,
            confidence=confidence,
            arguments=arguments,
            missing_information=missing_info if isinstance(missing_info, list) else [],
            should_call_tool=should_call_tool,
            reasoning=reasoning,
        )

        return ConversationPlanner._normalize_plan(
            result,
            message=message,
            has_birth_details=has_birth_details,
            has_matchmaking_details=has_matchmaking_details,
        )

    @staticmethod
    def _recover_from_failed_generation(
        exc: Exception,
        message: str,
        has_birth_details: bool,
        has_matchmaking_details: bool,
    ) -> PlannerResult | None:
        if not hasattr(exc, "response"):
            return None
        try:
            body = exc.response.json()
        except Exception:
            return None
        error = body.get("error") or {}
        if error.get("code") != "tool_use_failed":
            return None
        failed = error.get("failed_generation", "")
        match = re.search(r"<function=(\w+)\s*(\{.*?\})", failed, re.DOTALL)
        if not match:
            return None
        fn_name = match.group(1)
        try:
            fn_args = json.loads(match.group(2))
        except json.JSONDecodeError:
            return None
        if fn_name not in _TOOL_CALL_CONFIDENCE:
            return None
        logger.info("Recovered tool call from failed_generation: %s", fn_name)
        reasoning = fn_args.pop("reasoning", fn_name)
        action: PlannerAction = fn_name  # type: ignore[assignment]
        confidence = _TOOL_CALL_CONFIDENCE[action]
        should_call_tool = action in TOOL_ACTIONS
        missing_info = fn_args.get("missing_information", [])
        arguments = {k: v for k, v in fn_args.items() if k != "missing_information"}
        result = PlannerResult(
            action=action,
            confidence=confidence,
            arguments=arguments,
            missing_information=missing_info if isinstance(missing_info, list) else [],
            should_call_tool=should_call_tool,
            reasoning=reasoning,
        )
        return ConversationPlanner._normalize_plan(
            result,
            message=message,
            has_birth_details=has_birth_details,
            has_matchmaking_details=has_matchmaking_details,
        )
