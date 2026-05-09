from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from src.core.config import settings
from src.core.llm import GroqClient
from src.core.logging import get_logger

logger = get_logger(__name__)


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

        try:
            raw = await self._generate_raw_plan(
                message=message,
                has_birth_details=has_birth_details,
                has_matchmaking_details=has_matchmaking_details,
                is_authenticated=is_authenticated,
            )
            return PlannerResult.model_validate_json(raw)
        except ValidationError as exc:
            logger.warning("Planner schema validation failed: %s", exc)
        except Exception as exc:
            logger.warning("Planner execution failed: %s", exc)

        return self.fallback_result("Planner response was invalid, so the assistant fell back to respond_only.")
