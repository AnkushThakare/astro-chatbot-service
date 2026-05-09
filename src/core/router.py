from __future__ import annotations

from dataclasses import dataclass

from src.core.config import settings
from src.core.planner import PlannerResult


@dataclass
class ModelRoute:
    provider: str
    model: str
    reasoning_profile: str


def pick_model_route(plan: PlannerResult) -> ModelRoute:
    if plan.action in {"respond_only", "ask_clarification"}:
        return ModelRoute(provider="groq", model=settings.GROQ_MODEL, reasoning_profile="full-answer")
    if plan.action in {
        "show_kundali",
        "matchmaking",
        "book_pooja",
        "recommend_product",
        "suggest_consultant",
    } and plan.should_call_tool:
        return ModelRoute(provider="groq", model=settings.GROQ_MODEL, reasoning_profile="tool-aware")
    return ModelRoute(provider="groq", model=settings.GROQ_MODEL, reasoning_profile="fallback")
