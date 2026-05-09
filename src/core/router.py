from __future__ import annotations

from dataclasses import dataclass

from src.core.config import settings
from src.core.intent import IntentResult


@dataclass
class ModelRoute:
    provider: str
    model: str
    reasoning_profile: str


def pick_model_route(intent: IntentResult) -> ModelRoute:
    if intent.name in {"show_kundali", "general_astrology"}:
        return ModelRoute(provider="groq", model=settings.GROQ_MODEL, reasoning_profile="full-answer")
    if intent.name in {"recommend_product", "suggest_consultant"}:
        return ModelRoute(provider="groq", model=settings.GROQ_MODEL, reasoning_profile="tool-aware")
    return ModelRoute(provider="groq", model=settings.GROQ_MODEL, reasoning_profile="fallback")
