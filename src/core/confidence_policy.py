from __future__ import annotations

from src.core.planner import PlannerResult


UNKNOWN_ACTION_THRESHOLD = 0.0
CONFIDENCE_THRESHOLDS: dict[str, float] = {
    "respond_only": 0.50,
    "ask_clarification": 0.40,
    "show_kundali": 0.70,
    "suggest_kundali": 0.70,
    "recommend_product": 0.82,
    "suggest_product": 0.82,
    "matchmaking": 0.78,
    "suggest_matchmaking": 0.78,
    "suggest_consultant": 0.80,
    "book_pooja": 0.85,
    "suggest_booking": 0.85,
}


def get_threshold(action: str) -> float:
    return CONFIDENCE_THRESHOLDS.get(action, UNKNOWN_ACTION_THRESHOLD)


def is_confident(plan: PlannerResult) -> bool:
    return plan.confidence >= get_threshold(plan.action)
