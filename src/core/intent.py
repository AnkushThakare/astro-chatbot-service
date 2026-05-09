from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from src.core.config import settings


@dataclass
class IntentResult:
    name: str
    confidence: float
    reason: str


class IntentClassifier:
    @staticmethod
    @lru_cache
    def classifier_prompt() -> str:
        prompt_path = settings.prompts_dir / "intent_classifier.txt"
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8").strip()
        return "Classify the user message into one of: show_kundali, recommend_product, suggest_consultant, general_astrology."

    def classify(self, message: str) -> IntentResult:
        text = message.lower()
        if any(term in text for term in ("kundali", "birth chart", "chart", "horoscope")):
            return IntentResult("show_kundali", 0.93, "Detected kundali or chart vocabulary.")
        if any(term in text for term in ("product", "gemstone", "rudraksha", "remedy", "buy")):
            return IntentResult("recommend_product", 0.88, "Detected commerce or remedy vocabulary.")
        if any(term in text for term in ("consultant", "astrologer", "expert", "talk to", "consult")):
            return IntentResult("suggest_consultant", 0.90, "Detected expert or consultation vocabulary.")
        return IntentResult("general_astrology", 0.65, "Defaulted to general astrology intent.")
