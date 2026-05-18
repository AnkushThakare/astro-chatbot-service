from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EmotionResult:
    emotion: str
    intensity: str
    confidence: float

    @property
    def label(self) -> str:
        return self.emotion


def detect_emotion(message: str) -> EmotionResult:
    lowered = message.lower()

    if any(term in lowered for term in ("am i cursed", "black magic", "evil eye", "nazar", "scared", "fear")):
        return EmotionResult("fearful", "high", 0.93)
    if any(term in lowered for term in ("nothing is working", "stuck", "job", "career", "money", "finance")):
        if any(term in lowered for term in ("career", "job", "work", "money", "finance")):
            return EmotionResult("career_stress", "medium", 0.87)
        return EmotionResult("anxious", "medium", 0.79)
    if any(term in lowered for term in ("relationship", "marriage", "partner", "love life", "misunderstanding")):
        return EmotionResult("relationship_stress", "medium", 0.86)
    if any(term in lowered for term in ("health", "disease", "illness", "cancer", "doctor")):
        return EmotionResult("health_worry", "medium", 0.88)
    if any(term in lowered for term in ("mantra", "prayer", "pooja", "puja", "shiv", "hanuman", "devotional")):
        return EmotionResult("devotional", "low", 0.82)
    if any(term in lowered for term in ("confused", "unclear", "not sure", "what should i do")):
        return EmotionResult("confused", "medium", 0.76)
    if any(term in lowered for term in ("worried", "anxious", "panic", "heavy", "restless")):
        return EmotionResult("anxious", "medium", 0.81)
    return EmotionResult("calm", "low", 0.58)
