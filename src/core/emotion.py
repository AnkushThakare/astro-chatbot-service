from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EmotionResult:
    label: str
    confidence: float


def detect_emotion(message: str) -> EmotionResult:
    lowered = message.lower()
    if any(term in lowered for term in ("worried", "anxious", "panic", "scared", "fear")):
        return EmotionResult("anxious", 0.84)
    if any(term in lowered for term in ("sad", "depressed", "heartbroken", "hopeless")):
        return EmotionResult("sad", 0.82)
    if any(term in lowered for term in ("angry", "frustrated", "upset")):
        return EmotionResult("distressed", 0.75)
    return EmotionResult("calm", 0.58)
