from __future__ import annotations

import re
from typing import Any

_PRIORITY_ORDER = {"urgent": 0, "high": 1, "medium": 2, "low": 3}
_TOPIC_KEYWORDS = {
    "career": ("career", "job", "work", "profession", "promotion", "office", "interview"),
    "relationship": ("relationship", "marriage", "partner", "love", "shaadi", "rishta"),
    "finance": ("money", "finance", "wealth", "income", "business", "savings"),
    "health": ("health", "stress", "sleep", "anxiety", "doctor", "wellbeing", "wellness"),
    "spirituality": ("spiritual", "mantra", "meditation", "puja", "pooja", "temple", "prayer"),
}
_INTERRUPT_ACTIONS = {
    "career": "Give one practical move for this week and frame it as steady progress, not instant resolution.",
    "relationship": "Give one softer communication step instead of a dramatic all-or-nothing conclusion.",
    "finance": "Give one conservative money step and reduce impulsive decision pressure.",
    "health": "Ground the user in routine, sleep, and calm rather than adding fear.",
    "spirituality": "Anchor the user in one simple devotional or reflective practice they can repeat.",
    "general": "Give one small stabilizing step and show the user what pattern is worth watching next.",
}


def _normalize_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip().lower()


def _parse_memory_facts(long_term_context: str | None) -> dict[str, str]:
    facts: dict[str, str] = {}
    for raw_line in (long_term_context or "").splitlines():
        line = raw_line.strip()
        if not line.startswith("- "):
            continue
        body = line[2:]
        if ":" not in body:
            continue
        key, value = body.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key and value:
            facts[key] = value
    return facts


def _topic_scores_from_text(text: str) -> dict[str, int]:
    scores = {topic: 0 for topic in _TOPIC_KEYWORDS}
    for topic, keywords in _TOPIC_KEYWORDS.items():
        for keyword in keywords:
            if re.search(rf"\b{re.escape(keyword)}\b", text):
                scores[topic] += 1
    return scores


def _best_prediction(predictions: list[dict[str, str]]) -> dict[str, str] | None:
    if not predictions:
        return None
    return sorted(
        predictions,
        key=lambda item: (
            _PRIORITY_ORDER.get(str(item.get("priority") or "medium"), 99),
            str(item.get("title") or ""),
        ),
    )[0]


def analyze_personal_patterns(
    *,
    long_term_context: str | None,
    recent_messages: list[dict[str, str]] | None,
    transit_data: dict[str, Any] | None,
    predictions: list[dict[str, str]] | None,
) -> dict[str, Any]:
    facts = _parse_memory_facts(long_term_context)
    scores = {topic: 0 for topic in _TOPIC_KEYWORDS}
    repeat_counts = {topic: 0 for topic in _TOPIC_KEYWORDS}

    for fact_key in ("last_concern", "last_topic"):
        value = _normalize_text(facts.get(fact_key))
        for topic, topic_score in _topic_scores_from_text(value).items():
            scores[topic] += topic_score * 3
            if topic_score > 0:
                repeat_counts[topic] += 1

    for message in recent_messages or []:
        if message.get("role") != "user":
            continue
        normalized = _normalize_text(message.get("content"))
        topic_scores = _topic_scores_from_text(normalized)
        for topic, topic_score in topic_scores.items():
            scores[topic] += topic_score
            if topic_score > 0:
                repeat_counts[topic] += 1

    best_topic = max(scores, key=scores.get) if scores else "career"
    dominant_theme = best_topic if scores.get(best_topic, 0) > 0 else "general"
    repeat_count = repeat_counts.get(dominant_theme, 0) if dominant_theme != "general" else 0

    top_prediction = _best_prediction(predictions or [])
    transit_summary = ((transit_data or {}).get("summary") or "").strip()
    current_trigger = ""
    if top_prediction and top_prediction.get("title"):
        current_trigger = str(top_prediction.get("title") or "").strip()
    elif transit_summary:
        current_trigger = transit_summary.splitlines()[0].strip()

    confidence = "low"
    if repeat_count >= 3 or scores.get(dominant_theme, 0) >= 6:
        confidence = "high"
    elif repeat_count >= 2 or scores.get(dominant_theme, 0) >= 3:
        confidence = "medium"

    if dominant_theme == "general":
        pattern_statement = "The user has not shown one clear repeating life loop yet."
    elif repeat_count >= 3:
        pattern_statement = (
            f"The user keeps circling back to {dominant_theme} pressure. "
            "Treat this as a repeating pattern, not a one-off question."
        )
    elif repeat_count == 2:
        pattern_statement = (
            f"There is an emerging {dominant_theme} loop here. "
            "Acknowledge that the same theme is showing up again."
        )
    else:
        pattern_statement = (
            f"The strongest live theme right now is {dominant_theme}. "
            "Use pattern language only lightly."
        )

    return {
        "dominant_theme": dominant_theme,
        "confidence": confidence,
        "repeat_count": repeat_count,
        "pattern_statement": pattern_statement,
        "current_trigger": current_trigger or "No single transit trigger stood out more than the overall atmosphere.",
        "interrupt_action": _INTERRUPT_ACTIONS[dominant_theme],
        "emotion_trend": facts.get("emotion_trend"),
        "preferred_style": facts.get("preferred_style"),
    }


def build_pattern_summary(
    *,
    long_term_context: str | None,
    recent_messages: list[dict[str, str]] | None,
    transit_data: dict[str, Any] | None,
    predictions: list[dict[str, str]] | None,
) -> str | None:
    analysis = analyze_personal_patterns(
        long_term_context=long_term_context,
        recent_messages=recent_messages,
        transit_data=transit_data,
        predictions=predictions,
    )
    if analysis["dominant_theme"] == "general" and analysis["confidence"] == "low":
        return None

    return "\n".join(
        [
            "Pattern mirror (this is where you should feel unusually perceptive, not generic):",
            f"- Recurring theme: {analysis['dominant_theme']}",
            f"- Pattern confidence: {analysis['confidence']}",
            f"- What keeps repeating: {analysis['pattern_statement']}",
            f"- Current trigger: {analysis['current_trigger']}",
            f"- Best interruption: {analysis['interrupt_action']}",
            (
                "- Pattern usage: If relevant, say one direct line like "
                "'I keep seeing the same pattern here' or "
                "'This is not only today's problem; this loop has been building.' "
                "Then give one grounded step. Do not sound spooky, manipulative, or fatalistic."
            ),
        ]
    )
