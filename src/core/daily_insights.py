from __future__ import annotations

from datetime import date
import re
from typing import Any

from src.astro.kundli import compute_full_chart
from src.astro.predictions import generate_predictive_insights
from src.astro.transits import compute_current_transits
from src.core.pattern_engine import analyze_personal_patterns
_DEFAULT_ACTIONS = {
    "career": "Stay disciplined with one important work task today instead of scattering your effort.",
    "relationship": "Keep one conversation softer and more patient than usual today.",
    "finance": "Avoid impulsive spending today and review one practical money decision calmly.",
    "health": "Protect your routine today: food, rest, and mental steadiness matter more than pushing hard.",
    "spirituality": "Take 5 quiet minutes for mantra, prayer, or stillness before the day speeds up.",
    "general": "Keep the day simple and intentional. One steady action will help more than overthinking.",
}


def _strip_emoji_prefix(value: str | None) -> str:
    cleaned = re.sub(r"^[^\w]+", "", (value or "").strip())
    return cleaned or "Daily insight"


def _compose_headline(focus_area: str, top_prediction: dict[str, str] | None) -> str:
    if top_prediction and top_prediction.get("title"):
        title = _strip_emoji_prefix(top_prediction["title"])
        if focus_area == "general":
            return title
        return f"{focus_area.title()} focus: {title}"

    defaults = {
        "career": "Career timing deserves disciplined focus today",
        "relationship": "Relationship energy asks for patience today",
        "finance": "Money decisions need steadier judgment today",
        "health": "Your routine needs extra care today",
        "spirituality": "A quieter spiritual rhythm will help today",
        "general": "Your chart points to a steady, intentional day",
    }
    return defaults[focus_area]


def _compose_message(
    *,
    focus_area: str,
    transit_data: dict[str, Any] | None,
    top_prediction: dict[str, str] | None,
    style: str,
) -> str:
    transit_summary = ((transit_data or {}).get("summary") or "").strip()
    prediction_text = (top_prediction or {}).get("insight", "").strip()

    if style == "hinglish":
        parts: list[str] = []
        if prediction_text:
            parts.append(prediction_text)
        elif transit_summary:
            parts.append(transit_summary)
        else:
            parts.append("Aaj ka energy pattern steady effort aur clarity ko support karta hai.")
        parts.append(_DEFAULT_ACTIONS[focus_area])
        return " ".join(part.strip() for part in parts if part.strip())

    parts = []
    if prediction_text:
        parts.append(prediction_text)
    elif transit_summary:
        parts.append(transit_summary)
    else:
        parts.append("Today's chart pattern favors steadier choices over reactive ones.")
    parts.append(_DEFAULT_ACTIONS[focus_area])
    return " ".join(part.strip() for part in parts if part.strip())


def _compose_push_text(headline: str, action: str) -> str:
    sentence = f"{headline}. {action}"
    compact = re.sub(r"\s+", " ", sentence).strip()
    if len(compact) <= 160:
        return compact
    return compact[:157].rstrip() + "..."


async def generate_daily_insight(
    *,
    birth_details: dict[str, Any],
    memory_context: str | None = None,
    preferred_language: str = "en",
) -> dict[str, Any]:
    chart = await compute_full_chart(birth_details)
    transit_data = await compute_current_transits(chart, birth_details)
    predictions = generate_predictive_insights(chart, transit_data)
    pattern_analysis = analyze_personal_patterns(
        long_term_context=memory_context,
        recent_messages=None,
        transit_data=transit_data,
        predictions=predictions,
    )

    focus_area = str(pattern_analysis.get("dominant_theme") or "general")
    if focus_area not in _DEFAULT_ACTIONS:
        focus_area = "general"
    top_prediction = {
        "title": pattern_analysis.get("current_trigger") or "",
        "insight": pattern_analysis.get("pattern_statement") or "",
        "actionable": pattern_analysis.get("interrupt_action") or "",
    }
    style = "hinglish" if preferred_language.lower().startswith("hi") else "english"
    headline = _compose_headline(focus_area, top_prediction)
    action = (top_prediction or {}).get("actionable", "").strip() or _DEFAULT_ACTIONS[focus_area]
    message = _compose_message(
        focus_area=focus_area,
        transit_data=transit_data,
        top_prediction=top_prediction,
        style=style,
    )

    return {
        "generated_for_date": date.today().isoformat(),
        "focus_area": focus_area,
        "headline": headline,
        "message": message,
        "action": action,
        "push_text": _compose_push_text(headline, action),
        "transit_summary": (transit_data or {}).get("summary"),
        "top_prediction": top_prediction,
        "pattern_narrative": pattern_analysis.get("pattern_statement"),
        "pattern_confidence": pattern_analysis.get("confidence"),
        "prediction_count": len(predictions),
        "memory_context_used": bool(memory_context and memory_context.strip()),
        "topic_tags": [focus_area, "daily_check_in", "transits", "predictions"],
    }
