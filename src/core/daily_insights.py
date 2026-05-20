"""Daily Insight Generator — the morning message that builds the daily habit loop.

Combines chart + transits + predictions + energy flow + patterns into a
structured payload suitable for push notifications, in-app cards, and
the internal scheduler.
"""
from __future__ import annotations

from datetime import date, datetime
import re
from typing import Any

from src.astro.kundli import compute_full_chart
from src.astro.predictions import generate_predictive_insights
from src.astro.transits import compute_current_transits
from src.core.energy_flow import EnergyFlowSnapshot
from src.core.pattern_engine import analyze_personal_patterns

# ── Day-of-week planetary rulers (Vedic tradition) ──────────────
_DAY_PLANET: dict[int, str] = {
    0: "Moon",       # Monday
    1: "Mars",       # Tuesday
    2: "Mercury",    # Wednesday
    3: "Jupiter",    # Thursday
    4: "Venus",      # Friday
    5: "Saturn",     # Saturday
    6: "Sun",        # Sunday
}

_DEFAULT_ACTIONS: dict[str, str] = {
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
    return defaults.get(focus_area, defaults["general"])


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


def _override_action_from_energy(
    energy_snapshot: EnergyFlowSnapshot | None,
    default_action: str,
) -> str:
    """Override today's action if Energy Flow signals demand it."""
    if energy_snapshot is None:
        return default_action

    if energy_snapshot.stress_score >= 60:
        return (
            "Your energy pattern shows elevated stress. "
            "Before anything else, take 5 minutes of stillness and breathe deeply."
        )
    if energy_snapshot.behavioral_state == "overthinking_loop":
        return (
            "You've been circling a decision. Today, commit to one small step "
            "forward — even an imperfect one."
        )
    if energy_snapshot.behavioral_state == "drained_rhythm":
        return (
            "Late-night activity is affecting your rhythm. "
            "Prioritize rest and consider an earlier wind-down tonight."
        )
    return default_action


def _build_energy_summary(snapshot: EnergyFlowSnapshot | None) -> dict[str, Any] | None:
    """Build a compact energy flow summary for the API response."""
    if snapshot is None or snapshot.signal_count == 0:
        return None
    return {
        "overall_alignment": snapshot.overall_alignment,
        "stress_score": snapshot.stress_score,
        "emotional_state": snapshot.emotional_state,
        "behavioral_state": snapshot.behavioral_state,
        "focus_state": snapshot.focus_state,
    }


async def generate_daily_insight(
    *,
    birth_details: dict[str, Any],
    memory_context: str | None = None,
    preferred_language: str = "en",
    energy_snapshot: EnergyFlowSnapshot | None = None,
    user_name: str | None = None,
) -> dict[str, Any]:
    """Generate one personalized daily insight.

    Args:
        birth_details: User's birth details dict.
        memory_context: Stored memory facts string (optional).
        preferred_language: "en" or "hi" for language style.
        energy_snapshot: Latest behavioral energy flow snapshot (optional).
        user_name: User's first name for personalization (optional).

    Returns:
        Structured daily insight dict for API response / push notification.
    """
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
    # Let energy flow override the action if the user is stressed / stuck
    action = _override_action_from_energy(energy_snapshot, action)
    message = _compose_message(
        focus_area=focus_area,
        transit_data=transit_data,
        top_prediction=top_prediction,
        style=style,
    )

    now = datetime.now()
    ruling_planet = _DAY_PLANET[now.weekday()]

    return {
        "generated_for_date": date.today().isoformat(),
        "ruling_planet": ruling_planet,
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
        "energy_flow": _build_energy_summary(energy_snapshot),
        "topic_tags": [focus_area, "daily_check_in", "transits", "predictions"],
    }
