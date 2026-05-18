from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any


def generate_predictive_insights(
    chart: dict[str, Any],
    transit_data: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Generate proactive predictions a real pandit would give.

    Combines dasha transitions, yogas, transits, and planetary positions
    to produce actionable insights the user didn't ask for — but needs to hear.
    """
    insights: list[dict[str, str]] = []
    now = datetime.now()

    dasha = chart.get("dasha") or {}

    # 1. Upcoming dasha transition warnings
    insights.extend(_dasha_transition_insights(dasha, now))

    # 2. Current dasha period interpretation
    insights.extend(_current_dasha_insights(dasha))

    # 3. Yoga-based life insights
    insights.extend(_yoga_insights(chart.get("yogas") or []))

    # 4. Transit-based insights (if available)
    if transit_data and transit_data.get("available"):
        insights.extend(_transit_insights(transit_data))

    # 5. Planetary strength observations
    insights.extend(_planetary_strength_insights(chart))

    return insights[:6]  # Cap at 6 most important


def _dasha_transition_insights(dasha: dict[str, Any], now: datetime) -> list[dict[str, str]]:
    """Warn about upcoming dasha changes — this is what a pandit does proactively."""
    insights: list[dict[str, str]] = []

    # Check antardasha end date
    antara_end_str = dasha.get("antardasha_end")
    if antara_end_str:
        try:
            antara_end = datetime.strptime(antara_end_str, "%B %Y")
            months_left = (antara_end.year - now.year) * 12 + (antara_end.month - now.month)
            antara_planet = dasha.get("antardasha", "")
            maha_planet = dasha.get("mahadasha", "")

            if 0 < months_left <= 6:
                insights.append({
                    "type": "transition_warning",
                    "priority": "high",
                    "title": f"🔄 {antara_planet} antardasha ending in ~{months_left} months",
                    "insight": (
                        f"Your {antara_planet} sub-period within {maha_planet} mahadasha "
                        f"is ending around {antara_end_str}. This transition often brings "
                        f"shifts in the areas {antara_planet} governs. Be prepared for "
                        f"changes and stay flexible."
                    ),
                    "actionable": f"Good time to close pending matters related to {antara_planet}'s significations.",
                })
            elif 0 < months_left <= 2:
                insights.append({
                    "type": "transition_imminent",
                    "priority": "urgent",
                    "title": f"⚡ {antara_planet} antardasha ending very soon",
                    "insight": (
                        f"Major shift incoming — your {antara_planet} period ends {antara_end_str}. "
                        f"The energy around you will noticeably change. This is normal."
                    ),
                    "actionable": "Wrap up ongoing decisions. Avoid starting major new ventures this month.",
                })
        except (ValueError, TypeError):
            pass

    # Check mahadasha end date
    maha_end_str = dasha.get("mahadasha_end")
    if maha_end_str:
        try:
            maha_end = datetime.strptime(maha_end_str, "%B %Y")
            months_left = (maha_end.year - now.year) * 12 + (maha_end.month - now.month)
            maha_planet = dasha.get("mahadasha", "")

            if 0 < months_left <= 12:
                # Find next dasha planet from upcoming
                upcoming = dasha.get("upcoming_dashas") or []
                next_planet = upcoming[0]["planet"] if upcoming else "the next planet"
                insights.append({
                    "type": "major_transition",
                    "priority": "high",
                    "title": f"🌟 Major life shift: {maha_planet} → {next_planet} mahadasha in ~{months_left} months",
                    "insight": (
                        f"Your {maha_planet} mahadasha is ending around {maha_end_str}, "
                        f"transitioning to {next_planet}. This is one of the most significant "
                        f"astrological shifts — the entire theme of your life will evolve."
                    ),
                    "actionable": (
                        f"Start aligning with {next_planet}'s energy. "
                        f"This transition rewards preparation."
                    ),
                })
        except (ValueError, TypeError):
            pass

    return insights


# Dasha planet general significations
_DASHA_SIGNIFICATIONS: dict[str, str] = {
    "Sun": "authority, confidence, father, government, health vitality",
    "Moon": "emotions, mother, mind, travel, public image",
    "Mars": "energy, courage, property, siblings, surgery",
    "Mercury": "intellect, communication, business, education, skin",
    "Jupiter": "wisdom, children, wealth expansion, spirituality, guru",
    "Venus": "love, luxury, arts, marriage, vehicles, comfort",
    "Saturn": "discipline, career structure, delays that teach, karma, longevity",
    "Rahu": "ambition, foreign connections, unconventional paths, obsession",
    "Ketu": "spirituality, detachment, past karma resolution, moksha",
}


def _current_dasha_insights(dasha: dict[str, Any]) -> list[dict[str, str]]:
    """Interpret the current dasha period meaningfully."""
    insights: list[dict[str, str]] = []
    maha = dasha.get("mahadasha")
    antara = dasha.get("antardasha")

    if maha and antara and maha != antara:
        maha_sig = _DASHA_SIGNIFICATIONS.get(maha, "")
        antara_sig = _DASHA_SIGNIFICATIONS.get(antara, "")
        if maha_sig and antara_sig:
            insights.append({
                "type": "current_period",
                "priority": "medium",
                "title": f"📅 Current period: {maha}-{antara}",
                "insight": (
                    f"You're in {maha} mahadasha with {antara} antardasha. "
                    f"{maha} governs {maha_sig}. "
                    f"{antara} activates {antara_sig}. "
                    f"The blend of these energies defines your current life theme."
                ),
                "actionable": "",
            })

    return insights


def _yoga_insights(yogas: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Turn yoga detections into actionable life insights."""
    insights: list[dict[str, str]] = []

    for yoga in yogas[:3]:
        name = yoga.get("name", "")
        impact = yoga.get("impact", "")
        desc = yoga.get("description", "")

        if impact == "positive":
            insights.append({
                "type": "yoga_strength",
                "priority": "medium",
                "title": f"💫 {name} active in your chart",
                "insight": desc,
                "actionable": _yoga_actionable(name),
            })
        elif impact == "challenging":
            insights.append({
                "type": "yoga_challenge",
                "priority": "medium",
                "title": f"🔱 {name} — awareness needed",
                "insight": desc,
                "actionable": _yoga_actionable(name),
            })

    return insights


def _yoga_actionable(yoga_name: str) -> str:
    """Return actionable advice for each yoga."""
    actionables: dict[str, str] = {
        "Gajakesari Yoga": "Leverage your natural wisdom — teaching, writing, and mentoring bring the best results.",
        "Budhaditya Yoga": "Your communication skills are a superpower. Use them in business or education.",
        "Mangal Dosha": "Perform Mangal Shanti puja before marriage decisions. Wearing red coral may help.",
        "Sade Sati influence": "Practice patience. This period rewards discipline and inner work. Chant Shani mantra on Saturdays.",
        "Kemadruma Yoga": "Strengthen Moon through meditation, white clothing on Mondays, and pearl/moonstone.",
        "Hamsa Yoga": "Your spiritual path is strong. Temple visits, charity, and teaching amplify this yoga.",
        "Chandra-Mangal Yoga": "Financial decisions made with courage will pay off. Trust your instincts in money matters.",
        "Viparita Raja Yoga (potential)": "Your greatest strengths come from overcoming difficulties. Don't shy from challenges.",
    }
    return actionables.get(yoga_name, "")


def _transit_insights(transit_data: dict[str, Any]) -> list[dict[str, str]]:
    """Convert transit effects into proactive insights."""
    insights: list[dict[str, str]] = []

    for t in (transit_data.get("significant_transits") or [])[:3]:
        planet = t.get("planet", "")
        effects = t.get("effects") or []
        special = t.get("special_transit")
        sign = t.get("transit_sign", "")
        house = t.get("house_from_ascendant", "")

        if special == "sade_sati":
            insights.append({
                "type": "transit_major",
                "priority": "high",
                "title": f"🪐 Sade Sati active — Saturn in {sign}",
                "insight": effects[0] if effects else "Saturn is transiting near your Moon sign.",
                "actionable": "Chant Shani mantra, visit Shani temple on Saturdays, wear dark blue/black. Patience is your best tool.",
            })
        elif special == "jupiter_benefic":
            insights.append({
                "type": "transit_benefic",
                "priority": "medium",
                "title": f"✨ Jupiter blessing — transiting your {house}th house",
                "insight": effects[0] if effects else "Jupiter is in a favorable position for you.",
                "actionable": "Maximize this period for growth, learning, and expansion. Jupiter rewards generosity.",
            })
        elif effects:
            insights.append({
                "type": "transit_notable",
                "priority": "medium",
                "title": f"🔮 {planet} transiting your {house}th house",
                "insight": effects[0],
                "actionable": "",
            })

    return insights


def _planetary_strength_insights(chart: dict[str, Any]) -> list[dict[str, str]]:
    """Identify strong/weak placements and offer insight."""
    insights: list[dict[str, str]] = []
    enriched = chart.get("enriched_planets") or []

    for planet in enriched:
        if not isinstance(planet, dict):
            continue
        name = planet.get("name", "")
        is_retro = planet.get("is_retrograde", False)
        is_combust = planet.get("is_combust", False)

        if is_retro and name in {"Jupiter", "Saturn", "Venus", "Mercury"}:
            insights.append({
                "type": "retrograde",
                "priority": "low",
                "title": f"↩️ {name} retrograde in your birth chart",
                "insight": (
                    f"{name} is retrograde in your natal chart. This means {name}'s "
                    f"energy works more internally for you. You process "
                    f"{_DASHA_SIGNIFICATIONS.get(name, 'its themes')} differently than most."
                ),
                "actionable": f"Don't compare your {name}-related life areas with others. Your path is uniquely inward.",
            })

        if is_combust and name not in {"Sun"}:
            insights.append({
                "type": "combust",
                "priority": "low",
                "title": f"☀️ {name} combust — hidden potential",
                "insight": (
                    f"{name} is combust (too close to Sun) in your chart. "
                    f"This can reduce {name}'s visible expression but creates "
                    f"a deep inner reservoir of that energy."
                ),
                "actionable": f"Strengthen {name} through its gemstone or mantra.",
            })

    return insights


def format_predictions_for_prompt(insights: list[dict[str, str]]) -> str | None:
    """Format predictive insights for LLM prompt injection."""
    if not insights:
        return None

    high_priority = [i for i in insights if i.get("priority") in ("urgent", "high")]
    medium = [i for i in insights if i.get("priority") == "medium"]

    selected = high_priority[:3] + medium[:2]
    if not selected:
        return None

    parts = ["Proactive insights for this user (weave naturally when relevant):"]
    for i in selected:
        title = i.get("title", "")
        insight = i.get("insight", "")
        actionable = i.get("actionable", "")
        line = f"- {title}: {insight}"
        if actionable:
            line += f" → {actionable}"
        parts.append(line)

    return "\n".join(parts)
