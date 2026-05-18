from __future__ import annotations

from datetime import datetime
from typing import Any

from src.astro.kundli import compute_kundali, ZODIAC_SIGNS

# Planets we track for transits (excludes Rahu/Ketu for now — need separate computation)
TRANSIT_PLANETS = ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn"]

# Slow-moving planets whose transits are most significant
SLOW_PLANETS = {"Jupiter", "Saturn", "Rahu", "Ketu"}

# Transit significance by house (from natal Moon or Ascendant)
# Houses where slow planets create notable effects
SIGNIFICANT_TRANSIT_HOUSES = {
    1: "self, personality, new beginnings",
    4: "home, mother, emotional foundation",
    7: "relationships, partnerships, marriage",
    8: "transformation, hidden matters, sudden changes",
    10: "career, public image, authority",
    12: "losses, spirituality, foreign travel",
}

# Saturn-specific transit effects (Sade Sati and Dhaiya)
SATURN_SENSITIVE_HOUSES_FROM_MOON = {
    12: "Sade Sati (rising phase) — period of karmic restructuring begins",
    1: "Sade Sati (peak phase) — deepest transformation and pressure",
    2: "Sade Sati (setting phase) — gradual release, lessons consolidating",
    4: "Dhaiya (Kantaka Shani) — domestic and emotional pressure",
    8: "Ashtama Shani — sudden obstacles, health vigilance needed",
}

# Jupiter transit effects (generally benefic)
JUPITER_POSITIVE_HOUSES_FROM_MOON = {
    1: "expansion of self, new opportunities, optimism",
    5: "creativity, children, romance, speculative gains",
    7: "relationship growth, partnership blessings",
    9: "spiritual growth, luck, higher learning, travel",
    11: "income growth, wish fulfillment, social gains",
}


async def compute_current_transits(
    birth_chart: dict[str, Any],
    birth_details: dict[str, Any],
) -> dict[str, Any]:
    """Compute current planetary transits and their effect on the birth chart.

    Strategy: call the kundli API with current datetime + same location to get
    today's planetary positions, then overlay on birth chart houses.
    """
    now = datetime.now()

    # Build a transit payload — current time, same location as birth
    transit_payload = {
        "name": "transit",
        "latitude": float(birth_details.get("latitude", 0)),
        "longitude": float(birth_details.get("longitude", 0)),
        "birth_datetime": now,
        "ayanamsha": birth_details.get("ayanamsha", "LAHIRI"),
        "house_system": birth_details.get("house_system", "W"),
    }

    try:
        transit_chart = await compute_kundali(transit_payload)
    except Exception:
        return {"available": False, "reason": "transit_computation_failed"}

    # Extract transit planet positions (sign indices)
    transit_signs = transit_chart.get("planets_sign_indices") or {}
    transit_sign_names = transit_chart.get("planets_sign_names") or {}

    # Extract birth chart house-sign mapping
    birth_house_signs = birth_chart.get("house_sign_indices") or {}
    birth_planets_signs = birth_chart.get("planets_sign_indices") or {}

    # Build reverse map: sign_index → house_number (in birth chart)
    sign_to_birth_house: dict[int, int] = {}
    for house_str, sign_idx in birth_house_signs.items():
        house_num = int(house_str)
        sign_to_birth_house[sign_idx] = house_num

    # Moon sign index from birth chart (for house-from-moon calculations)
    birth_moon_sign = birth_planets_signs.get("Moon")
    birth_asc_sign = int(birth_chart.get("ascendant_sign_index", 0))

    transits: list[dict[str, Any]] = []
    active_effects: list[str] = []
    retrograde_planets: list[str] = []

    for planet in TRANSIT_PLANETS:
        transit_sign_idx = transit_signs.get(planet)
        if transit_sign_idx is None:
            continue

        # Which birth house is this transit planet sitting in?
        birth_house = sign_to_birth_house.get(transit_sign_idx)
        transit_sign_name = transit_sign_names.get(planet, ZODIAC_SIGNS[transit_sign_idx] if transit_sign_idx < 12 else "Unknown")

        # House from Moon (for traditional transit reading)
        house_from_moon = None
        if birth_moon_sign is not None:
            house_from_moon = ((transit_sign_idx - birth_moon_sign) % 12) + 1

        # House from Ascendant
        house_from_asc = ((transit_sign_idx - birth_asc_sign) % 12) + 1

        transit_entry: dict[str, Any] = {
            "planet": planet,
            "transit_sign": transit_sign_name,
            "transit_sign_index": transit_sign_idx,
            "house_from_ascendant": house_from_asc,
            "house_from_moon": house_from_moon,
            "birth_house": birth_house,
            "is_slow_planet": planet in SLOW_PLANETS,
        }

        # Determine significance
        effects: list[str] = []

        # Saturn-specific effects from Moon
        if planet == "Saturn" and house_from_moon is not None:
            saturn_effect = SATURN_SENSITIVE_HOUSES_FROM_MOON.get(house_from_moon)
            if saturn_effect:
                effects.append(saturn_effect)
                transit_entry["special_transit"] = "sade_sati" if house_from_moon in {12, 1, 2} else "dhaiya"

        # Jupiter-specific effects from Moon
        if planet == "Jupiter" and house_from_moon is not None:
            jupiter_effect = JUPITER_POSITIVE_HOUSES_FROM_MOON.get(house_from_moon)
            if jupiter_effect:
                effects.append(f"Jupiter blessing: {jupiter_effect}")
                transit_entry["special_transit"] = "jupiter_benefic"

        # General significance for slow planets in key houses
        if planet in SLOW_PLANETS and house_from_asc in SIGNIFICANT_TRANSIT_HOUSES:
            house_meaning = SIGNIFICANT_TRANSIT_HOUSES[house_from_asc]
            effects.append(f"{planet} transiting {house_from_asc}th house — area of {house_meaning}")

        transit_entry["effects"] = effects
        transit_entry["is_significant"] = len(effects) > 0
        transits.append(transit_entry)
        active_effects.extend(effects)

    # Summarize
    significant_transits = [t for t in transits if t["is_significant"]]

    return {
        "available": True,
        "computed_at": now.isoformat(),
        "transits": transits,
        "significant_transits": significant_transits,
        "active_effects": active_effects,
        "summary": _build_transit_summary(significant_transits),
    }


def _build_transit_summary(significant_transits: list[dict[str, Any]]) -> str:
    """Build a human-readable summary of significant transits."""
    if not significant_transits:
        return "No major planetary transits affecting your chart right now."

    parts: list[str] = []
    for t in significant_transits[:4]:  # Max 4 most important
        planet = t["planet"]
        sign = t["transit_sign"]
        house_asc = t["house_from_ascendant"]
        effects = t.get("effects") or []
        special = t.get("special_transit")

        if special == "sade_sati":
            parts.append(f"🪐 {planet} in {sign} ({effects[0] if effects else 'Sade Sati active'})")
        elif special == "jupiter_benefic":
            parts.append(f"✨ {planet} in {sign} (transiting {house_asc}th house — {effects[0] if effects else 'benefic'})")
        else:
            effect_text = effects[0] if effects else f"transiting your {house_asc}th house"
            parts.append(f"🔮 {planet} in {sign}: {effect_text}")

    return "\n".join(parts)


def format_transits_for_prompt(transit_data: dict[str, Any]) -> str | None:
    """Format transit data for injection into LLM prompt."""
    if not transit_data.get("available"):
        return None

    significant = transit_data.get("significant_transits") or []
    if not significant:
        # Still show current positions of slow planets
        all_transits = transit_data.get("transits") or []
        slow = [t for t in all_transits if t.get("is_slow_planet")]
        if not slow:
            return None
        parts = ["Current planetary transits:"]
        for t in slow:
            parts.append(f"- {t['planet']} in {t['transit_sign']} (your {t['house_from_ascendant']}th house)")
        return "\n".join(parts)

    parts = ["Current transits affecting your chart:"]
    for t in significant[:4]:
        planet = t["planet"]
        sign = t["transit_sign"]
        house = t["house_from_ascendant"]
        effects = t.get("effects") or []
        effect_str = "; ".join(effects[:2]) if effects else ""
        line = f"- {planet} in {sign} → {house}th house"
        if effect_str:
            line += f" ({effect_str})"
        parts.append(line)

    return "\n".join(parts)
