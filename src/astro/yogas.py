from __future__ import annotations

from typing import Any


def detect_yogas(chart: dict[str, Any]) -> list[dict[str, str]]:
    """Detect common Vedic yogas from chart data."""
    yogas: list[dict[str, str]] = []
    planets = _planet_houses(chart)
    if not planets:
        return yogas

    moon_house = planets.get("moon")
    jupiter_house = planets.get("jupiter")
    sun_house = planets.get("sun")
    mercury_house = planets.get("mercury")
    saturn_house = planets.get("saturn")
    mars_house = planets.get("mars")
    venus_house = planets.get("venus")
    rahu_house = planets.get("rahu")
    ketu_house = planets.get("ketu")

    # Gajakesari Yoga: Jupiter in kendra (1,4,7,10) from Moon
    if moon_house is not None and jupiter_house is not None:
        diff = (jupiter_house - moon_house) % 12
        if diff in {0, 3, 6, 9}:  # 1st, 4th, 7th, 10th from Moon
            yogas.append({
                "name": "Gajakesari Yoga",
                "description": "Jupiter in kendra from Moon — brings wisdom, respect, and good fortune.",
                "impact": "positive",
            })

    # Budhaditya Yoga: Sun and Mercury in same house
    if sun_house is not None and mercury_house is not None and sun_house == mercury_house:
        yogas.append({
            "name": "Budhaditya Yoga",
            "description": "Sun-Mercury conjunction — sharp intellect, communication skills, and analytical ability.",
            "impact": "positive",
        })

    # Chandra-Mangal Yoga: Moon and Mars in same house
    if moon_house is not None and mars_house is not None and moon_house == mars_house:
        yogas.append({
            "name": "Chandra-Mangal Yoga",
            "description": "Moon-Mars conjunction — financial prosperity through courage and determination.",
            "impact": "positive",
        })

    # Mangal Dosha: Mars in 1st, 4th, 7th, 8th, or 12th house
    if mars_house is not None and mars_house in {1, 4, 7, 8, 12}:
        yogas.append({
            "name": "Mangal Dosha",
            "description": f"Mars in {mars_house}th house — can cause delays or friction in marriage. Remedies available.",
            "impact": "challenging",
        })

    # Sade Sati check: Saturn transiting over or near Moon sign
    # (simplified: Saturn and Moon in same or adjacent houses)
    if saturn_house is not None and moon_house is not None:
        diff = abs(saturn_house - moon_house)
        if diff <= 1 or diff >= 11:
            yogas.append({
                "name": "Sade Sati influence",
                "description": "Saturn near Moon sign — a period of karmic lessons, patience, and inner growth.",
                "impact": "challenging",
            })

    # Viparita Raja Yoga: lords of 6th, 8th, or 12th in each other's houses
    # (simplified: planets in 6th, 8th, or 12th can create hidden strength)
    dusthana = {6, 8, 12}
    dusthana_planets = [name for name, house in planets.items() if house in dusthana]
    if len(dusthana_planets) >= 2:
        yogas.append({
            "name": "Viparita Raja Yoga (potential)",
            "description": "Multiple planets in dusthana houses — challenges transform into hidden strengths over time.",
            "impact": "mixed",
        })

    # Kemadruma Yoga: No planets in 2nd or 12th from Moon
    if moon_house is not None:
        adjacent = {(moon_house % 12) + 1, ((moon_house - 2) % 12) + 1}
        has_adjacent = any(h in adjacent for name, h in planets.items() if name != "moon")
        if not has_adjacent:
            yogas.append({
                "name": "Kemadruma Yoga",
                "description": "No planets adjacent to Moon — can feel emotionally isolated. Remedy: strengthen Moon.",
                "impact": "challenging",
            })

    # Hamsa Yoga: Jupiter in kendra (1,4,7,10) in own or exaltation sign
    if jupiter_house is not None and jupiter_house in {1, 4, 7, 10}:
        jupiter_sign = _planet_sign(chart, "jupiter")
        if jupiter_sign in {"sagittarius", "pisces", "cancer"}:
            yogas.append({
                "name": "Hamsa Yoga",
                "description": "Jupiter strong in kendra — blessed with wisdom, spirituality, and respected status.",
                "impact": "positive",
            })

    return yogas


def _planet_houses(chart: dict[str, Any]) -> dict[str, int]:
    """Extract planet→house mapping from chart (handles both formats)."""
    result: dict[str, int] = {}

    # Try enriched format first (from core-service)
    d1 = (chart.get("charts") or {}).get("D1") or {}
    planets_list = d1.get("planets") or []
    if isinstance(planets_list, list) and planets_list:
        for p in planets_list:
            if not isinstance(p, dict):
                continue
            name = (p.get("name") or p.get("id") or "").lower()
            house = p.get("house_num")
            if name and isinstance(house, int):
                result[name] = house
        return result

    # Fallback: compute from sign indices
    planets_sign_indices = chart.get("planets_sign_indices") or {}
    house_sign_indices = chart.get("house_sign_indices") or {}
    for name, sign_idx in planets_sign_indices.items():
        if not isinstance(name, str) or not isinstance(sign_idx, int):
            continue
        for house, h_sign_idx in house_sign_indices.items():
            if h_sign_idx == sign_idx:
                result[name.lower()] = int(house)
                break
    return result


def _planet_sign(chart: dict[str, Any], planet_name: str) -> str | None:
    """Get sign name for a planet."""
    d1 = (chart.get("charts") or {}).get("D1") or {}
    for p in d1.get("planets") or []:
        if isinstance(p, dict):
            name = (p.get("name") or p.get("id") or "").lower()
            if name == planet_name:
                sign = p.get("sign_name")
                return sign.lower() if isinstance(sign, str) else None

    signs = chart.get("planets_sign_names") or {}
    sign = signs.get(planet_name.title())
    return sign.lower() if isinstance(sign, str) else None
