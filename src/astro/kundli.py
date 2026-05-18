from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from src.astro.dasha import calculate_vimshottari_dasha
from src.astro.ephemeris import julian_day, swiss_ephemeris_status
from src.astro.yogas import detect_yogas
from src.core.config import settings

ZODIAC_SIGNS = [
    "Aries",
    "Taurus",
    "Gemini",
    "Cancer",
    "Leo",
    "Virgo",
    "Libra",
    "Scorpio",
    "Sagittarius",
    "Capricorn",
    "Aquarius",
    "Pisces",
]


def _extract_moon_longitude(chart: dict[str, Any]) -> float | None:
    """Extract Moon's sidereal longitude from core-service chart format."""
    d1 = (chart.get("charts") or {}).get("D1") or {}
    planets = d1.get("planets") or []
    for planet in planets:
        if not isinstance(planet, dict):
            continue
        name = (planet.get("name") or planet.get("id") or "").lower()
        if name == "moon":
            gd = planet.get("global_degree")
            if isinstance(gd, (int, float)):
                return float(gd)
            lon = planet.get("longitude")
            if isinstance(lon, (int, float)):
                return float(lon)
    return None


def _extract_planet_data(chart: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract enriched planet data from core-service chart format."""
    d1 = (chart.get("charts") or {}).get("D1") or {}
    planets = d1.get("planets") or []
    result: list[dict[str, Any]] = []
    for planet in planets:
        if not isinstance(planet, dict):
            continue
        result.append({
            "name": planet.get("name") or planet.get("id"),
            "house": planet.get("house_num"),
            "sign_id": planet.get("sign_id"),
            "longitude": planet.get("global_degree") or planet.get("longitude"),
            "local_degree": planet.get("local_degree"),
            "nakshatra": planet.get("nakshatra_name"),
            "nakshatra_id": planet.get("nakshatra_id"),
            "is_retrograde": planet.get("is_retrograde", False),
            "is_combust": planet.get("is_combust", False),
            "sign_lord": planet.get("sign_lord"),
        })
    return result


def _normalize_birth_datetime(value: datetime | str | None) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    raise ValueError("birth_datetime is required")


def _stub_chart(payload: dict[str, Any], source: str) -> dict[str, Any]:
    birth_datetime = _normalize_birth_datetime(payload.get("birth_datetime"))
    latitude = float(payload.get("latitude", 0.0))
    longitude = float(payload.get("longitude", 0.0))
    seed = int(
        abs(latitude * 10)
        + abs(longitude * 10)
        + birth_datetime.year
        + birth_datetime.month
        + birth_datetime.day
        + birth_datetime.hour
    )
    ascendant_index = seed % 12
    planets = {
        "Sun": (seed + 1) % 12,
        "Moon": (seed + 4) % 12,
        "Mars": (seed + 6) % 12,
        "Mercury": (seed + 3) % 12,
        "Jupiter": (seed + 8) % 12,
        "Venus": (seed + 10) % 12,
        "Saturn": (seed + 11) % 12,
    }
    houses = {house: (ascendant_index + house - 1) % 12 for house in range(1, 13)}
    dasha = calculate_vimshottari_dasha(birth_datetime, planets["Moon"])
    return {
        "backend": {
            "mode": source,
            "ephemeris": swiss_ephemeris_status(),
            "julian_day": julian_day(birth_datetime),
        },
        "name": payload.get("name"),
        "latitude": latitude,
        "longitude": longitude,
        "birth_datetime": birth_datetime.isoformat(),
        "ascendant_sign_index": ascendant_index,
        "ascendant_sign_name": ZODIAC_SIGNS[ascendant_index],
        "house_sign_indices": houses,
        "house_sign_names": {house: ZODIAC_SIGNS[index] for house, index in houses.items()},
        "planets_sign_indices": planets,
        "planets_sign_names": {name: ZODIAC_SIGNS[index] for name, index in planets.items()},
        "dasha": dasha,
        "summary": (
            f"Stub kundali computed with {source} mode. "
            f"Ascendant is {ZODIAC_SIGNS[ascendant_index]}."
        ),
    }


def _local_chart(payload: dict[str, Any]) -> dict[str, Any]:
    target_path = Path(settings.ASTROLOGY_ENGINE_PATH).resolve()
    if not target_path.exists():
        raise FileNotFoundError(f"Astrology engine path does not exist: {target_path}")

    if str(target_path) not in sys.path:
        sys.path.insert(0, str(target_path))

    from astrology_engine import get_vedic_chart_json

    birth_datetime = _normalize_birth_datetime(payload.get("birth_datetime"))
    chart = get_vedic_chart_json(
        float(payload["latitude"]),
        float(payload["longitude"]),
        birth_datetime,
        ayanamsha=payload.get("ayanamsha", "LAHIRI"),
        house_system=payload.get("house_system", "W"),
    )
    chart["backend"] = {"mode": "local", "ephemeris": swiss_ephemeris_status()}
    moon_index = int(chart.get("planets_sign_indices", {}).get("Moon", 0))
    chart.setdefault("dasha", calculate_vimshottari_dasha(birth_datetime, moon_index))
    return chart


async def _remote_chart(payload: dict[str, Any]) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{settings.ASTROLOGY_SERVICE_URL}/kundli/generate",
            json={
                "name": payload.get("name"),
                "latitude": float(payload["latitude"]),
                "longitude": float(payload["longitude"]),
                "birth_datetime": _normalize_birth_datetime(payload.get("birth_datetime")).isoformat(),
                "ayanamsa": payload.get("ayanamsha", "LAHIRI"),
                "house_system": payload.get("house_system", "W"),
            },
        )
        response.raise_for_status()
    chart = response.json()
    chart["backend"] = {"mode": "remote", "ephemeris": swiss_ephemeris_status()}
    moon_index = int(chart.get("planets_sign_indices", {}).get("Moon", 0))
    chart.setdefault(
        "dasha",
        calculate_vimshottari_dasha(_normalize_birth_datetime(payload.get("birth_datetime")), moon_index),
    )
    return chart


async def compute_kundali(payload: dict[str, Any]) -> dict[str, Any]:
    mode = settings.ASTROLOGY_ENGINE_MODE.lower()

    if mode == "stub":
        return _stub_chart(payload, "stub")
    if mode == "local":
        try:
            return _local_chart(payload)
        except Exception:
            return _stub_chart(payload, "local-fallback")
    if mode == "remote":
        try:
            return await _remote_chart(payload)
        except Exception:
            return _stub_chart(payload, "remote-fallback")
    return _stub_chart(payload, f"unknown-mode-{mode}")


async def compute_full_chart(payload: dict[str, Any]) -> dict[str, Any]:
    chart = await compute_kundali(payload)
    if "house_sign_indices" not in chart:
        ascendant_index = int(chart.get("ascendant_sign_index", 0))
        houses = {house: (ascendant_index + house - 1) % 12 for house in range(1, 13)}
        chart["house_sign_indices"] = houses
        chart["house_sign_names"] = {house: ZODIAC_SIGNS[index] for house, index in houses.items()}

    birth_datetime = _normalize_birth_datetime(payload.get("birth_datetime"))
    moon_index = int(chart.get("planets_sign_indices", {}).get("Moon", 0))
    moon_lon = _extract_moon_longitude(chart)

    # Always recompute dasha with real moon longitude when available
    chart["dasha"] = calculate_vimshottari_dasha(
        birth_datetime, moon_index, moon_longitude=moon_lon,
    )

    # Enrich with extracted planet data from core-service format
    enriched_planets = _extract_planet_data(chart)
    if enriched_planets:
        chart["enriched_planets"] = enriched_planets

    # Detect yogas
    chart["yogas"] = detect_yogas(chart)

    return chart
