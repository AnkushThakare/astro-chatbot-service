from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

MAHADASHA_SEQUENCE = [
    "Ketu",
    "Venus",
    "Sun",
    "Moon",
    "Mars",
    "Rahu",
    "Jupiter",
    "Saturn",
    "Mercury",
]

# Vimshottari dasha periods in years (total = 120 years)
MAHADASHA_YEARS: dict[str, int] = {
    "Ketu": 7,
    "Venus": 20,
    "Sun": 6,
    "Moon": 10,
    "Mars": 7,
    "Rahu": 18,
    "Jupiter": 16,
    "Saturn": 19,
    "Mercury": 17,
}

# 27 Nakshatras and their ruling planets
NAKSHATRAS = [
    ("Ashwini", "Ketu"),
    ("Bharani", "Venus"),
    ("Krittika", "Sun"),
    ("Rohini", "Moon"),
    ("Mrigashira", "Mars"),
    ("Ardra", "Rahu"),
    ("Punarvasu", "Jupiter"),
    ("Pushya", "Saturn"),
    ("Ashlesha", "Mercury"),
    ("Magha", "Ketu"),
    ("Purva Phalguni", "Venus"),
    ("Uttara Phalguni", "Sun"),
    ("Hasta", "Moon"),
    ("Chitra", "Mars"),
    ("Swati", "Rahu"),
    ("Vishakha", "Jupiter"),
    ("Anuradha", "Saturn"),
    ("Jyeshtha", "Mercury"),
    ("Moola", "Ketu"),
    ("Purva Ashadha", "Venus"),
    ("Uttara Ashadha", "Sun"),
    ("Shravana", "Moon"),
    ("Dhanishta", "Mars"),
    ("Shatabhisha", "Rahu"),
    ("Purva Bhadrapada", "Jupiter"),
    ("Uttara Bhadrapada", "Saturn"),
    ("Revati", "Mercury"),
]

NAKSHATRA_SPAN_DEGREES = 360.0 / 27  # 13.333... degrees each


def moon_nakshatra(moon_longitude: float) -> tuple[int, str, str, float]:
    """Return (index, nakshatra_name, ruling_planet, fraction_elapsed) for Moon's longitude."""
    lon = moon_longitude % 360.0
    index = int(lon / NAKSHATRA_SPAN_DEGREES)
    if index >= 27:
        index = 26
    name, ruler = NAKSHATRAS[index]
    fraction_elapsed = (lon % NAKSHATRA_SPAN_DEGREES) / NAKSHATRA_SPAN_DEGREES
    return index, name, ruler, fraction_elapsed


def _dasha_start_planet_index(ruler: str) -> int:
    """Find index of the nakshatra ruler in the mahadasha sequence."""
    for i, planet in enumerate(MAHADASHA_SEQUENCE):
        if planet == ruler:
            return i
    return 0


def calculate_vimshottari_dasha(
    birth_datetime: datetime,
    moon_sign_index: int,
    *,
    moon_longitude: float | None = None,
) -> dict[str, Any]:
    """Calculate real Vimshottari dasha if moon_longitude is available, else fallback to approximation."""
    if moon_longitude is None:
        # Approximate: use sign midpoint as moon longitude
        moon_longitude = (moon_sign_index * 30.0) + 15.0

    nak_index, nak_name, nak_ruler, fraction_elapsed = moon_nakshatra(moon_longitude)

    # The dasha at birth is the nakshatra ruler's dasha
    start_index = _dasha_start_planet_index(nak_ruler)

    # Balance of the first dasha = remaining fraction of nakshatra * full period
    first_dasha_planet = MAHADASHA_SEQUENCE[start_index]
    first_dasha_total_years = MAHADASHA_YEARS[first_dasha_planet]
    balance_years = first_dasha_total_years * (1.0 - fraction_elapsed)
    balance_days = balance_years * 365.25

    # Build dasha timeline from birth
    timeline: list[dict[str, Any]] = []
    cursor = birth_datetime
    for cycle_offset in range(18):  # 2 full cycles covers 240 years
        idx = (start_index + cycle_offset) % len(MAHADASHA_SEQUENCE)
        planet = MAHADASHA_SEQUENCE[idx]
        if cycle_offset == 0:
            period_days = balance_days
        else:
            period_days = MAHADASHA_YEARS[planet] * 365.25
        end = cursor + timedelta(days=period_days)
        timeline.append({
            "planet": planet,
            "start": cursor,
            "end": end,
            "years": round(period_days / 365.25, 2),
        })
        cursor = end

    # Find current mahadasha
    now = datetime.now(timezone.utc) if birth_datetime.tzinfo else datetime.now()
    current_maha = timeline[0]
    for entry in timeline:
        if entry["start"] <= now < entry["end"]:
            current_maha = entry
            break

    # Calculate antardasha within current mahadasha
    maha_planet = current_maha["planet"]
    maha_start_idx = _dasha_start_planet_index(maha_planet)
    maha_total_days = (current_maha["end"] - current_maha["start"]).total_seconds() / 86400
    antardasha_cursor = current_maha["start"]
    current_antara = {"planet": maha_planet, "start": current_maha["start"], "end": current_maha["end"]}
    antardasha_timeline: list[dict[str, Any]] = []

    for ad_offset in range(9):
        ad_idx = (maha_start_idx + ad_offset) % len(MAHADASHA_SEQUENCE)
        ad_planet = MAHADASHA_SEQUENCE[ad_idx]
        # Antardasha proportion = sub_period_years / 120 * maha_total_days
        ad_days = (MAHADASHA_YEARS[ad_planet] / 120.0) * maha_total_days
        ad_end = antardasha_cursor + timedelta(days=ad_days)
        antardasha_timeline.append({
            "planet": ad_planet,
            "start": antardasha_cursor,
            "end": ad_end,
        })
        if antardasha_cursor <= now < ad_end:
            current_antara = {"planet": ad_planet, "start": antardasha_cursor, "end": ad_end}
        antardasha_cursor = ad_end

    # Build upcoming dasha changes (next 2 major transitions)
    upcoming: list[dict[str, str]] = []
    for entry in timeline:
        if entry["start"] > now:
            upcoming.append({
                "planet": entry["planet"],
                "starts": entry["start"].strftime("%B %Y"),
                "years": str(entry["years"]),
            })
            if len(upcoming) >= 2:
                break

    return {
        "mahadasha": current_maha["planet"],
        "mahadasha_start": current_maha["start"].strftime("%B %Y"),
        "mahadasha_end": current_maha["end"].strftime("%B %Y"),
        "antardasha": current_antara["planet"],
        "antardasha_start": current_antara["start"].strftime("%B %Y"),
        "antardasha_end": current_antara["end"].strftime("%B %Y"),
        "nakshatra": nak_name,
        "nakshatra_ruler": nak_ruler,
        "upcoming_dashas": upcoming,
        "timeline": [
            {
                "planet": e["planet"],
                "start": e["start"].strftime("%Y-%m"),
                "end": e["end"].strftime("%Y-%m"),
                "years": e["years"],
            }
            for e in timeline[:12]
        ],
    }
