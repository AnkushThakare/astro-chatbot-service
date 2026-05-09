from __future__ import annotations

from datetime import datetime
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


def calculate_vimshottari_dasha(birth_datetime: datetime, moon_sign_index: int) -> dict[str, Any]:
    seed = birth_datetime.year + birth_datetime.month + birth_datetime.day + moon_sign_index
    major_index = seed % len(MAHADASHA_SEQUENCE)
    sub_index = (seed + birth_datetime.hour) % len(MAHADASHA_SEQUENCE)
    return {
        "mahadasha": MAHADASHA_SEQUENCE[major_index],
        "antardasha": MAHADASHA_SEQUENCE[sub_index],
        "seed": seed,
    }
