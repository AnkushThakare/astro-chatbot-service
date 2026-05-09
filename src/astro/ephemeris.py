from __future__ import annotations

from datetime import datetime
from typing import Any

try:
    import swisseph as swe
except Exception:
    swe = None


def swiss_ephemeris_status() -> dict[str, Any]:
    if swe is None:
        return {
            "available": False,
            "backend": "unavailable",
        }
    return {
        "available": True,
        "backend": "pyswisseph",
        "version": getattr(swe, "version", "unknown"),
    }


def julian_day(birth_datetime: datetime) -> float | None:
    if swe is None:
        return None
    return swe.julday(
        birth_datetime.year,
        birth_datetime.month,
        birth_datetime.day,
        birth_datetime.hour + (birth_datetime.minute / 60.0),
    )

