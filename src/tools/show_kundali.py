from __future__ import annotations

from typing import Any


def show_kundali(chart: dict[str, Any]) -> str:
    if not chart:
        return "Kundali data is unavailable."
    if chart.get("summary"):
        return str(chart["summary"])

    ascendant = chart.get("ascendant_sign_name") or chart.get("ascendant_sign_index")
    planets = chart.get("planets_sign_names") or chart.get("planets_sign_indices") or {}
    planet_line = ", ".join(f"{name}: {value}" for name, value in planets.items())
    return f"Ascendant: {ascendant}. Planet placements: {planet_line}."
