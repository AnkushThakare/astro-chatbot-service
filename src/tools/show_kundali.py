from __future__ import annotations

from typing import Any


def show_kundali(chart: dict[str, Any]) -> str:
    if not chart:
        return "Kundali data is unavailable."
    if chart.get("summary"):
        return str(chart["summary"])
    charts = chart.get("charts")
    if isinstance(charts, dict):
        d1 = charts.get("D1") or {}
        ascendant = ((d1.get("ascendant") or {}).get("sign_name")) or "Unknown"
        planets = d1.get("planets") or []
        if isinstance(planets, list) and planets:
            planet_line = ", ".join(
                f"{planet.get('name', planet.get('id', 'Planet'))}: H{planet.get('house_num', '?')}"
                for planet in planets[:7]
                if isinstance(planet, dict)
            )
            return f"Ascendant: {ascendant}. Key placements: {planet_line}."

    ascendant = chart.get("ascendant_sign_name") or chart.get("ascendant_sign_index")
    planets = chart.get("planets_sign_names") or chart.get("planets_sign_indices") or {}
    planet_line = ", ".join(f"{name}: {value}" for name, value in planets.items())
    return f"Ascendant: {ascendant}. Planet placements: {planet_line}."
