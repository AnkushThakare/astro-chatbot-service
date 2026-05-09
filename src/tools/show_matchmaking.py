from __future__ import annotations

from typing import Any


def show_matchmaking(result: dict[str, Any]) -> str:
    if not result:
        return "Matchmaking data is unavailable."
    if result.get("summary"):
        return str(result["summary"])

    total_score = result.get("total_score")
    maximum_score = result.get("maximum_score")
    if isinstance(total_score, (int, float)) and isinstance(maximum_score, (int, float)) and maximum_score > 0:
        percent = (float(total_score) / float(maximum_score)) * 100
        return (
            f"Compatibility score: {float(total_score):.1f}/{float(maximum_score):.1f} "
            f"({percent:.0f}% match)."
        )

    kootas = result.get("kootas")
    if isinstance(kootas, dict) and kootas:
        highlights = ", ".join(f"{name}: {value}" for name, value in list(kootas.items())[:4])
        return f"Matchmaking breakdown: {highlights}."

    return "Matchmaking result received."
