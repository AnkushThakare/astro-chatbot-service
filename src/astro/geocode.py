from __future__ import annotations

from typing import Any

import httpx


def _approximate_timezone(longitude: float) -> str:
    offset = int(round(longitude / 15.0))
    if offset == 0:
        return "Etc/UTC"
    sign = "-" if offset > 0 else "+"
    return f"Etc/GMT{sign}{abs(offset)}"


async def geocode_place(place: str) -> dict[str, Any]:
    params = {
        "q": place,
        "format": "jsonv2",
        "limit": 1,
    }
    headers = {"User-Agent": "astro-chatbot-service/0.1"}
    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.get(
            "https://nominatim.openstreetmap.org/search",
            params=params,
            headers=headers,
        )
        response.raise_for_status()

    payload = response.json()
    if not payload:
        raise ValueError(f"Could not geocode place: {place}")

    result = payload[0]
    latitude = float(result["lat"])
    longitude = float(result["lon"])
    return {
        "place": result.get("display_name", place),
        "latitude": latitude,
        "longitude": longitude,
        "timezone": _approximate_timezone(longitude),
        "provider": "nominatim",
    }
