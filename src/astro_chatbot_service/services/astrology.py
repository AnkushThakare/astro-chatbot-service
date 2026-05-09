from __future__ import annotations

from typing import Any

import httpx

from app.core.config import Settings
from astro_chatbot_service.adapters.local_astrology_engine import generate_chart
from astro_chatbot_service.models.schemas import BirthDetails


class AstrologyService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def build_summary(self, birth_details: BirthDetails | None) -> str | None:
        if birth_details is None:
            return None

        try:
            chart = await self._load_chart(birth_details)
        except Exception as exc:
            return f"Astrology context unavailable: {exc.__class__.__name__}"

        planets = chart.get("planets_sign_indices", {})
        ascendant_index = chart.get("ascendant_sign_index")
        planet_line = ", ".join(f"{name}:{sign}" for name, sign in planets.items())
        return (
            f"Name: {birth_details.name or 'unknown'}. "
            f"Ascendant sign index: {ascendant_index}. "
            f"Planet sign indices: {planet_line}."
        )

    async def _load_chart(self, birth_details: BirthDetails) -> dict[str, Any]:
        payload = {
            "name": birth_details.name,
            "latitude": birth_details.latitude,
            "longitude": birth_details.longitude,
            "birth_datetime": birth_details.birth_datetime.isoformat(),
            "ayanamsha": birth_details.ayanamsha,
            "house_system": birth_details.house_system,
        }

        if self.settings.ASTROLOGY_ENGINE_MODE == "local":
            return generate_chart(self.settings.ASTROLOGY_ENGINE_PATH, payload)

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{self.settings.ASTROLOGY_SERVICE_URL}/kundli/generate",
                json={
                    "name": payload["name"],
                    "latitude": payload["latitude"],
                    "longitude": payload["longitude"],
                    "birth_datetime": payload["birth_datetime"],
                    "ayanamsa": payload["ayanamsha"],
                    "house_system": payload["house_system"],
                },
            )
            response.raise_for_status()
        return response.json()
