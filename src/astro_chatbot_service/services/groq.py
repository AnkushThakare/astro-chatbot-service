from __future__ import annotations

from typing import Any

import httpx

from app.core.config import Settings


class GroqClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def generate(self, messages: list[dict[str, str]]) -> str:
        if not self.settings.GROQ_API_KEY:
            return (
                "Groq is not configured yet. The service scaffold is working, but "
                "you need GROQ_API_KEY for live model responses."
            )

        payload: dict[str, Any] = {
            "model": self.settings.GROQ_MODEL,
            "messages": messages,
            "temperature": 0.4,
        }
        headers = {"Authorization": f"Bearer {self.settings.GROQ_API_KEY}"}
        async with httpx.AsyncClient(timeout=self.settings.GROQ_TIMEOUT_SECONDS) as client:
            response = await client.post(
                f"{self.settings.GROQ_BASE_URL}/chat/completions",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
