from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from src.core.config import Settings
from src.observability.langfuse import end_llm_generation, fail_llm_generation, start_llm_generation


class GroqClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def is_configured(self) -> bool:
        return bool(self.settings.GROQ_API_KEY)

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> str:
        if not self.is_configured:
            raise RuntimeError("GROQ_API_KEY is not configured")

        payload: dict[str, Any] = {
            "model": self.settings.GROQ_MODEL,
            "messages": messages,
            "temperature": 0.4,
        }
        headers = {"Authorization": f"Bearer {self.settings.GROQ_API_KEY}"}
        generation = start_llm_generation(
            model=self.settings.GROQ_MODEL,
            messages=messages,
            provider="groq",
            session_id=session_id,
            user_id=user_id,
        )
        async with httpx.AsyncClient(timeout=self.settings.GROQ_TIMEOUT_SECONDS) as client:
            try:
                response = await client.post(
                    f"{self.settings.GROQ_BASE_URL}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
            except Exception as exc:
                fail_llm_generation(generation, str(exc))
                raise
        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage") if isinstance(data, dict) else None
        end_llm_generation(generation, output=content, usage=usage)
        return content

    @staticmethod
    def _extract_delta_content(payload: dict[str, Any]) -> str:
        choices = payload.get("choices")
        if not choices:
            return ""

        delta = choices[0].get("delta") or {}
        content = delta.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "".join(parts)
        return ""

    async def stream_generate(
        self,
        messages: list[dict[str, str]],
        *,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> AsyncGenerator[str, None]:
        if not self.is_configured:
            raise RuntimeError("GROQ_API_KEY is not configured")

        payload: dict[str, Any] = {
            "model": self.settings.GROQ_MODEL,
            "messages": messages,
            "temperature": 0.4,
            "stream": True,
        }
        headers = {"Authorization": f"Bearer {self.settings.GROQ_API_KEY}"}
        generation = start_llm_generation(
            model=self.settings.GROQ_MODEL,
            messages=messages,
            provider="groq",
            session_id=session_id,
            user_id=user_id,
        )
        accumulated_parts: list[str] = []

        async with httpx.AsyncClient(timeout=self.settings.GROQ_TIMEOUT_SECONDS) as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self.settings.GROQ_BASE_URL}/chat/completions",
                    json=payload,
                    headers=headers,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        chunk = line[6:].strip()
                        if chunk == "[DONE]":
                            break
                        try:
                            parsed = json.loads(chunk)
                        except json.JSONDecodeError:
                            continue

                        delta_text = self._extract_delta_content(parsed)
                        if not delta_text:
                            continue

                        accumulated_parts.append(delta_text)
                        yield delta_text
            except Exception as exc:
                fail_llm_generation(generation, str(exc))
                raise

        end_llm_generation(generation, output="".join(accumulated_parts), usage=None)
