from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from src.core.config import Settings
from src.core.logging import get_logger
from src.observability.langfuse import end_llm_generation, fail_llm_generation, start_llm_generation

logger = get_logger(__name__)

MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 0.5


def _is_retryable_groq(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in {429, 500, 502, 503, 504}
    return isinstance(exc, (httpx.TimeoutException, httpx.ConnectError))


class GroqClient:
    _shared_client: httpx.AsyncClient | None = None

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.last_usage: dict[str, Any] | None = None

    @classmethod
    def _get_client(cls, timeout: int) -> httpx.AsyncClient:
        if cls._shared_client is None or cls._shared_client.is_closed:
            cls._shared_client = httpx.AsyncClient(
                timeout=timeout,
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            )
        return cls._shared_client

    @classmethod
    async def close(cls) -> None:
        if cls._shared_client is not None and not cls._shared_client.is_closed:
            await cls._shared_client.aclose()
            cls._shared_client = None

    @property
    def is_configured(self) -> bool:
        return bool(self.settings.GROQ_API_KEY)

    def _trace_metadata(self, extra_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "environment": "eval" if self.settings.EVAL_MODE else self.settings.APP_ENV,
        }
        run_id = os.getenv("EVAL_RUN_ID")
        if self.settings.EVAL_MODE and run_id:
            metadata["run_id"] = run_id
        if extra_metadata:
            metadata.update(extra_metadata)
        return metadata

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.4,
        response_format: dict[str, Any] | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        trace_metadata: dict[str, Any] | None = None,
    ) -> str:
        if not self.is_configured:
            raise RuntimeError("GROQ_API_KEY is not configured")
        self.last_usage = None

        payload: dict[str, Any] = {
            "model": model or self.settings.GROQ_MODEL,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        headers = {"Authorization": f"Bearer {self.settings.GROQ_API_KEY}"}
        generation = start_llm_generation(
            model=payload["model"],
            messages=messages,
            provider="groq",
            session_id=session_id,
            user_id=user_id,
            metadata=self._trace_metadata(trace_metadata),
        )
        last_exc: Exception | None = None
        response: httpx.Response | None = None
        client = self._get_client(self.settings.GROQ_TIMEOUT_SECONDS)
        for attempt in range(1 + MAX_RETRIES):
            try:
                response = await client.post(
                    f"{self.settings.GROQ_BASE_URL}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                break
            except Exception as exc:
                last_exc = exc
                if attempt < MAX_RETRIES and _is_retryable_groq(exc):
                    wait = RETRY_BACKOFF_SECONDS * (2 ** attempt)
                    logger.info("Retrying Groq generate (attempt %d/%d) after %.1fs: %s", attempt + 1, MAX_RETRIES, wait, exc)
                    await asyncio.sleep(wait)
                    continue
                fail_llm_generation(generation, str(exc))
                raise
        if response is None:
            fail_llm_generation(generation, str(last_exc))
            raise last_exc  # type: ignore[misc]
        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage") if isinstance(data, dict) else None
        self.last_usage = usage if isinstance(usage, dict) else None
        end_llm_generation(generation, output=content, usage=usage)
        return content

    async def generate_with_tools(
        self,
        messages: list[dict[str, str]],
        *,
        tools: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 0.1,
        tool_choice: str | dict[str, Any] = "auto",
        session_id: str | None = None,
        user_id: str | None = None,
        trace_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.is_configured:
            raise RuntimeError("GROQ_API_KEY is not configured")
        self.last_usage = None

        payload: dict[str, Any] = {
            "model": model or self.settings.GROQ_MODEL,
            "messages": messages,
            "temperature": temperature,
            "tools": tools,
            "tool_choice": tool_choice,
        }
        headers = {"Authorization": f"Bearer {self.settings.GROQ_API_KEY}"}
        generation = start_llm_generation(
            model=payload["model"],
            messages=messages,
            provider="groq",
            session_id=session_id,
            user_id=user_id,
            metadata=self._trace_metadata(trace_metadata),
        )
        last_exc: Exception | None = None
        response: httpx.Response | None = None
        client = self._get_client(self.settings.GROQ_TIMEOUT_SECONDS)
        for attempt in range(1 + MAX_RETRIES):
            try:
                response = await client.post(
                    f"{self.settings.GROQ_BASE_URL}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                break
            except Exception as exc:
                last_exc = exc
                if attempt < MAX_RETRIES and _is_retryable_groq(exc):
                    wait = RETRY_BACKOFF_SECONDS * (2 ** attempt)
                    logger.info("Retrying Groq tool call (attempt %d/%d) after %.1fs: %s", attempt + 1, MAX_RETRIES, wait, exc)
                    await asyncio.sleep(wait)
                    continue
                fail_llm_generation(generation, str(exc))
                raise
        if response is None:
            fail_llm_generation(generation, str(last_exc))
            raise last_exc  # type: ignore[misc]
        data = response.json()
        message = data["choices"][0]["message"]
        usage = data.get("usage") if isinstance(data, dict) else None
        self.last_usage = usage if isinstance(usage, dict) else None
        end_llm_generation(generation, output=json.dumps(message, default=str), usage=usage)
        return message

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
        model: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        trace_metadata: dict[str, Any] | None = None,
    ) -> AsyncGenerator[str, None]:
        if not self.is_configured:
            raise RuntimeError("GROQ_API_KEY is not configured")
        self.last_usage = None

        payload: dict[str, Any] = {
            "model": model or self.settings.GROQ_MODEL,
            "messages": messages,
            "temperature": 0.4,
            "stream": True,
        }
        headers = {"Authorization": f"Bearer {self.settings.GROQ_API_KEY}"}
        generation = start_llm_generation(
            model=payload["model"],
            messages=messages,
            provider="groq",
            session_id=session_id,
            user_id=user_id,
            metadata=self._trace_metadata(trace_metadata),
        )
        accumulated_parts: list[str] = []

        last_connect_exc: Exception | None = None
        client = self._get_client(self.settings.GROQ_TIMEOUT_SECONDS)
        for attempt in range(1 + MAX_RETRIES):
            try:
                response_ctx = client.stream(
                    "POST",
                    f"{self.settings.GROQ_BASE_URL}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                response = await response_ctx.__aenter__()
                response.raise_for_status()
                break
            except Exception as exc:
                last_connect_exc = exc
                if attempt < MAX_RETRIES and _is_retryable_groq(exc):
                    wait = RETRY_BACKOFF_SECONDS * (2 ** attempt)
                    logger.info("Retrying Groq stream (attempt %d/%d) after %.1fs: %s", attempt + 1, MAX_RETRIES, wait, exc)
                    await asyncio.sleep(wait)
                    continue
                fail_llm_generation(generation, str(exc))
                raise
        else:
            fail_llm_generation(generation, str(last_connect_exc))
            raise last_connect_exc  # type: ignore[misc]

        try:
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
        except asyncio.CancelledError:
            fail_llm_generation(generation, "stream_cancelled")
            raise
        except Exception as exc:
            fail_llm_generation(generation, str(exc))
            raise
        finally:
            await response_ctx.__aexit__(None, None, None)

        end_llm_generation(generation, output="".join(accumulated_parts), usage=None)
