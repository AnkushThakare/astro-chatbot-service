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


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in {429, 500, 502, 503, 504}
    return isinstance(exc, (httpx.TimeoutException, httpx.ConnectError))


# Keep old name for backward-compat in retryable check
_is_retryable_groq = _is_retryable


class LLMClient:
    """Provider-agnostic OpenAI-compatible LLM client.

    Works with any provider exposing /chat/completions (Groq, Gemini, OpenAI, etc).
    """

    _shared_clients: dict[str, httpx.AsyncClient] = {}

    def __init__(
        self,
        settings: Settings,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
        provider_name: str = "groq",
        timeout: int | None = None,
    ) -> None:
        self.settings = settings
        self.provider_name = provider_name
        self._api_key = api_key or settings.GROQ_API_KEY
        self._base_url = base_url or settings.GROQ_BASE_URL
        self._default_model = default_model or settings.GROQ_MODEL
        self._timeout = timeout or settings.GROQ_TIMEOUT_SECONDS
        self.last_usage: dict[str, Any] | None = None
        self._needs_thinking_config = self._default_model.startswith("gemini-3")

    def _apply_thinking_config(self, payload: dict[str, Any]) -> None:
        """For Gemini 3.x models, minimize thinking to avoid token budget exhaustion."""
        if not self._needs_thinking_config:
            return
        payload["reasoning_effort"] = "low"
        # Thinking tokens count against max_tokens — ensure enough room for actual output
        if "max_tokens" in payload and payload["max_tokens"] < 8192:
            payload["max_tokens"] = 8192

    @classmethod
    def _get_client(cls, base_url: str, timeout: int) -> httpx.AsyncClient:
        key = f"{base_url}:{timeout}"
        client = cls._shared_clients.get(key)
        if client is None or client.is_closed:
            client = httpx.AsyncClient(
                timeout=timeout,
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            )
            cls._shared_clients[key] = client
        return client

    @classmethod
    async def close(cls) -> None:
        for key, client in list(cls._shared_clients.items()):
            if not client.is_closed:
                await client.aclose()
            del cls._shared_clients[key]

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key)

    def _trace_metadata(self, extra_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "environment": "eval" if self.settings.EVAL_MODE else self.settings.APP_ENV,
            "provider": self.provider_name,
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
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        trace_metadata: dict[str, Any] | None = None,
    ) -> str:
        if not self.is_configured:
            raise RuntimeError(f"{self.provider_name} API key is not configured")
        self.last_usage = None

        payload: dict[str, Any] = {
            "model": model or self._default_model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format is not None:
            payload["response_format"] = response_format
        self._apply_thinking_config(payload)
        headers = {"Authorization": f"Bearer {self._api_key}"}
        generation = start_llm_generation(
            model=payload["model"],
            messages=messages,
            provider=self.provider_name,
            session_id=session_id,
            user_id=user_id,
            metadata=self._trace_metadata(trace_metadata),
        )
        last_exc: Exception | None = None
        response: httpx.Response | None = None
        client = self._get_client(self._base_url, self._timeout)
        for attempt in range(1 + MAX_RETRIES):
            try:
                response = await client.post(
                    f"{self._base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                break
            except Exception as exc:
                last_exc = exc
                if attempt < MAX_RETRIES and _is_retryable(exc):
                    wait = RETRY_BACKOFF_SECONDS * (2 ** attempt)
                    logger.info("Retrying %s generate (attempt %d/%d) after %.1fs: %s", self.provider_name, attempt + 1, MAX_RETRIES, wait, exc)
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
            raise RuntimeError(f"{self.provider_name} API key is not configured")
        self.last_usage = None

        payload: dict[str, Any] = {
            "model": model or self._default_model,
            "messages": messages,
            "temperature": temperature,
            "tools": tools,
            "tool_choice": tool_choice,
        }
        self._apply_thinking_config(payload)
        headers = {"Authorization": f"Bearer {self._api_key}"}
        generation = start_llm_generation(
            model=payload["model"],
            messages=messages,
            provider=self.provider_name,
            session_id=session_id,
            user_id=user_id,
            metadata=self._trace_metadata(trace_metadata),
        )
        last_exc: Exception | None = None
        response: httpx.Response | None = None
        client = self._get_client(self._base_url, self._timeout)
        for attempt in range(1 + MAX_RETRIES):
            try:
                response = await client.post(
                    f"{self._base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                break
            except Exception as exc:
                last_exc = exc
                if attempt < MAX_RETRIES and _is_retryable(exc):
                    wait = RETRY_BACKOFF_SECONDS * (2 ** attempt)
                    logger.info("Retrying %s tool call (attempt %d/%d) after %.1fs: %s", self.provider_name, attempt + 1, MAX_RETRIES, wait, exc)
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
        max_tokens: int | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        trace_metadata: dict[str, Any] | None = None,
    ) -> AsyncGenerator[str, None]:
        if not self.is_configured:
            raise RuntimeError(f"{self.provider_name} API key is not configured")
        self.last_usage = None

        payload: dict[str, Any] = {
            "model": model or self._default_model,
            "messages": messages,
            "temperature": 0.4,
            "stream": True,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        self._apply_thinking_config(payload)
        headers = {"Authorization": f"Bearer {self._api_key}"}
        generation = start_llm_generation(
            model=payload["model"],
            messages=messages,
            provider=self.provider_name,
            session_id=session_id,
            user_id=user_id,
            metadata=self._trace_metadata(trace_metadata),
        )
        accumulated_parts: list[str] = []

        last_connect_exc: Exception | None = None
        client = self._get_client(self._base_url, self._timeout)
        response_ctx = None
        for attempt in range(1 + MAX_RETRIES):
            try:
                response_ctx = client.stream(
                    "POST",
                    f"{self._base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                response = await response_ctx.__aenter__()
                response.raise_for_status()
                break
            except Exception as exc:
                # Clean up the context manager on failure before retrying
                if response_ctx is not None:
                    try:
                        await response_ctx.__aexit__(type(exc), exc, exc.__traceback__)
                    except Exception:
                        pass
                    response_ctx = None
                last_connect_exc = exc
                if attempt < MAX_RETRIES and _is_retryable(exc):
                    wait = RETRY_BACKOFF_SECONDS * (2 ** attempt)
                    logger.info("Retrying %s stream (attempt %d/%d) after %.1fs: %s", self.provider_name, attempt + 1, MAX_RETRIES, wait, exc)
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
            if response_ctx is not None:
                await response_ctx.__aexit__(None, None, None)

        end_llm_generation(generation, output="".join(accumulated_parts), usage=None)


# Backward-compatible alias — existing code that imports GroqClient still works
GroqClient = LLMClient


def create_llm_client(
    settings: Settings,
    *,
    role: str = "response",
) -> LLMClient:
    """Factory to create an LLMClient from settings for a given role.

    role='planner' → uses PLANNER_LLM_* settings (falls back to GROQ_*)
    role='response' → uses RESPONSE_LLM_* settings (falls back to GROQ_*)
    """
    if role == "planner":
        provider = settings.PLANNER_LLM_PROVIDER
        api_key = settings.PLANNER_LLM_API_KEY or settings.GROQ_API_KEY
        base_url = settings.PLANNER_LLM_BASE_URL or settings.GROQ_BASE_URL
        model = settings.PLANNER_LLM_MODEL or settings.GROQ_PLANNER_MODEL
    else:
        provider = settings.RESPONSE_LLM_PROVIDER
        api_key = settings.RESPONSE_LLM_API_KEY or settings.GROQ_API_KEY
        base_url = settings.RESPONSE_LLM_BASE_URL or settings.GROQ_BASE_URL
        model = settings.RESPONSE_LLM_MODEL or settings.GROQ_MODEL

    return LLMClient(
        settings,
        api_key=api_key,
        base_url=base_url,
        default_model=model,
        provider_name=provider,
    )
