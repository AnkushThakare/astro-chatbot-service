from __future__ import annotations

from functools import lru_cache
from typing import Any

from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)


@lru_cache
def get_langfuse_client() -> Any | None:
    if not settings.langfuse_enabled:
        return None

    try:
        from langfuse import Langfuse
    except ImportError:
        logger.warning("Langfuse SDK is not installed; skipping LLM tracing")
        return None

    try:
        return Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_BASE_URL,
        )
    except Exception:
        logger.exception("Failed to initialize Langfuse client")
        return None


def start_llm_generation(
    *,
    model: str,
    messages: list[dict[str, str]],
    provider: str,
    session_id: str | None = None,
    user_id: str | None = None,
) -> Any | None:
    client = get_langfuse_client()
    if client is None:
        return None

    try:
        trace = client.trace(
            name="chat-message",
            user_id=user_id,
            session_id=session_id,
            input=messages,
            metadata={"provider": provider},
        )
        return trace.generation(
            name="llm-completion",
            model=model,
            input=messages,
            metadata={"provider": provider},
        )
    except Exception:
        logger.exception("Failed to start Langfuse generation")
        return None


def end_llm_generation(
    generation: Any | None,
    *,
    output: str,
    usage: dict[str, Any] | None = None,
) -> None:
    if generation is None:
        return

    try:
        if hasattr(generation, "end"):
            generation.end(output=output, usage=usage)
        else:
            generation.update(output=output, usage=usage)
        client = get_langfuse_client()
        if client is not None and hasattr(client, "flush"):
            client.flush()
    except Exception:
        logger.exception("Failed to finalize Langfuse generation")


def fail_llm_generation(generation: Any | None, error_message: str) -> None:
    if generation is None:
        return

    try:
        if hasattr(generation, "end"):
            generation.end(status_message=error_message, level="ERROR")
        else:
            generation.update(status_message=error_message, level="ERROR")
        client = get_langfuse_client()
        if client is not None and hasattr(client, "flush"):
            client.flush()
    except Exception:
        logger.exception("Failed to record Langfuse generation failure")
