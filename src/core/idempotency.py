from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any

from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)

try:
    import redis.asyncio as redis
except ImportError:  # pragma: no cover - optional dependency fallback
    redis = None

_RESERVED_MARKER = "__reserved__"


@dataclass
class _IdempotencyEntry:
    created_at: float
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    completed: bool = False
    rendered_events: list[str] = field(default_factory=list)


class InMemoryIdempotencyStore:
    def __init__(self, ttl_seconds: int = 300) -> None:
        self.ttl_seconds = ttl_seconds
        self._entries: dict[str, _IdempotencyEntry] = {}
        self._lock = asyncio.Lock()

    def _purge_expired(self, now: float) -> None:
        for key, entry in list(self._entries.items()):
            if now - entry.created_at >= self.ttl_seconds:
                self._entries.pop(key, None)

    async def reserve_or_replay(self, key: str) -> tuple[bool, list[str] | None]:
        while True:
            async with self._lock:
                now = time.monotonic()
                self._purge_expired(now)
                entry = self._entries.get(key)
                if entry is None:
                    self._entries[key] = _IdempotencyEntry(created_at=now)
                    return False, None
                if entry.completed:
                    return True, list(entry.rendered_events)
                waiter = entry.ready
            await waiter.wait()

    async def complete(self, key: str, rendered_events: list[str]) -> None:
        async with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return
            entry.completed = True
            entry.rendered_events = list(rendered_events)
            entry.ready.set()

    async def abort(self, key: str) -> None:
        async with self._lock:
            entry = self._entries.pop(key, None)
            if entry is not None:
                entry.ready.set()


class ChatIdempotencyStore:
    POLL_INTERVAL_SECONDS = 0.1

    def __init__(
        self,
        *,
        ttl_seconds: int = 300,
        redis_url: str | None = None,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.redis_url = redis_url or settings.REDIS_URL
        self._fallback = InMemoryIdempotencyStore(ttl_seconds=ttl_seconds)
        self._redis_client: Any | None = None
        self._redis_lock = asyncio.Lock()

    async def _get_redis(self) -> Any | None:
        if not self.redis_url or redis is None:
            return None

        if self._redis_client is not None:
            return self._redis_client

        async with self._redis_lock:
            if self._redis_client is not None:
                return self._redis_client
            try:
                client = redis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                )
                await client.ping()
            except Exception as exc:
                logger.warning("redis_idempotency_unavailable", exc_info=exc)
                return None
            self._redis_client = client
            return self._redis_client

    async def close(self) -> None:
        if self._redis_client is not None:
            try:
                await self._redis_client.aclose()
            finally:
                self._redis_client = None

    async def reserve_or_replay(self, key: str) -> tuple[bool, list[str] | None]:
        client = await self._get_redis()
        if client is None:
            self._fallback.ttl_seconds = self.ttl_seconds
            return await self._fallback.reserve_or_replay(key)

        while True:
            try:
                reserved = await client.set(
                    key,
                    _RESERVED_MARKER,
                    ex=self.ttl_seconds,
                    nx=True,
                )
                if reserved:
                    return False, None

                cached = await client.get(key)
                if cached is None:
                    continue
                if cached == _RESERVED_MARKER:
                    await asyncio.sleep(self.POLL_INTERVAL_SECONDS)
                    continue
                return True, json.loads(cached)
            except Exception as exc:
                logger.warning("redis_idempotency_reserve_failed", exc_info=exc)
                self._fallback.ttl_seconds = self.ttl_seconds
                return await self._fallback.reserve_or_replay(key)

    async def complete(self, key: str, rendered_events: list[str]) -> None:
        client = await self._get_redis()
        if client is None:
            self._fallback.ttl_seconds = self.ttl_seconds
            await self._fallback.complete(key, rendered_events)
            return

        try:
            await client.set(key, json.dumps(rendered_events), ex=self.ttl_seconds)
        except Exception as exc:
            logger.warning("redis_idempotency_complete_failed", exc_info=exc)
            self._fallback.ttl_seconds = self.ttl_seconds
            await self._fallback.complete(key, rendered_events)

    async def abort(self, key: str) -> None:
        client = await self._get_redis()
        if client is None:
            self._fallback.ttl_seconds = self.ttl_seconds
            await self._fallback.abort(key)
            return

        try:
            await client.delete(key)
        except Exception as exc:
            logger.warning("redis_idempotency_abort_failed", exc_info=exc)
            self._fallback.ttl_seconds = self.ttl_seconds
            await self._fallback.abort(key)


chat_idempotency_store = ChatIdempotencyStore()
