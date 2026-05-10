from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

from fastapi import HTTPException, Request, status

from src.core.logging import get_logger

logger = get_logger(__name__)


class InMemoryRateLimiter:
    """Simple in-memory sliding window rate limiter.

    Tracks request timestamps per key (IP or user ID) and rejects
    requests that exceed the configured limit within the window.

    For production with multiple workers, replace with Redis-backed
    rate limiting.
    """

    def __init__(
        self,
        requests_per_window: int = 30,
        window_seconds: int = 60,
    ) -> None:
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self._buckets: dict[str, list[float]] = defaultdict(list)

    def _cleanup_bucket(self, key: str, now: float) -> None:
        cutoff = now - self.window_seconds
        bucket = self._buckets[key]
        self._buckets[key] = [ts for ts in bucket if ts > cutoff]

    def check(self, key: str) -> bool:
        """Return True if the request is allowed, False if rate limited."""
        now = time.monotonic()
        self._cleanup_bucket(key, now)
        if len(self._buckets[key]) >= self.requests_per_window:
            return False
        self._buckets[key].append(now)
        return True

    def remaining(self, key: str) -> int:
        now = time.monotonic()
        self._cleanup_bucket(key, now)
        return max(0, self.requests_per_window - len(self._buckets[key]))


# Global instance — shared across requests within a single worker
chat_rate_limiter = InMemoryRateLimiter(requests_per_window=30, window_seconds=60)


def get_rate_limit_key(request: Request) -> str:
    """Extract rate limit key from request — prefer user ID, fall back to IP."""
    if hasattr(request.state, "user_id") and request.state.user_id:
        return f"user:{request.state.user_id}"
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return f"ip:{forwarded.split(',')[0].strip()}"
    client_host = request.client.host if request.client else "unknown"
    return f"ip:{client_host}"


def check_rate_limit(request: Request) -> None:
    """Raise 429 if the caller has exceeded the chat rate limit."""
    key = get_rate_limit_key(request)
    if not chat_rate_limiter.check(key):
        logger.warning("Rate limit exceeded for %s", key)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests. Please wait before sending more messages.",
            headers={"Retry-After": str(chat_rate_limiter.window_seconds)},
        )
