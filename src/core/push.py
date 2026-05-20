"""Push notification service — delivers messages via Expo Push API.

Supports Expo push tokens (ExponentPushToken[...]) out of the box.
FCM/APNs tokens are routed through Expo's unified API automatically.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)

EXPO_PUSH_URL = "https://exp.host/--/api/v2/push/send"
_BATCH_SIZE = 100  # Expo recommends ≤100 per request


@dataclass
class PushResult:
    token: str
    success: bool
    error: str | None = None


class PushService:
    """Thin wrapper around the Expo Push API."""

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=15)
        return self._client

    async def send(
        self,
        tokens: list[str],
        title: str,
        body: str,
        *,
        data: dict[str, Any] | None = None,
        sound: str = "default",
        badge: int | None = None,
    ) -> list[PushResult]:
        """Send a push notification to one or more Expo push tokens.

        Returns a PushResult per token indicating success or failure.
        """
        if not tokens:
            return []

        messages = []
        for token in tokens:
            msg: dict[str, Any] = {
                "to": token,
                "title": title,
                "body": body,
                "sound": sound,
            }
            if data:
                msg["data"] = data
            if badge is not None:
                msg["badge"] = badge
            messages.append(msg)

        results: list[PushResult] = []
        client = await self._get_client()

        # Send in batches of 100
        for i in range(0, len(messages), _BATCH_SIZE):
            batch = messages[i : i + _BATCH_SIZE]
            try:
                response = await client.post(
                    EXPO_PUSH_URL,
                    json=batch,
                    headers={
                        "Accept": "application/json",
                        "Content-Type": "application/json",
                    },
                )
                response.raise_for_status()
                resp_data = response.json().get("data", [])
                for j, item in enumerate(resp_data):
                    token = batch[j]["to"]
                    if item.get("status") == "ok":
                        results.append(PushResult(token=token, success=True))
                    else:
                        error_msg = item.get("message") or item.get("details", {}).get("error", "unknown")
                        results.append(PushResult(token=token, success=False, error=str(error_msg)))
            except Exception as exc:
                logger.warning("push_batch_failed | batch_size=%d | error=%s", len(batch), exc)
                for msg in batch:
                    results.append(PushResult(token=msg["to"], success=False, error=str(exc)))

        return results

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
