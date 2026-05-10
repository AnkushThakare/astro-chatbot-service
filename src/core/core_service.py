from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

import httpx

from src.auth.jwt import AuthenticatedUser
from src.core.config import Settings
from src.core.logging import get_logger

logger = get_logger(__name__)

MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 0.5


def _is_retryable(exc: Exception) -> bool:
    """Return True for transient errors worth retrying."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return isinstance(exc, (httpx.TimeoutException, httpx.ConnectError))


class CoreServiceError(Exception):
    """Raised when a core-service call fails with a non-transient error."""


class CoreServiceClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.base_url = settings.CORE_SERVICE_BASE_URL.rstrip("/")

    async def _request(
        self,
        method: str,
        path: str,
        *,
        headers: dict[str, str],
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> httpx.Response:
        """Send an HTTP request with automatic retry on transient failures."""
        last_exc: Exception | None = None
        for attempt in range(1 + MAX_RETRIES):
            try:
                async with httpx.AsyncClient(timeout=self.settings.CORE_SERVICE_TIMEOUT_SECONDS) as client:
                    response = await client.request(
                        method,
                        f"{self.base_url}{path}",
                        headers=headers,
                        params=params,
                        json=json,
                    )
                    response.raise_for_status()
                    return response
            except Exception as exc:
                last_exc = exc
                if attempt < MAX_RETRIES and _is_retryable(exc):
                    wait = RETRY_BACKOFF_SECONDS * (2 ** attempt)
                    logger.info(
                        "Retrying %s %s (attempt %d/%d) after %.1fs: %s",
                        method, path, attempt + 1, MAX_RETRIES, wait, exc,
                    )
                    await asyncio.sleep(wait)
                    continue
                raise
        raise last_exc  # type: ignore[misc]

    def _build_headers(
        self,
        current_user: AuthenticatedUser | None = None,
        *,
        auth_required: bool = False,
    ) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if current_user is not None and current_user.raw_token:
            headers["Authorization"] = f"Bearer {current_user.raw_token}"
        elif auth_required:
            raise ValueError("Authenticated core-service request requires a bearer token")
        return headers

    @staticmethod
    def _resolve_timezone_name(payload: dict[str, Any]) -> str | None:
        timezone_name = payload.get("timezone_str") or payload.get("timezone_name")
        if isinstance(timezone_name, str) and timezone_name.strip():
            return timezone_name.strip()

        birth_datetime = payload.get("birth_datetime")
        if isinstance(birth_datetime, str):
            try:
                birth_datetime = datetime.fromisoformat(birth_datetime)
            except ValueError:
                return None

        if isinstance(birth_datetime, datetime) and birth_datetime.tzinfo is not None:
            return getattr(birth_datetime.tzinfo, "key", None)
        return None

    @staticmethod
    def _serialize_birth_details(payload: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
        birth_datetime = payload.get("birth_datetime")
        if isinstance(birth_datetime, str):
            birth_datetime = datetime.fromisoformat(birth_datetime)
        if not isinstance(birth_datetime, datetime):
            raise ValueError("birth_datetime is required")

        timezone_name = CoreServiceClient._resolve_timezone_name(payload)
        if not timezone_name:
            return {}, None

        birth_details = {
            "name": payload.get("name") or "User",
            "date_of_birth": birth_datetime.date().isoformat(),
            "time_of_birth": birth_datetime.timetz().replace(tzinfo=None).isoformat(),
            "latitude": float(payload["latitude"]),
            "longitude": float(payload["longitude"]),
            "timezone_str": timezone_name,
        }
        return birth_details, timezone_name

    async def generate_kundli(
        self,
        payload: dict[str, Any],
        current_user: AuthenticatedUser | None,
    ) -> dict[str, Any] | None:
        birth_details, timezone_name = self._serialize_birth_details(payload)
        if not birth_details or timezone_name is None:
            logger.info("Skipping core-service kundli call because timezone information is unavailable")
            return None

        try:
            headers = self._build_headers(current_user, auth_required=True)
        except ValueError:
            logger.info("Skipping core-service kundli call because no bearer token is available")
            return None

        request_payload = {
            "birth_details": birth_details,
            "lang": payload.get("lang", "en"),
            "ayanamsa": str(payload.get("ayanamsha", "lahiri")).lower(),
            "house_system": payload.get("house_system", "W"),
            "include_summary": True,
            "summary_type": "medium",
        }

        try:
            response = await self._request(
                "POST", "/astrology/kundli", headers=headers, json=request_payload,
            )
            return response.json()
        except Exception as exc:
            logger.warning("Core-service kundli call failed: %s", exc)
            return None

    async def generate_matchmaking(
        self,
        payload: dict[str, Any],
        current_user: AuthenticatedUser | None,
    ) -> dict[str, Any] | None:
        primary_birth_details, _ = self._serialize_birth_details(payload["primary"]["birth_details"])
        partner_birth_details, _ = self._serialize_birth_details(payload["partner"]["birth_details"])
        if not primary_birth_details or not partner_birth_details:
            logger.info(
                "Skipping core-service matchmaking call because timezone information is unavailable"
            )
            return None

        try:
            headers = self._build_headers(current_user, auth_required=True)
        except ValueError:
            logger.info("Skipping core-service matchmaking call because no bearer token is available")
            return None

        request_payload = {
            "primary": {
                "birth_details": primary_birth_details,
                "gender": payload["primary"]["gender"],
            },
            "partner": {
                "birth_details": partner_birth_details,
                "gender": payload["partner"]["gender"],
            },
            "lang": payload.get("lang", "en"),
            "include_summary": payload.get("include_summary", True),
            "summary_type": payload.get("summary_type", "medium"),
        }

        try:
            response = await self._request(
                "POST", "/astrology/matchmaking", headers=headers, json=request_payload,
            )
            return response.json()
        except Exception as exc:
            logger.warning("Core-service matchmaking call failed: %s", exc)
            return None

    async def search_products(self, query: str, limit: int = 3) -> list[dict[str, Any]]:
        params = {
            "search": query,
            "limit": limit,
            "offset": 0,
            "in_stock": "true",
        }

        try:
            response = await self._request(
                "GET", "/catalog/products", headers=self._build_headers(), params=params,
            )
            payload = response.json()
        except Exception as exc:
            logger.warning("Core-service product search failed: %s", exc)
            return []

        items = payload.get("items")
        return items if isinstance(items, list) else []

    async def list_home_puja_services(self, query: str, limit: int = 3) -> list[dict[str, Any]]:
        params = {
            "search": query,
            "limit": limit,
            "offset": 0,
        }

        try:
            response = await self._request(
                "GET", "/catalog/services", headers=self._build_headers(), params=params,
            )
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            logger.warning("Core-service home puja service search failed: %s", exc)
            if exc.response.status_code < 500:
                raise CoreServiceError(f"Service search failed: {exc.response.text}") from exc
            return []
        except Exception as exc:
            logger.warning("Core-service home puja service search failed: %s", exc)
            return []

        items = payload.get("items")
        return items if isinstance(items, list) else []

    async def list_public_pandits(self, query: str, limit: int = 3) -> list[dict[str, Any]]:
        params = {
            "keywords": query,
            "limit": limit,
            "offset": 0,
        }

        try:
            response = await self._request(
                "GET", "/catalog/pandits", headers=self._build_headers(), params=params,
            )
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            logger.warning("Core-service public pandit search failed: %s", exc)
            if exc.response.status_code < 500:
                raise CoreServiceError(f"Pandit search failed: {exc.response.text}") from exc
            return []
        except Exception as exc:
            logger.warning("Core-service public pandit search failed: %s", exc)
            return []

        items = payload.get("items")
        return items if isinstance(items, list) else []

    async def list_temple_services(self, query: str, limit: int = 3) -> list[dict[str, Any]]:
        params = {
            "search": query,
            "page": 1,
            "size": limit,
        }

        try:
            response = await self._request(
                "GET", "/temples/services", headers=self._build_headers(), params=params,
            )
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            logger.warning("Core-service temple service search failed: %s", exc)
            if exc.response.status_code < 500:
                raise CoreServiceError(f"Temple service search failed: {exc.response.text}") from exc
            return []
        except Exception as exc:
            logger.warning("Core-service temple service search failed: %s", exc)
            return []

        items = payload.get("items")
        return items if isinstance(items, list) else []

    async def search_pandits(
        self,
        query: str,
        current_user: AuthenticatedUser | None,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        try:
            headers = self._build_headers(current_user, auth_required=True)
        except ValueError:
            logger.info("Skipping core-service pandit lookup because no bearer token is available")
            return []

        params: dict[str, Any] = {
            "category": "pandit",
            "limit": limit,
            "offset": 0,
        }
        specialty = self._infer_specialty(query)
        if specialty:
            params["specialty"] = specialty

        try:
            response = await self._request(
                "GET", "/providers/browse", headers=headers, params=params,
            )
            payload = response.json()
        except Exception as exc:
            logger.warning("Core-service pandit search failed: %s", exc)
            return []

        items = payload.get("items")
        return items if isinstance(items, list) else []

    async def preview_home_puja_price(
        self,
        payload: dict[str, Any],
        current_user: AuthenticatedUser | None,
    ) -> dict[str, Any] | None:
        try:
            headers = self._build_headers(current_user, auth_required=True)
        except ValueError:
            logger.info("Skipping booking price preview because no bearer token is available")
            return None

        try:
            response = await self._request(
                "POST", "/bookings/preview-price", headers=headers, json=payload,
            )
            return response.json()
        except httpx.HTTPStatusError as exc:
            logger.warning("Core-service booking price preview failed: %s", exc)
            if exc.response.status_code < 500:
                raise CoreServiceError(f"Price preview failed: {exc.response.text}") from exc
            return None
        except Exception as exc:
            logger.warning("Core-service booking price preview failed: %s", exc)
            return None

    async def create_home_puja_booking(
        self,
        payload: dict[str, Any],
        current_user: AuthenticatedUser | None,
    ) -> dict[str, Any] | None:
        try:
            headers = self._build_headers(current_user, auth_required=True)
        except ValueError:
            logger.info("Skipping home puja booking because no bearer token is available")
            return None

        try:
            response = await self._request(
                "POST", "/bookings", headers=headers, json=payload,
            )
            return response.json()
        except httpx.HTTPStatusError as exc:
            logger.warning("Core-service home puja booking failed: %s", exc)
            if exc.response.status_code < 500:
                raise CoreServiceError(f"Booking creation failed: {exc.response.text}") from exc
            return None
        except Exception as exc:
            logger.warning("Core-service home puja booking failed: %s", exc)
            return None

    async def create_temple_booking(
        self,
        payload: dict[str, Any],
        current_user: AuthenticatedUser | None,
    ) -> dict[str, Any] | None:
        try:
            headers = self._build_headers(current_user, auth_required=True)
        except ValueError:
            logger.info("Skipping temple booking because no bearer token is available")
            return None

        try:
            response = await self._request(
                "POST", "/temple/bookings", headers=headers, json=payload,
            )
            return response.json()
        except httpx.HTTPStatusError as exc:
            logger.warning("Core-service temple booking failed: %s", exc)
            if exc.response.status_code < 500:
                raise CoreServiceError(f"Temple booking failed: {exc.response.text}") from exc
            return None
        except Exception as exc:
            logger.warning("Core-service temple booking failed: %s", exc)
            return None

    @staticmethod
    def _infer_specialty(query: str) -> str | None:
        lowered = query.lower()
        if any(term in lowered for term in ("career", "job", "work", "promotion", "naukri")):
            return "career"
        if any(term in lowered for term in ("marriage", "relationship", "love", "partner", "shaadi", "rishta", "compatibility")):
            return "relationship"
        if any(term in lowered for term in ("health", "stress", "anxiety", "mind", "illness", "swasthya")):
            return "healing"
        if any(term in lowered for term in ("money", "finance", "business", "wealth", "dhan", "investment")):
            return "finance"
        if any(term in lowered for term in ("puja", "pooja", "ritual", "havan", "homam", "vedic")):
            return "ritual"
        if any(term in lowered for term in ("education", "study", "exam", "padhai", "university")):
            return "education"
        if any(term in lowered for term in ("family", "child", "pregnancy", "santan", "parivar")):
            return "family"
        if any(term in lowered for term in ("vaastu", "vastu", "griha", "house", "property", "real estate")):
            return "vaastu"
        if any(term in lowered for term in ("kundali", "kundli", "birth chart", "horoscope", "dasha", "transit", "gochar")):
            return "kundali"
        if any(term in lowered for term in ("remedy", "upay", "rudraksha", "mantra", "spiritual")):
            return "remedies"
        return None
