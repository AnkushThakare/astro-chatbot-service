from __future__ import annotations

import asyncio
from datetime import datetime
import time
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
    CACHE_TTL_SECONDS = 120
    _query_cache: dict[tuple[str, str, int], tuple[float, list[dict[str, Any]]]] = {}
    _birth_details_cache: dict[str, tuple[float, dict[str, Any] | None]] = {}
    _birth_profile_cache: dict[str, tuple[float, dict[str, Any] | None]] = {}
    _shared_client: httpx.AsyncClient | None = None

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.base_url = settings.CORE_SERVICE_BASE_URL.rstrip("/")

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

    @classmethod
    def _cache_get(cls, namespace: str, query: str, limit: int) -> list[dict[str, Any]] | None:
        cache_key = (namespace, query.strip().lower(), limit)
        cached = cls._query_cache.get(cache_key)
        if cached is None:
            return None
        if time.time() - cached[0] > cls.CACHE_TTL_SECONDS:
            cls._query_cache.pop(cache_key, None)
            return None
        return cached[1]

    @classmethod
    def _cache_set(cls, namespace: str, query: str, limit: int, value: list[dict[str, Any]]) -> None:
        cache_key = (namespace, query.strip().lower(), limit)
        cls._query_cache[cache_key] = (time.time(), value)

    @classmethod
    def _birth_cache_get(cls, user_id: str, ttl_seconds: int) -> dict[str, Any] | None | type(Ellipsis):
        cached = cls._birth_details_cache.get(user_id)
        if cached is None:
            return Ellipsis
        if time.time() - cached[0] > ttl_seconds:
            cls._birth_details_cache.pop(user_id, None)
            return Ellipsis
        return cached[1]

    @classmethod
    def _birth_cache_set(cls, user_id: str, payload: dict[str, Any] | None) -> None:
        cls._birth_details_cache[user_id] = (time.time(), payload)

    @classmethod
    def invalidate_birth_details_cache(cls, user_id: str) -> None:
        cls._birth_details_cache.pop(user_id, None)
        cls._birth_profile_cache.pop(user_id, None)

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
        client = self._get_client(self.settings.CORE_SERVICE_TIMEOUT_SECONDS)
        for attempt in range(1 + MAX_RETRIES):
            try:
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

    @staticmethod
    def _deserialize_birth_details(payload: dict[str, Any]) -> dict[str, Any] | None:
        source = payload.get("birth_details") if isinstance(payload.get("birth_details"), dict) else payload
        if not isinstance(source, dict):
            return None

        birth_datetime = source.get("birth_datetime")
        if isinstance(birth_datetime, str) and birth_datetime.strip():
            resolved_birth_datetime = birth_datetime
        else:
            date_of_birth = source.get("date_of_birth")
            time_of_birth = source.get("time_of_birth")
            if not isinstance(date_of_birth, str) or not isinstance(time_of_birth, str):
                return None
            resolved_birth_datetime = f"{date_of_birth}T{time_of_birth}"

        try:
            latitude = float(source["latitude"])
            longitude = float(source["longitude"])
        except (KeyError, TypeError, ValueError):
            return None

        normalized: dict[str, Any] = {
            "name": source.get("name"),
            "latitude": latitude,
            "longitude": longitude,
            "birth_datetime": resolved_birth_datetime,
        }
        timezone_name = (
            source.get("timezone_str")
            or source.get("timezone_name")
            or source.get("timezone")
        )
        if isinstance(timezone_name, str) and timezone_name.strip():
            normalized["timezone_str"] = timezone_name.strip()
        return normalized

    @staticmethod
    def _serialize_partial_birth_profile(payload: dict[str, Any]) -> dict[str, Any]:
        updates: dict[str, Any] = {}
        date_parts = payload.get("date_parts")
        if isinstance(date_parts, tuple) and len(date_parts) == 3:
            day, month, year = date_parts
            updates["birth_date"] = f"{int(year):04d}-{int(month):02d}-{int(day):02d}"

        time_parts = payload.get("time_parts")
        if isinstance(time_parts, tuple) and len(time_parts) == 2:
            hour, minute = time_parts
            updates["birth_time"] = f"{int(hour):02d}:{int(minute):02d}:00"

        place = payload.get("place")
        if isinstance(place, str) and place.strip():
            updates["birth_place"] = place.strip()
        return updates

    @staticmethod
    def _deserialize_partial_birth_profile(payload: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(payload, dict):
            return None

        partial: dict[str, Any] = {
            "date_parts": None,
            "time_parts": None,
            "place": None,
        }

        birth_date = payload.get("birth_date")
        if isinstance(birth_date, str) and birth_date.strip():
            try:
                parsed_date = datetime.fromisoformat(f"{birth_date.strip()}T00:00:00")
                partial["date_parts"] = (
                    parsed_date.day,
                    parsed_date.month,
                    parsed_date.year,
                )
            except ValueError:
                pass

        birth_time = payload.get("birth_time")
        if isinstance(birth_time, str) and birth_time.strip():
            raw_time = birth_time.strip()
            for fmt in ("%H:%M:%S", "%H:%M"):
                try:
                    parsed_time = datetime.strptime(raw_time, fmt)
                    partial["time_parts"] = (parsed_time.hour, parsed_time.minute)
                    break
                except ValueError:
                    continue

        birth_place = payload.get("birth_place")
        if isinstance(birth_place, str) and birth_place.strip():
            partial["place"] = birth_place.strip()

        if all(partial[key] is None for key in ("date_parts", "time_parts", "place")):
            return None
        return partial

    async def get_user_birth_details(
        self,
        user_id: str,
        current_user: AuthenticatedUser | None,
    ) -> dict[str, Any] | None:
        cached = self._birth_cache_get(user_id, self.settings.BIRTH_DETAILS_CACHE_TTL_SECONDS)
        if cached is not Ellipsis:
            return cached

        try:
            headers = self._build_headers(current_user, auth_required=True)
        except ValueError:
            logger.info("Skipping core-service birth detail fetch because no bearer token is available")
            return None

        try:
            response = await self._request(
                "GET",
                f"/users/{user_id}/birth-details",
                headers=headers,
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                self._birth_cache_set(user_id, None)
                return None
            logger.warning("Core-service birth detail fetch failed: %s", exc)
            return None
        except Exception as exc:
            logger.warning("Core-service birth detail fetch failed: %s", exc)
            return None

        normalized = self._deserialize_birth_details(response.json())
        self._birth_cache_set(user_id, normalized)
        return normalized

    async def save_user_birth_details(
        self,
        user_id: str,
        payload: dict[str, Any],
        current_user: AuthenticatedUser | None,
    ) -> dict[str, Any] | None:
        try:
            headers = self._build_headers(current_user, auth_required=True)
            birth_details, _ = self._serialize_birth_details(payload)
        except ValueError as exc:
            logger.warning("Skipping core-service birth detail save: %s", exc)
            return None

        try:
            response = await self._request(
                "POST",
                f"/users/{user_id}/birth-details",
                headers=headers,
                json={"birth_details": birth_details},
            )
        except Exception as exc:
            logger.warning("Core-service birth detail save failed: %s", exc)
            return None

        self.invalidate_birth_details_cache(user_id)
        normalized = self._deserialize_birth_details(response.json())
        self._birth_cache_set(user_id, normalized)
        return normalized

    async def get_user_birth_profile(
        self,
        user_id: str,
        current_user: AuthenticatedUser | None,
    ) -> dict[str, Any] | None:
        cached = self._birth_profile_cache.get(user_id)
        if cached is not None:
            if time.time() - cached[0] <= self.settings.BIRTH_DETAILS_CACHE_TTL_SECONDS:
                return cached[1]
            self._birth_profile_cache.pop(user_id, None)

        try:
            headers = self._build_headers(current_user, auth_required=True)
        except ValueError:
            logger.info("Skipping core-service profile birth fetch because no bearer token is available")
            return None

        try:
            response = await self._request(
                "GET",
                "/auth/me",
                headers=headers,
            )
        except Exception as exc:
            logger.warning("Core-service profile birth fetch failed: %s", exc)
            return None

        normalized = self._deserialize_partial_birth_profile(response.json())
        self._birth_profile_cache[user_id] = (time.time(), normalized)
        return normalized

    async def save_user_birth_profile(
        self,
        user_id: str,
        payload: dict[str, Any],
        current_user: AuthenticatedUser | None,
    ) -> dict[str, Any] | None:
        updates = self._serialize_partial_birth_profile(payload)
        if not updates:
            return None

        try:
            headers = self._build_headers(current_user, auth_required=True)
        except ValueError:
            logger.info("Skipping core-service profile birth save because no bearer token is available")
            return None

        try:
            response = await self._request(
                "PUT",
                "/auth/me",
                headers=headers,
                json=updates,
            )
        except Exception as exc:
            logger.warning("Core-service profile birth save failed: %s", exc)
            return None

        normalized = self._deserialize_partial_birth_profile(response.json())
        self._birth_profile_cache[user_id] = (time.time(), normalized)
        return normalized

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
        cached = self._cache_get("products", query, limit)
        if cached is not None:
            return cached
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
        result = items if isinstance(items, list) else []
        self._cache_set("products", query, limit, result)
        return result

    async def list_home_puja_services(self, query: str, limit: int = 3) -> list[dict[str, Any]]:
        cached = self._cache_get("home_puja_services", query, limit)
        if cached is not None:
            return cached
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
        result = items if isinstance(items, list) else []
        self._cache_set("home_puja_services", query, limit, result)
        return result

    async def list_public_pandits(self, query: str, limit: int = 3) -> list[dict[str, Any]]:
        cached = self._cache_get("public_pandits", query, limit)
        if cached is not None:
            return cached
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
        result = items if isinstance(items, list) else []
        self._cache_set("public_pandits", query, limit, result)
        return result

    async def list_temple_services(self, query: str, limit: int = 3) -> list[dict[str, Any]]:
        cached = self._cache_get("temple_services", query, limit)
        if cached is not None:
            return cached
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
        result = items if isinstance(items, list) else []
        self._cache_set("temple_services", query, limit, result)
        return result

    async def search_pandits(
        self,
        query: str,
        current_user: AuthenticatedUser | None,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        cached = self._cache_get("consultant_pandits", query, limit)
        if cached is not None:
            return cached
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
        result = items if isinstance(items, list) else []
        self._cache_set("consultant_pandits", query, limit, result)
        return result

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
