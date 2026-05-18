import asyncio

from src.core.config import settings
from src.core.daily_insights import generate_daily_insight


def _birth_details() -> dict[str, object]:
    return {
        "name": "Ada",
        "latitude": 12.9716,
        "longitude": 77.5946,
        "birth_datetime": "1990-01-01T12:00:00+00:00",
        "timezone_str": "Etc/UTC",
        "ayanamsha": "LAHIRI",
        "house_system": "W",
    }


def test_generate_daily_insight_uses_memory_to_focus_the_message() -> None:
    insight = asyncio.run(
        generate_daily_insight(
            birth_details=_birth_details(),
            memory_context="- last_concern: career delay and promotion pressure\n- life_area: work",
            preferred_language="en",
        )
    )

    assert insight["focus_area"] == "career"
    assert insight["memory_context_used"] is True
    assert insight["headline"]
    assert insight["push_text"]
    assert insight["pattern_narrative"]
    assert "daily_check_in" in insight["topic_tags"]


def test_internal_daily_insight_requires_internal_api_key(client) -> None:
    response = client.post(
        "/api/v1/internal/daily-insight",
        json={"birth_details": _birth_details()},
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "Invalid internal API key"


def test_internal_daily_insight_returns_scheduler_ready_payload(client) -> None:
    response = client.post(
        "/api/v1/internal/daily-insight",
        json={
            "birth_details": _birth_details(),
            "long_term_memory": "- last_concern: relationship misunderstandings\n- preference: brief advice",
            "preferred_language": "en",
        },
        headers={"X-Internal-API-Key": settings.INTERNAL_API_KEY},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["insight"]["focus_area"] == "relationship"
    assert payload["insight"]["pattern_narrative"]
    assert payload["insight"]["headline"]
    assert payload["insight"]["message"]
    assert payload["insight"]["push_text"]
