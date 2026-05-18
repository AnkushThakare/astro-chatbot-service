from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

from src.astro.transits import (
    compute_current_transits,
    format_transits_for_prompt,
)


def _stub_birth_chart() -> dict[str, Any]:
    """A birth chart with Aries ascendant, Moon in Cancer (index 3)."""
    return {
        "ascendant_sign_index": 0,  # Aries
        "ascendant_sign_name": "Aries",
        "planets_sign_indices": {
            "Sun": 1,
            "Moon": 3,
            "Mars": 6,
            "Mercury": 1,
            "Jupiter": 8,
            "Venus": 10,
            "Saturn": 11,
        },
        "planets_sign_names": {
            "Sun": "Taurus",
            "Moon": "Cancer",
            "Mars": "Libra",
            "Mercury": "Taurus",
            "Jupiter": "Sagittarius",
            "Venus": "Aquarius",
            "Saturn": "Pisces",
        },
        "house_sign_indices": {
            "1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5,
            "7": 6, "8": 7, "9": 8, "10": 9, "11": 10, "12": 11,
        },
    }


def _stub_birth_details() -> dict[str, Any]:
    return {
        "latitude": 28.6139,
        "longitude": 77.2090,
        "birth_datetime": "1990-05-15T10:30:00",
    }


def _stub_transit_chart(saturn_sign: int = 3) -> dict[str, Any]:
    """Transit chart where Saturn is at given sign index."""
    signs = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
             "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"]
    return {
        "planets_sign_indices": {
            "Sun": 1,
            "Moon": 7,
            "Mars": 0,
            "Mercury": 2,
            "Jupiter": 3,
            "Venus": 5,
            "Saturn": saturn_sign,
        },
        "planets_sign_names": {
            "Sun": "Taurus",
            "Moon": "Scorpio",
            "Mars": "Aries",
            "Mercury": "Gemini",
            "Jupiter": "Cancer",
            "Venus": "Virgo",
            "Saturn": signs[saturn_sign],
        },
    }


def test_transit_detects_sade_sati() -> None:
    """Saturn transiting Moon sign (Cancer, index 3) = Sade Sati peak."""
    birth_chart = _stub_birth_chart()
    birth_details = _stub_birth_details()

    with patch("src.astro.transits.compute_kundali", new_callable=AsyncMock) as mock_kundali:
        mock_kundali.return_value = _stub_transit_chart(saturn_sign=3)
        result = asyncio.run(compute_current_transits(birth_chart, birth_details))

    assert result["available"] is True
    significant = result["significant_transits"]
    saturn_transit = next((t for t in significant if t["planet"] == "Saturn"), None)
    assert saturn_transit is not None
    assert saturn_transit.get("special_transit") == "sade_sati"


def test_transit_detects_jupiter_benefic() -> None:
    """Jupiter transiting 1st from Moon (same sign) = benefic."""
    birth_chart = _stub_birth_chart()
    birth_details = _stub_birth_details()

    with patch("src.astro.transits.compute_kundali", new_callable=AsyncMock) as mock_kundali:
        mock_kundali.return_value = _stub_transit_chart()
        result = asyncio.run(compute_current_transits(birth_chart, birth_details))

    assert result["available"] is True
    jupiter_transit = next(
        (t for t in result["significant_transits"] if t["planet"] == "Jupiter"), None
    )
    assert jupiter_transit is not None
    assert jupiter_transit.get("special_transit") == "jupiter_benefic"


def test_transit_handles_computation_failure() -> None:
    birth_chart = _stub_birth_chart()
    birth_details = _stub_birth_details()

    with patch("src.astro.transits.compute_kundali", new_callable=AsyncMock) as mock_kundali:
        mock_kundali.side_effect = Exception("API down")
        result = asyncio.run(compute_current_transits(birth_chart, birth_details))

    assert result["available"] is False


def test_format_transits_returns_none_when_unavailable() -> None:
    assert format_transits_for_prompt({"available": False}) is None


def test_format_transits_shows_significant() -> None:
    data = {
        "available": True,
        "significant_transits": [
            {
                "planet": "Saturn",
                "transit_sign": "Cancer",
                "house_from_ascendant": 4,
                "effects": ["Sade Sati (peak phase)"],
                "special_transit": "sade_sati",
            }
        ],
        "transits": [],
    }
    result = format_transits_for_prompt(data)
    assert result is not None
    assert "Saturn" in result
    assert "Cancer" in result


def test_transit_computes_house_from_moon_correctly() -> None:
    """Verify house-from-moon calculation."""
    birth_chart = _stub_birth_chart()  # Moon at sign 3
    birth_details = _stub_birth_details()

    with patch("src.astro.transits.compute_kundali", new_callable=AsyncMock) as mock_kundali:
        mock_kundali.return_value = _stub_transit_chart(saturn_sign=2)  # Saturn at Gemini (index 2)
        result = asyncio.run(compute_current_transits(birth_chart, birth_details))

    saturn = next((t for t in result["transits"] if t["planet"] == "Saturn"), None)
    assert saturn is not None
    # House from Moon: (2 - 3) % 12 + 1 = 11 + 1 = 12
    assert saturn["house_from_moon"] == 12  # 12th from Moon = Sade Sati rising


def test_transit_summary_not_empty_for_significant() -> None:
    """Transit summary should contain text when there are significant transits."""
    birth_chart = _stub_birth_chart()
    birth_details = _stub_birth_details()

    with patch("src.astro.transits.compute_kundali", new_callable=AsyncMock) as mock_kundali:
        mock_kundali.return_value = _stub_transit_chart(saturn_sign=3)
        result = asyncio.run(compute_current_transits(birth_chart, birth_details))

    assert result["summary"]
    assert len(result["summary"]) > 10
