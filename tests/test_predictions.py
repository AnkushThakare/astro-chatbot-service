from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pytest

from src.astro.predictions import (
    generate_predictive_insights,
    format_predictions_for_prompt,
)


def _chart_with_dasha(
    maha: str = "Saturn",
    antara: str = "Mercury",
    maha_end_months: int = 8,
    antara_end_months: int = 3,
) -> dict[str, Any]:
    now = datetime.now()
    maha_end = now + timedelta(days=maha_end_months * 30)
    antara_end = now + timedelta(days=antara_end_months * 30)
    return {
        "dasha": {
            "mahadasha": maha,
            "antardasha": antara,
            "mahadasha_start": "January 2020",
            "mahadasha_end": maha_end.strftime("%B %Y"),
            "antardasha_start": "March 2026",
            "antardasha_end": antara_end.strftime("%B %Y"),
            "upcoming_dashas": [
                {"planet": "Mercury", "starts": "March 2027", "years": "17"},
            ],
        },
        "yogas": [
            {
                "name": "Gajakesari Yoga",
                "description": "Jupiter in kendra from Moon",
                "impact": "positive",
            },
            {
                "name": "Mangal Dosha",
                "description": "Mars in 7th house",
                "impact": "challenging",
            },
        ],
        "enriched_planets": [
            {"name": "Jupiter", "is_retrograde": True, "is_combust": False},
            {"name": "Mercury", "is_retrograde": False, "is_combust": True},
        ],
    }


def test_generates_dasha_transition_warning() -> None:
    chart = _chart_with_dasha(antara_end_months=4)
    insights = generate_predictive_insights(chart)
    transition = [i for i in insights if i["type"] == "transition_warning"]
    assert len(transition) >= 1
    assert "Mercury" in transition[0]["title"]


def test_generates_major_transition_for_mahadasha_ending_soon() -> None:
    chart = _chart_with_dasha(maha_end_months=10)
    insights = generate_predictive_insights(chart)
    major = [i for i in insights if i["type"] == "major_transition"]
    assert len(major) >= 1
    assert "Saturn" in major[0]["title"]
    assert "Mercury" in major[0]["title"]  # Next planet


def test_generates_yoga_insights() -> None:
    chart = _chart_with_dasha()
    insights = generate_predictive_insights(chart)
    yoga_insights = [i for i in insights if i["type"].startswith("yoga_")]
    assert len(yoga_insights) >= 1
    names = [i["title"] for i in yoga_insights]
    assert any("Gajakesari" in n for n in names)


def test_generates_retrograde_insight() -> None:
    chart = _chart_with_dasha()
    insights = generate_predictive_insights(chart)
    retro = [i for i in insights if i["type"] == "retrograde"]
    assert len(retro) >= 1
    assert "Jupiter" in retro[0]["title"]


def test_generates_combust_insight() -> None:
    # Minimal chart with just combust planet and short dasha to reduce other insights
    chart = {
        "dasha": {
            "mahadasha": "Sun",
            "antardasha": "Moon",
            "mahadasha_start": "January 2020",
            "mahadasha_end": "January 2030",
            "antardasha_start": "March 2026",
            "antardasha_end": "January 2028",
        },
        "yogas": [],
        "enriched_planets": [
            {"name": "Mercury", "is_retrograde": False, "is_combust": True},
        ],
    }
    insights = generate_predictive_insights(chart)
    combust = [i for i in insights if i["type"] == "combust"]
    assert len(combust) >= 1
    assert "Mercury" in combust[0]["title"]


def test_transit_insights_included_when_data_available() -> None:
    chart = _chart_with_dasha()
    transit_data = {
        "available": True,
        "significant_transits": [
            {
                "planet": "Saturn",
                "transit_sign": "Pisces",
                "house_from_ascendant": 12,
                "effects": ["Sade Sati (peak phase) — deepest transformation"],
                "special_transit": "sade_sati",
            }
        ],
    }
    insights = generate_predictive_insights(chart, transit_data)
    transit = [i for i in insights if i["type"] == "transit_major"]
    assert len(transit) >= 1
    assert "Sade Sati" in transit[0]["title"]


def test_format_predictions_returns_none_for_empty() -> None:
    assert format_predictions_for_prompt([]) is None


def test_format_predictions_prioritizes_high_priority() -> None:
    insights = [
        {"type": "transition_warning", "priority": "high", "title": "High prio", "insight": "Important", "actionable": "Do this"},
        {"type": "current_period", "priority": "medium", "title": "Medium prio", "insight": "Less important", "actionable": ""},
        {"type": "retrograde", "priority": "low", "title": "Low prio", "insight": "Minor", "actionable": ""},
    ]
    result = format_predictions_for_prompt(insights)
    assert result is not None
    assert "High prio" in result
    assert "Medium prio" in result
    assert "Low prio" not in result  # Low priority excluded


def test_caps_at_six_insights() -> None:
    chart = _chart_with_dasha(antara_end_months=2, maha_end_months=6)
    chart["yogas"] = [
        {"name": f"Yoga{i}", "description": f"desc{i}", "impact": "positive"}
        for i in range(10)
    ]
    insights = generate_predictive_insights(chart)
    assert len(insights) <= 6
