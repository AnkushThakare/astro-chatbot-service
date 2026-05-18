from __future__ import annotations

import pytest

from src.core.product_policy import (
    CONCERN_PRODUCT_MAP,
    PLANET_PRODUCT_MAP,
    enrich_product_query,
    validate_product_search_query,
)


class TestValidateProductSearchQuery:
    def test_strips_blocked_terms(self) -> None:
        assert "sapphire" not in validate_product_search_query("blue sapphire bracelet")

    def test_preserves_allowed_terms(self) -> None:
        result = validate_product_search_query("rudraksha bracelet")
        assert "rudraksha" in result
        assert "bracelet" in result

    def test_returns_original_if_fully_stripped(self) -> None:
        result = validate_product_search_query("neelam pukhraj")
        assert result == "neelam pukhraj"  # Returns original as fallback


class TestPlanetProductMap:
    def test_all_nine_planets_mapped(self) -> None:
        expected = {"sun", "moon", "mars", "mercury", "jupiter", "venus", "saturn", "rahu", "ketu"}
        assert set(PLANET_PRODUCT_MAP.keys()) == expected

    def test_each_planet_has_required_fields(self) -> None:
        for planet, data in PLANET_PRODUCT_MAP.items():
            assert "primary" in data, f"{planet} missing primary product"
            assert "bracelet" in data, f"{planet} missing bracelet"
            assert "concerns" in data, f"{planet} missing concerns"
            assert "day" in data, f"{planet} missing day"
            assert len(data["concerns"]) >= 3, f"{planet} has too few concerns"


class TestConcernProductMap:
    @pytest.mark.parametrize(
        ("concern", "expected_contains"),
        [
            ("career", "5 mukhi"),
            ("sade sati", "7 mukhi"),
            ("anxiety", "2 mukhi"),
            ("mangal dosha", "3 mukhi"),
            ("exams", "4 mukhi"),
            ("love", "6 mukhi"),
            ("kaal sarp", "8 mukhi"),
            ("spiritual", "9 mukhi"),
        ],
    )
    def test_concern_maps_to_correct_product(self, concern: str, expected_contains: str) -> None:
        product = CONCERN_PRODUCT_MAP[concern]
        assert expected_contains in product


class TestEnrichProductQuery:
    def test_specific_mukhi_query_passes_through(self) -> None:
        result = enrich_product_query("7 mukhi rudraksha")
        assert "7 mukhi" in result

    def test_concern_based_enrichment(self) -> None:
        result = enrich_product_query("something for anxiety")
        assert "2 mukhi" in result

    def test_career_concern_enrichment(self) -> None:
        result = enrich_product_query("help with career")
        assert "5 mukhi" in result

    def test_afflicted_planet_enrichment(self) -> None:
        # Afflicted planets used when no concern keyword matches
        result = enrich_product_query("rudraksha for me", afflicted_planets=["Saturn"])
        assert "7 mukhi" in result

    def test_dasha_based_enrichment(self) -> None:
        result = enrich_product_query("rudraksha bracelet", current_dasha="Rahu")
        assert "8 mukhi" in result

    def test_falls_back_to_validation(self) -> None:
        result = enrich_product_query("something random")
        assert result == "something random"

    def test_sade_sati_enrichment(self) -> None:
        result = enrich_product_query("I'm going through sade sati")
        assert "7 mukhi" in result

    def test_mangal_dosha_enrichment(self) -> None:
        result = enrich_product_query("mangal dosha remedy")
        assert "3 mukhi" in result

    def test_afflicted_planet_takes_priority_over_generic(self) -> None:
        result = enrich_product_query("rudraksha", afflicted_planets=["Moon", "Saturn"])
        assert "2 mukhi" in result  # Moon is first afflicted planet
