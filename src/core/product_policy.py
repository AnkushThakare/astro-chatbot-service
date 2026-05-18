from __future__ import annotations

from typing import Any

ALLOWED_PRODUCT_CATEGORIES = {
    "rudraksha",
    "rudraksha_mala",
    "bracelet",
    "pendant",
    # Add more categories as your catalog expands
}

BLOCKED_PRODUCT_TERMS = {
    "gemstone",
    "yantra",
    "sapphire",
    "ruby",
    "emerald",
    "pukhraj",
    "neelam",
    "panna",
    "moonga",
}

# ── Planet → Product Mapping ─────────────────────────────────────
# Maps planetary afflictions to specific catalog products.
# Used by the recommendation engine to turn chart analysis into product suggestions.
PLANET_PRODUCT_MAP: dict[str, dict[str, Any]] = {
    "sun": {
        "primary": "1 mukhi rudraksha",
        "alternative": "12 mukhi rudraksha",
        "bracelet": "solar energy bracelet",
        "concerns": ["low confidence", "eyesight", "authority issues", "father problems"],
        "day": "Sunday",
    },
    "moon": {
        "primary": "2 mukhi rudraksha",
        "alternative": "2 mukhi rudraksha pendant",
        "bracelet": "calming energy bracelet",
        "concerns": ["anxiety", "emotional instability", "sleep issues", "depression", "mother problems", "mental stress"],
        "day": "Monday",
    },
    "mars": {
        "primary": "3 mukhi rudraksha",
        "alternative": "3 mukhi rudraksha pendant",
        "bracelet": "mars protection bracelet",
        "concerns": ["anger", "low energy", "accidents", "property disputes", "mangal dosha", "delayed marriage"],
        "day": "Tuesday",
    },
    "mercury": {
        "primary": "4 mukhi rudraksha",
        "alternative": "4 mukhi rudraksha pendant",
        "bracelet": "focus clarity bracelet",
        "concerns": ["communication problems", "poor memory", "skin issues", "business loss", "exam stress"],
        "day": "Wednesday",
    },
    "jupiter": {
        "primary": "5 mukhi rudraksha",
        "alternative": "5 mukhi rudraksha mala",
        "bracelet": "wisdom prosperity bracelet",
        "concerns": ["financial instability", "lack of wisdom", "delayed children", "guru issues", "spiritual disconnect"],
        "day": "Thursday",
    },
    "venus": {
        "primary": "6 mukhi rudraksha",
        "alternative": "6 mukhi rudraksha pendant",
        "bracelet": "love harmony bracelet",
        "concerns": ["relationship troubles", "lack of luxury", "artistic blocks", "reproductive health"],
        "day": "Friday",
    },
    "saturn": {
        "primary": "7 mukhi rudraksha",
        "alternative": "14 mukhi rudraksha",
        "bracelet": "saturn protection bracelet",
        "concerns": ["sade sati", "delays", "obstacles", "career stagnation", "chronic health", "legal troubles", "shani dhaiya"],
        "day": "Saturday",
    },
    "rahu": {
        "primary": "8 mukhi rudraksha",
        "alternative": "8 mukhi rudraksha pendant",
        "bracelet": "rahu protection bracelet",
        "concerns": ["confusion", "addiction", "fear", "sudden setbacks", "kaal sarp dosha", "foreign travel problems"],
        "day": "Saturday",
    },
    "ketu": {
        "primary": "9 mukhi rudraksha",
        "alternative": "9 mukhi rudraksha pendant",
        "bracelet": "spiritual grounding bracelet",
        "concerns": ["spiritual confusion", "detachment", "past karma", "mysterious health problems"],
        "day": "Tuesday",
    },
}

# ── Concern → Product Mapping ────────────────────────────────────
# Maps life concerns (what users actually say) to product search queries.
# This handles "I'm stressed about career" → appropriate product.
CONCERN_PRODUCT_MAP: dict[str, str] = {
    # Career & Finance
    "career": "5 mukhi rudraksha",
    "career stagnation": "7 mukhi rudraksha",
    "job": "5 mukhi rudraksha",
    "business": "4 mukhi rudraksha",
    "wealth": "5 mukhi rudraksha mala",
    "money problems": "5 mukhi rudraksha",
    "financial": "5 mukhi rudraksha",
    # Relationships
    "marriage": "6 mukhi rudraksha",
    "relationship": "6 mukhi rudraksha",
    "love": "6 mukhi rudraksha",
    "delayed marriage": "3 mukhi rudraksha",
    "mangal dosha": "3 mukhi rudraksha",
    "manglik": "3 mukhi rudraksha",
    # Health & Wellbeing
    "health": "5 mukhi rudraksha mala",
    "stress": "2 mukhi rudraksha",
    "anxiety": "2 mukhi rudraksha",
    "depression": "2 mukhi rudraksha",
    "sleep": "2 mukhi rudraksha",
    "anger": "3 mukhi rudraksha",
    "peace": "5 mukhi rudraksha mala",
    # Spiritual
    "meditation": "5 mukhi rudraksha mala",
    "spiritual": "9 mukhi rudraksha",
    "protection": "rudraksha bracelet",
    "negative energy": "8 mukhi rudraksha",
    "evil eye": "rudraksha bracelet",
    # Planetary periods
    "sade sati": "7 mukhi rudraksha",
    "shani": "7 mukhi rudraksha",
    "saturn": "7 mukhi rudraksha",
    "rahu": "8 mukhi rudraksha",
    "ketu": "9 mukhi rudraksha",
    "kaal sarp": "8 mukhi rudraksha",
    # Education
    "exams": "4 mukhi rudraksha",
    "studies": "4 mukhi rudraksha",
    "education": "4 mukhi rudraksha",
    "memory": "4 mukhi rudraksha",
    # General
    "general protection": "5 mukhi rudraksha",
    "all planets": "10 mukhi rudraksha",
    "navagraha": "rudraksha bracelet",
}


def validate_product_search_query(search_query: str) -> str:
    """Sanitize a product search query by stripping blocked terms.

    Returns the cleaned query.  If nothing useful remains after
    stripping, returns the original query so the catalog API can
    still attempt a best-effort match against real products.
    """
    words = search_query.lower().split()
    cleaned = [w for w in words if w not in BLOCKED_PRODUCT_TERMS]
    result = " ".join(cleaned).strip()
    return result if result else search_query


def enrich_product_query(
    search_query: str,
    afflicted_planets: list[str] | None = None,
    current_dasha: str | None = None,
) -> str:
    """Enrich a product search query using chart context.

    If the user's query is vague (e.g., "rudraksha for protection"),
    use their chart data to suggest the most relevant product.
    """
    query_lower = search_query.lower()

    # First check if query already has a specific mukhi number
    if "mukhi" in query_lower:
        return validate_product_search_query(search_query)

    # Try to match user concern to specific product
    for concern, product in CONCERN_PRODUCT_MAP.items():
        if concern in query_lower:
            return product

    # If we have afflicted planets from chart, use the most relevant
    if afflicted_planets:
        planet = afflicted_planets[0].lower()
        planet_data = PLANET_PRODUCT_MAP.get(planet)
        if planet_data:
            return planet_data["primary"]

    # If we know current dasha, suggest for that planet
    if current_dasha:
        planet_data = PLANET_PRODUCT_MAP.get(current_dasha.lower())
        if planet_data:
            return planet_data["primary"]

    return validate_product_search_query(search_query)
