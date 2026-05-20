"""Local product catalog index for intelligent matching.

Fetches all products from core-service, caches them in memory, and provides
smart local matching that doesn't depend on core-service text search quality.
"""

from __future__ import annotations

import re
import time
from typing import Any

from src.core.logging import get_logger

logger = get_logger(__name__)

# How often to refresh the full catalog from core-service (seconds)
CATALOG_REFRESH_INTERVAL = 300  # 5 minutes

# Broad queries to fetch the full catalog — covers all product types
_SEED_QUERIES = ["rudraksha", "bracelet", "mala", "pendant"]

# Planetary / concern aliases for matching product names
_PLANET_ALIASES: dict[str, set[str]] = {
    "sun": {"sun", "surya", "1 mukhi", "1mukhi", "ek mukhi"},
    "moon": {"moon", "chandra", "2 mukhi", "2mukhi"},
    "mars": {"mars", "mangal", "3 mukhi", "3mukhi"},
    "mercury": {"mercury", "budh", "4 mukhi", "4mukhi"},
    "jupiter": {"jupiter", "guru", "brihaspati", "5 mukhi", "5mukhi"},
    "venus": {"venus", "shukra", "6 mukhi", "6mukhi"},
    "saturn": {"saturn", "shani", "7 mukhi", "7mukhi"},
    "rahu": {"rahu", "8 mukhi", "8mukhi"},
    "ketu": {"ketu", "9 mukhi", "9mukhi"},
}

_CONCERN_ALIASES: dict[str, set[str]] = {
    "career": {"career", "job", "profession", "promotion", "work", "naukri"},
    "protection": {"protection", "negative", "evil", "shield", "suraksha"},
    "health": {"health", "healing", "wellness", "swasthya"},
    "peace": {"peace", "calm", "stress", "anxiety", "shanti", "tension"},
    "meditation": {"meditation", "spiritual", "dhyan", "sadhana"},
    "relationship": {"relationship", "love", "marriage", "partner", "shaadi"},
    "wealth": {"wealth", "money", "finance", "prosperity", "dhan", "lakshmi"},
    "energy": {"energy", "power", "strength", "shakti", "oorja"},
    "education": {"education", "study", "exam", "memory", "studies", "padhai"},
}

# Map concern keywords → expected product name tokens for boosting
_CONCERN_PRODUCT_BOOST: dict[str, list[str]] = {
    "career": ["5 mukhi", "jupiter", "wisdom", "prosperity"],
    "career stagnation": ["7 mukhi", "saturn", "obstacle"],
    "protection": ["protection", "shield", "bracelet"],
    "peace": ["2 mukhi", "calming", "peace", "mala"],
    "meditation": ["5 mukhi", "mala", "meditation", "rudraksha mala"],
    "relationship": ["6 mukhi", "venus", "love", "harmony"],
    "anxiety": ["2 mukhi", "moon", "calming"],
    "anger": ["3 mukhi", "mars", "agni"],
    "education": ["4 mukhi", "mercury", "focus", "clarity"],
    "exam": ["4 mukhi", "mercury", "focus", "clarity"],
    "exam stress": ["4 mukhi", "mercury", "focus", "clarity"],
    "wealth": ["5 mukhi", "lakshmi", "prosperity"],
    "sade sati": ["7 mukhi", "saturn", "shani"],
    "rahu": ["8 mukhi", "rahu"],
    "ketu": ["9 mukhi", "ketu", "spiritual"],
    "health": ["5 mukhi", "mala", "healing"],
    "negative energy": ["8 mukhi", "protection", "bracelet"],
    "stress": ["2 mukhi", "calming", "moon"],
    "sleep": ["2 mukhi", "calming", "moon"],
    "delayed marriage": ["3 mukhi", "mars", "mangal"],
    "mangal dosha": ["3 mukhi", "mars", "mangal"],
}


def _tokenize(text: str) -> set[str]:
    """Lowercase tokenize for matching."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _score_product(
    product: dict[str, Any],
    query_tokens: set[str],
    query_lower: str,
    concern_boosts: tuple[list[str], list[str]],
) -> float:
    """Score a product against a search query. Higher = better match."""
    name = (product.get("name") or "").lower()
    slug = (product.get("slug") or "").lower()
    category = ((product.get("category") or {}).get("name") or "").lower()
    description = (product.get("short_description") or product.get("description") or "").lower()

    # Combine all searchable text
    searchable = f"{name} {slug} {category} {description}"
    name_tokens = _tokenize(name)

    score = 0.0

    # 1. Exact query match in product name (strongest signal)
    if query_lower in name:
        score += 200

    # 2. Token overlap with product name
    overlap = query_tokens & name_tokens
    score += len(overlap) * 30

    # 3. Mukhi number match (very specific)
    mukhi_match = re.search(r"(\d+)\s*mukhi", query_lower)
    if mukhi_match:
        mukhi_num = mukhi_match.group(1)
        if re.search(rf"\b{mukhi_num}\s*mukhi", name):
            score += 150
        elif mukhi_num in name_tokens:
            score += 50

    # 4. Product type match (rudraksha, bracelet, mala, pendant)
    product_types = {"rudraksha", "bracelet", "mala", "pendant"}
    query_types = query_tokens & product_types
    name_types = name_tokens & product_types
    if query_types and query_types & name_types:
        score += 40

    # 5. Concern-based boost — if user asks "career rudraksha" and product
    #    name contains "5 mukhi" or "wisdom", boost it
    primary_boosts, secondary_boosts = concern_boosts
    for boost_term in primary_boosts:
        if boost_term in searchable:
            score += 40  # Strong boost for primary concern match
    for boost_term in secondary_boosts:
        if boost_term in searchable:
            score += 10  # Weaker boost for alias/secondary matches

    # 6. Category match
    if any(t in category for t in query_tokens):
        score += 15

    # 7. Slug match (slug often has clean product identifiers)
    slug_tokens = _tokenize(slug)
    slug_overlap = query_tokens & slug_tokens
    score += len(slug_overlap) * 10

    # 8. Description match (weaker signal)
    desc_overlap = query_tokens & _tokenize(description)
    score += len(desc_overlap) * 5

    # 9. Planet alias matching — "shani bracelet" should match "saturn" in name
    for planet, aliases in _PLANET_ALIASES.items():
        if query_tokens & aliases:
            planet_in_product = planet in searchable or any(a in searchable for a in aliases)
            if planet_in_product:
                score += 40

    return score


def _get_concern_boosts(query: str) -> tuple[list[str], list[str]]:
    """Get product name boost terms based on the user's concern.

    Returns (primary_boosts, secondary_boosts). Primary boosts get higher
    weight in scoring so "career stagnation" → 7 mukhi beats 5 mukhi.
    """
    query_lower = query.lower()
    primary: list[str] = []
    secondary: list[str] = []

    # Check longest concerns first (so "career stagnation" beats "career")
    for concern in sorted(_CONCERN_PRODUCT_BOOST, key=len, reverse=True):
        if concern in query_lower:
            primary.extend(_CONCERN_PRODUCT_BOOST[concern])
            break  # Only use the most specific concern match

    # Also check concern aliases (weaker signal)
    for concern_category, aliases in _CONCERN_ALIASES.items():
        if _tokenize(query) & aliases:
            category_boosts = _CONCERN_PRODUCT_BOOST.get(concern_category, [])
            for b in category_boosts:
                if b not in primary and b not in secondary:
                    secondary.append(b)

    return primary, secondary


class ProductCatalogIndex:
    """Caches the full product catalog and provides intelligent local matching.

    Instead of depending on core-service text search quality, this fetches
    ALL products and does smart scoring locally.
    """

    def __init__(self) -> None:
        self._products: list[dict[str, Any]] = []
        self._last_refresh: float = 0
        self._refreshing = False

    @property
    def is_loaded(self) -> bool:
        return len(self._products) > 0

    @property
    def product_count(self) -> int:
        return len(self._products)

    @property
    def needs_refresh(self) -> bool:
        return time.time() - self._last_refresh > CATALOG_REFRESH_INTERVAL

    async def refresh(self, core_service_client: Any) -> None:
        """Fetch all products from core-service using broad seed queries."""
        if self._refreshing:
            return
        self._refreshing = True
        try:
            all_products: dict[str, dict[str, Any]] = {}

            for query in _SEED_QUERIES:
                try:
                    results = await core_service_client.search_products(query, limit=50)
                    for product in results:
                        pid = str(product.get("id") or product.get("slug") or "")
                        if pid:
                            all_products[pid] = product
                except Exception as exc:
                    logger.warning("Catalog index fetch failed for '%s': %s", query, exc)

            if all_products:
                self._products = list(all_products.values())
                self._last_refresh = time.time()
                logger.info(
                    "Product catalog index refreshed: %d products",
                    len(self._products),
                )
            else:
                logger.warning("Product catalog index refresh returned 0 products")
        finally:
            self._refreshing = False

    async def ensure_fresh(self, core_service_client: Any) -> None:
        """Refresh the catalog if stale."""
        if self.needs_refresh:
            await self.refresh(core_service_client)

    def match_products(
        self,
        query: str,
        *,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        """Find best-matching products from the cached catalog.

        Uses token overlap, mukhi number matching, concern-based boosting,
        and planet alias matching to score products.
        """
        if not self._products:
            return []

        query_lower = query.lower().strip()
        query_tokens = _tokenize(query)
        concern_boosts = _get_concern_boosts(query)

        scored: list[tuple[float, dict[str, Any]]] = []
        for product in self._products:
            score = _score_product(product, query_tokens, query_lower, concern_boosts)
            if score > 0:
                scored.append((score, product))

        scored.sort(key=lambda x: x[0], reverse=True)

        if scored:
            logger.debug(
                "Catalog index matched '%s' → top: %s (score=%.1f), total_matches=%d",
                query,
                (scored[0][1].get("name") or "?")[:40],
                scored[0][0],
                len(scored),
            )

        return [product for _, product in scored[:limit]]


# Singleton instance — shared across the app
catalog_index = ProductCatalogIndex()
