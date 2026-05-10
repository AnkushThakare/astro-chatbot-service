from __future__ import annotations

ALLOWED_PRODUCT_CATEGORIES = {
    "rudraksha",
    "rudraksha_mala",
    "bracelet",
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
