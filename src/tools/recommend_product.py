from __future__ import annotations


def recommend_product(query: str, kundali_summary: str | None = None) -> dict[str, str]:
    lowered = query.lower()
    if "saturn" in lowered or "shani" in lowered:
        summary = (
            "Suggested products: blue sapphire guidance note, black sesame donation kit, "
            "and a Saturday discipline journal."
        )
    elif "love" in lowered or "relationship" in lowered:
        summary = (
            "Suggested products: rose quartz bracelet, Venus mantra card set, "
            "and a relationship reflection workbook."
        )
    else:
        summary = (
            "Suggested products: daily astrology planner, rudraksha mala, "
            "and a beginner kundali interpretation guide."
        )

    if kundali_summary:
        summary += f" Kundali context considered: {kundali_summary}"

    return {"tool": "recommend_product", "summary": summary}

