from __future__ import annotations


def suggest_consultant(query: str, kundali_summary: str | None = None) -> dict[str, str]:
    lowered = query.lower()
    if "career" in lowered or "job" in lowered:
        summary = "Suggested consultant profile: Vedic astrologer focused on career timing and transit-based planning."
    elif "marriage" in lowered or "relationship" in lowered:
        summary = "Suggested consultant profile: compatibility and relationship astrologer experienced with matchmaking."
    else:
        summary = "Suggested consultant profile: general Vedic astrologer for a full-life consultation."

    if kundali_summary:
        summary += f" Kundali context considered: {kundali_summary}"

    return {"tool": "suggest_consultant", "summary": summary}

