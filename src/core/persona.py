from __future__ import annotations

from functools import lru_cache

from src.core.config import settings


@lru_cache
def load_persona_prompt() -> str:
    persona_path = settings.prompts_dir / "persona_v1.txt"
    if persona_path.exists():
        loaded = persona_path.read_text(encoding="utf-8").strip()
        if loaded:
            return loaded
    return settings.DEFAULT_SYSTEM_PROMPT


def build_persona_prompt(
    long_term_context: str | None,
    retrieval_context: str,
    tool_context: str,
    retrieval_policy_context: str | None = None,
    chart_summary: str | None = None,
    transit_summary: str | None = None,
    prediction_summary: str | None = None,
    pattern_summary: str | None = None,
    behavior_summary: str | None = None,
) -> str:
    sections = [load_persona_prompt()]
    if long_term_context:
        sections.append(
            "User memory (from previous conversations):\n"
            + long_term_context
            + "\nMemory usage: Reference these facts naturally when relevant to the current question. "
            "If the user's concern matches a prior topic, acknowledge continuity briefly "
            "(e.g., 'You mentioned career concerns earlier'). "
            "Do NOT list all facts — only use what connects to the current message. "
            "Build on previous conversations rather than starting fresh."
        )
    if chart_summary:
        sections.append(
            "User's birth chart (use naturally when relevant — weave placements and dasha into your response):\n"
            + chart_summary
        )
    if transit_summary:
        sections.append(
            "Live planetary transits (happening RIGHT NOW — mention proactively when relevant to the user's question):\n"
            + transit_summary
            + "\nTransit usage: A real pandit always checks current transits. "
            "Mention these naturally — e.g., 'With Saturn currently in your 7th house, "
            "relationship patience is key right now.' Do NOT list all transits — pick the 1-2 most relevant."
        )
    if prediction_summary:
        sections.append(
            "Predictive insights (share proactively like a caring pandit would):\n"
            + prediction_summary
            + "\nPrediction usage: A great astrologer doesn't just answer questions — they anticipate. "
            "If the user asks about career and you see a dasha transition coming, mention it. "
            "If they seem worried and a benefic transit is active, reassure with specifics. "
            "Share at most 1-2 insights per response, woven naturally."
        )
    if pattern_summary:
        sections.append(
            "Recurring personal pattern read (use this to make the chatbot feel eerily specific in a grounded way):\n"
            + pattern_summary
        )
    if behavior_summary:
        sections.append(
            "Behavioral energy flow read (use this only when it sharpens relevance, never as therapy language):\n"
            + behavior_summary
        )
    sections.append("Retrieved knowledge:\n" + retrieval_context)
    if retrieval_policy_context:
        sections.append("Retrieved policy:\n" + retrieval_policy_context)
    sections.append("Tool context:\n" + tool_context)
    return "\n\n".join(sections)
