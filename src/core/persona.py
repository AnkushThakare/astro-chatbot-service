from __future__ import annotations

from functools import lru_cache
import re

from src.core.config import settings


@lru_cache
def load_persona_prompt() -> str:
    persona_path = settings.prompts_dir / "persona_v1.txt"
    if persona_path.exists():
        loaded = persona_path.read_text(encoding="utf-8").strip()
        if loaded:
            return loaded
    return settings.DEFAULT_SYSTEM_PROMPT


def _parse_memory_facts(long_term_context: str | None) -> dict[str, str]:
    facts: dict[str, str] = {}
    for raw_line in (long_term_context or "").splitlines():
        line = raw_line.strip()
        if not line.startswith("- "):
            continue
        body = line[2:]
        if ":" not in body:
            continue
        key, value = body.split(":", 1)
        key = key.strip()
        value = re.sub(r"\s+", " ", value.strip())
        if key and value:
            facts[key] = value
    return facts


def _build_personalization_guidance(long_term_context: str | None) -> str | None:
    facts = _parse_memory_facts(long_term_context)
    if not facts:
        return None

    lines: list[str] = []
    language = facts.get("language_preference")
    if language == "hinglish":
        lines.append("- Preferred reply language usually leans Hinglish.")
    elif language == "english":
        lines.append("- Preferred reply language usually leans plain English.")

    detail = facts.get("detail_preference")
    if detail == "brief":
        lines.append("- User usually prefers brief, direct answers.")
    elif detail == "detailed":
        lines.append("- User usually prefers richer, more detailed explanations.")

    guidance_mode = facts.get("guidance_mode")
    if guidance_mode == "reassuring":
        lines.append("- User responds better to steady, calming guidance than blunt intensity.")
    elif guidance_mode == "devotional":
        lines.append("- User is open to devotional language, mantra, or pooja framing when relevant.")
    elif guidance_mode == "practical":
        lines.append("- User responds well to practical, usable next steps.")

    concern = facts.get("last_concern") or facts.get("life_area")
    if concern:
        lines.append(f"- Strongest recurring concern: {concern}.")

    if facts.get("birth_details_status") == "complete":
        lines.append("- Birth details are already available, so chart-grounded personalization is expected.")

    if not lines:
        return None

    lines.append(
        "- Personalization usage: match the user's usual language and answer depth unless the current message clearly asks for something different."
    )
    return "User personalization snapshot:\n" + "\n".join(lines)


def _extract_behavior_profile_summary(behavior_summary: str | None) -> list[str]:
    if not behavior_summary:
        return []
    lines: list[str] = []
    for raw_line in behavior_summary.splitlines():
        line = raw_line.strip()
        if line.startswith("- Emotional state: "):
            lines.append("- Live emotional state: " + line.removeprefix("- Emotional state: ").strip() + ".")
        elif line.startswith("- Behavioral state: "):
            lines.append("- Live behavioral pattern: " + line.removeprefix("- Behavioral state: ").strip() + ".")
        elif line.startswith("- Focus state: "):
            lines.append("- Live focus pattern: " + line.removeprefix("- Focus state: ").strip() + ".")
    return lines[:3]


def build_user_profile_summary(
    *,
    long_term_context: str | None,
    session_state: dict[str, object] | None,
    behavior_summary: str | None,
    response_language: str | None,
    birth_details_available: bool,
) -> str | None:
    facts = _parse_memory_facts(long_term_context)
    lines: list[str] = []

    language = response_language or facts.get("language_preference")
    if language:
        if language == "english":
            lines.append("- Reply language: English only. Do NOT mix Hindi/Hinglish words. Use Saturn not Shani, Hello not Pranam.")
        else:
            lines.append(f"- Reply language to use now: {language}.")

    detail_preference = facts.get("detail_preference")
    if detail_preference:
        lines.append(f"- Preferred answer depth: {detail_preference}.")

    guidance_mode = facts.get("guidance_mode")
    if guidance_mode:
        lines.append(f"- Guidance style that usually lands best: {guidance_mode}.")

    main_concern = None
    if isinstance(session_state, dict):
        raw_concern = session_state.get("main_concern")
        if isinstance(raw_concern, str) and raw_concern.strip():
            main_concern = raw_concern.strip()
    if not main_concern:
        main_concern = facts.get("last_concern") or facts.get("life_area")
    if main_concern:
        lines.append(f"- Recurring user concern: {main_concern}.")

    if birth_details_available or facts.get("birth_details_status") == "complete":
        lines.append("- Chart personalization is available right now.")
    else:
        lines.append("- Chart personalization is not available yet, so do not imply exact placements.")

    if isinstance(session_state, dict):
        last_user_goal = session_state.get("last_user_goal")
        if isinstance(last_user_goal, str) and last_user_goal.strip():
            lines.append(f"- Last known user goal: {last_user_goal.strip()}.")
        last_tool_summary = session_state.get("last_tool_summary")
        if isinstance(last_tool_summary, str) and last_tool_summary.strip():
            lines.append(f"- Prior chat context worth continuing from: {last_tool_summary.strip()}.")

    lines.extend(_extract_behavior_profile_summary(behavior_summary))
    if not lines:
        return None

    lines.append(
        "- Profile usage: use this to continue the relationship naturally; do not announce the whole profile back to the user."
    )
    return "Persistent user profile:\n" + "\n".join(lines)


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
    session_state: dict[str, object] | None = None,
    response_language: str | None = None,
    birth_details_available: bool = False,
    user_profile_summary: str | None = None,
) -> str:
    sections = [load_persona_prompt()]
    if long_term_context:
        sections.append(
            "User memory (from previous conversations):\n"
            + long_term_context
            + "\nMemory usage: Reference these facts naturally when relevant to the current question. "
            "If the user's concern matches a prior topic, acknowledge continuity briefly "
            "(e.g., 'You mentioned career concerns earlier'). "
            "Do NOT list all facts - only use what connects to the current message. "
            "Build on previous conversations rather than starting fresh."
        )
    personalization_guidance = _build_personalization_guidance(long_term_context)
    if personalization_guidance:
        sections.append(personalization_guidance)
    if user_profile_summary is None:
        user_profile_summary = build_user_profile_summary(
            long_term_context=long_term_context,
            session_state=session_state,
            behavior_summary=behavior_summary,
            response_language=response_language,
            birth_details_available=birth_details_available,
        )
    if user_profile_summary:
        sections.append(user_profile_summary)
    if chart_summary:
        sections.append(
            "User's birth chart (use naturally when relevant - weave placements and dasha into your response):\n"
            + chart_summary
        )
    if transit_summary:
        sections.append(
            "Live planetary transits (happening RIGHT NOW - mention proactively when relevant to the user's question):\n"
            + transit_summary
            + "\nTransit usage: A real pandit always checks current transits. "
            "Mention these naturally - e.g., 'With Saturn currently in your 7th house, "
            "relationship patience is key right now.' Do NOT list all transits - pick the 1-2 most relevant."
        )
    if prediction_summary:
        sections.append(
            "Predictive insights (share proactively like a caring pandit would):\n"
            + prediction_summary
            + "\nPrediction usage: A great astrologer doesn't just answer questions - they anticipate. "
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
    sections.append(
        "Retrieved knowledge (IMPORTANT - base your answer on this content, use specific details like planet names, "
        "house numbers, dasha periods, and remedy specifics from below rather than improvising generic advice):\n"
        + retrieval_context
    )
    if retrieval_policy_context:
        sections.append(
            "Retrieved policy (use these guidelines to shape product/service mentions):\n"
            + retrieval_policy_context
        )
    sections.append("Tool context:\n" + tool_context)
    return "\n\n".join(sections)
