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
) -> str:
    sections = [load_persona_prompt()]
    if long_term_context:
        sections.append("Long-term memory:\n" + long_term_context)
    sections.append("Retrieved context:\n" + retrieval_context)
    sections.append("Tool context:\n" + tool_context)
    return "\n\n".join(sections)
