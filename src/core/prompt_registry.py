from __future__ import annotations

import hashlib
from pathlib import Path

from src.core.config import settings

PROMPT_VERSIONS = {
    "persona_v1": "v1.4.0",
    "planner": "v1.2.0",
    "emotion_detector": "v1.0.0",
    "memory_extractor": "v1.1.0",
}


def _prompt_path(prompt_name: str) -> Path:
    return settings.prompts_dir / f"{prompt_name}.txt"


def get_prompt_version_hash(prompt_name: str) -> str:
    prompt_path = _prompt_path(prompt_name)
    try:
        content = prompt_path.read_text(encoding="utf-8")
    except OSError:
        return "missing"
    return hashlib.sha1(content.encode("utf-8")).hexdigest()[:8]


def current_prompt_metadata() -> dict[str, str]:
    return {
        "persona": PROMPT_VERSIONS["persona_v1"],
        "planner": PROMPT_VERSIONS["planner"],
        "emotion_detector": PROMPT_VERSIONS["emotion_detector"],
        "memory_extractor": PROMPT_VERSIONS["memory_extractor"],
        "content_hash": get_prompt_version_hash("persona_v1"),
    }
