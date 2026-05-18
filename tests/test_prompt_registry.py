from pathlib import Path

from src.core import prompt_registry
from src.core.prompt_registry import current_prompt_metadata, get_prompt_version_hash


def test_prompt_registry_returns_current_metadata() -> None:
    metadata = current_prompt_metadata()

    assert metadata["persona"] == "v1.4.0"
    assert metadata["planner"] == "v1.2.0"
    assert len(metadata["content_hash"]) == 8 or metadata["content_hash"] == "missing"


def test_prompt_version_hash_returns_missing_for_unknown_prompt() -> None:
    assert get_prompt_version_hash("does_not_exist") == "missing"


def test_prompt_version_hash_changes_when_prompt_content_changes(tmp_path) -> None:
    prompt_file = tmp_path / "persona_v1.txt"
    prompt_file.write_text("first version", encoding="utf-8")

    original_prompt_path = prompt_registry._prompt_path

    try:
        prompt_registry._prompt_path = lambda prompt_name: Path(prompt_file)  # type: ignore[assignment]
        first_hash = get_prompt_version_hash("persona_v1")
        prompt_file.write_text("second version", encoding="utf-8")
        second_hash = get_prompt_version_hash("persona_v1")
    finally:
        prompt_registry._prompt_path = original_prompt_path  # type: ignore[assignment]

    assert first_hash != second_hash
