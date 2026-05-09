from src.core.persona import load_persona_prompt


def test_persona_prompt_is_not_empty() -> None:
    assert load_persona_prompt().strip()
