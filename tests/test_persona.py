from src.core.persona import build_persona_prompt


def test_build_persona_prompt_separates_knowledge_and_policy_sections() -> None:
    prompt = build_persona_prompt(
        long_term_context="User prefers brief answers.",
        retrieval_context="- Saturn in the 10th house can slow visible career gains.",
        retrieval_policy_context="- Do not invent catalog products when no product results are available.",
        tool_context="Tool: recommend_product\nNo product results were found.",
    )

    assert "User memory (from previous conversations):\nUser prefers brief answers." in prompt
    assert (
        "Retrieved knowledge:\n- Saturn in the 10th house can slow visible career gains."
        in prompt
    )
    assert (
        "Retrieved policy:\n- Do not invent catalog products when no product results are available."
        in prompt
    )
    assert "Tool context:\nTool: recommend_product\nNo product results were found." in prompt


def test_build_persona_prompt_omits_policy_section_when_absent() -> None:
    prompt = build_persona_prompt(
        long_term_context=None,
        retrieval_context="No retrieved astrology notes were matched.",
        retrieval_policy_context=None,
        tool_context="No tool output used.",
    )

    assert "Retrieved knowledge:\nNo retrieved astrology notes were matched." in prompt
    assert "Retrieved policy:" not in prompt


def test_build_persona_prompt_includes_pattern_section_when_present() -> None:
    prompt = build_persona_prompt(
        long_term_context=None,
        retrieval_context="No retrieved astrology notes were matched.",
        retrieval_policy_context=None,
        tool_context="No tool output used.",
        pattern_summary="Recurring theme: career\nWhat keeps repeating: The user keeps circling back to career pressure.",
    )

    assert "Recurring personal pattern read" in prompt
    assert "Recurring theme: career" in prompt
