from src.core.persona import build_persona_prompt, build_user_profile_summary


def test_build_persona_prompt_separates_knowledge_and_policy_sections() -> None:
    prompt = build_persona_prompt(
        long_term_context=(
            "User prefers brief answers.\n"
            "- language_preference: hinglish\n"
            "- detail_preference: brief\n"
            "- guidance_mode: practical\n"
            "- birth_details_status: complete"
        ),
        retrieval_context="- Saturn in the 10th house can slow visible career gains.",
        retrieval_policy_context="- Do not invent catalog products when no product results are available.",
        tool_context="Tool: recommend_product\nNo product results were found.",
    )

    assert "User memory (from previous conversations):\nUser prefers brief answers." in prompt
    assert "User personalization snapshot:" in prompt
    assert "Preferred reply language usually leans Hinglish." in prompt
    assert "Retrieved knowledge (IMPORTANT" in prompt
    assert "- Saturn in the 10th house can slow visible career gains." in prompt
    assert "Retrieved policy (use these guidelines to shape product/service mentions):" in prompt
    assert "- Do not invent catalog products when no product results are available." in prompt
    assert "Tool context:\nTool: recommend_product\nNo product results were found." in prompt


def test_build_persona_prompt_omits_policy_section_when_absent() -> None:
    prompt = build_persona_prompt(
        long_term_context=None,
        retrieval_context="No retrieved astrology notes were matched.",
        retrieval_policy_context=None,
        tool_context="No tool output used.",
    )

    assert "No retrieved astrology notes were matched." in prompt
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


def test_build_persona_prompt_includes_behavior_section_when_present() -> None:
    prompt = build_persona_prompt(
        long_term_context=None,
        retrieval_context="No retrieved astrology notes were matched.",
        retrieval_policy_context=None,
        tool_context="No tool output used.",
        behavior_summary="Energy flow snapshot:\n- Emotional state: elevated_stress\n- Behavioral state: overthinking_loop",
    )

    assert "Behavioral energy flow read" in prompt
    assert "overthinking_loop" in prompt


def test_build_user_profile_summary_combines_memory_session_and_behavior() -> None:
    summary = build_user_profile_summary(
        long_term_context=(
            "- language_preference: hinglish\n"
            "- detail_preference: detailed\n"
            "- guidance_mode: reassuring\n"
            "- last_concern: career\n"
            "- birth_details_status: complete"
        ),
        session_state={
            "main_concern": "career growth",
            "last_user_goal": "job switch guidance",
            "last_tool_summary": "Saturn mahadasha with 10th house pressure.",
        },
        behavior_summary=(
            "Energy flow snapshot:\n"
            "- Emotional state: elevated_stress\n"
            "- Focus state: scattered_focus\n"
            "- Behavioral state: overthinking_loop"
        ),
        response_language="hinglish",
        birth_details_available=True,
    )

    assert summary is not None
    assert "Reply language to use now: hinglish." in summary
    assert "Preferred answer depth: detailed." in summary
    assert "Recurring user concern: career growth." in summary
    assert "Chart personalization is available right now." in summary
    assert "Live behavioral pattern: overthinking_loop." in summary
