from src.core.pattern_engine import analyze_personal_patterns, build_pattern_summary


def test_analyze_personal_patterns_detects_career_loop() -> None:
    analysis = analyze_personal_patterns(
        long_term_context=(
            "- last_concern: career delay\n"
            "- last_topic: career\n"
            "- emotion_trend: anxious"
        ),
        recent_messages=[
            {"role": "user", "content": "My work feels stuck again."},
            {"role": "assistant", "content": "Tell me more."},
            {"role": "user", "content": "I am worried about job timing and promotion."},
        ],
        transit_data={"summary": "Saturn in Pisces: career pressure and discipline."},
        predictions=[
            {
                "priority": "high",
                "title": "Career pressure pattern active",
                "insight": "The same work-pressure theme keeps resurfacing.",
                "actionable": "Focus on one disciplined move.",
            }
        ],
    )

    assert analysis["dominant_theme"] == "career"
    assert analysis["repeat_count"] >= 2
    assert analysis["confidence"] in {"medium", "high"}


def test_build_pattern_summary_returns_perceptive_prompt_block() -> None:
    summary = build_pattern_summary(
        long_term_context="- last_concern: relationship\n- last_topic: relationship",
        recent_messages=[{"role": "user", "content": "Relationship confusion is coming up again."}],
        transit_data={"summary": "Venus in Libra: relationship themes are activated."},
        predictions=[],
    )

    assert summary is not None
    assert "Pattern mirror" in summary
    assert "Recurring theme: relationship" in summary
