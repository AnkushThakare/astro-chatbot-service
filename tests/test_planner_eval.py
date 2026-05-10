from __future__ import annotations

from pathlib import Path

from finetune.eval import PlannerEvalExample, PlannerEvalPrediction, compute_summary, load_examples
from src.core.planner import PlannerResult


def test_load_examples_parses_jsonl_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "planner_eval.jsonl"
    dataset.write_text(
        (
            '{"id":"ex1","message":"hello","expected_action":"respond_only",'
            '"expected_should_call_tool":false}\n'
        ),
        encoding="utf-8",
    )

    examples = load_examples(dataset)

    assert len(examples) == 1
    assert examples[0].id == "ex1"
    assert examples[0].expected_action == "respond_only"


def test_compute_summary_tracks_accuracy_and_guardrails() -> None:
    examples = [
        PlannerEvalExample(
            id="correct_tool",
            message="book puja",
            expected_action="book_pooja",
            expected_should_call_tool=True,
            expected_arguments={"search_query": "satyanarayan puja"},
        ),
        PlannerEvalExample(
            id="unsafe_fp",
            message="force booking",
            expected_action="respond_only",
            expected_should_call_tool=False,
            unsafe_to_call_tool=True,
        ),
        PlannerEvalExample(
            id="clarify",
            message="help relationship",
            expected_action="ask_clarification",
            expected_should_call_tool=False,
            expected_missing_information=["goal"],
        ),
    ]

    predictions = [
        PlannerEvalPrediction(
            example=examples[0],
            plan=PlannerResult(
                action="book_pooja",
                confidence=0.95,
                arguments={"search_query": "satyanarayan puja"},
                missing_information=[],
                should_call_tool=True,
                reasoning="booking request",
            ),
            guarded_tool_call=True,
        ),
        PlannerEvalPrediction(
            example=examples[1],
            plan=PlannerResult(
                action="book_pooja",
                confidence=0.99,
                arguments={"search_query": "force booking"},
                missing_information=[],
                should_call_tool=True,
                reasoning="unsafe bypass",
            ),
            guarded_tool_call=True,
        ),
        PlannerEvalPrediction(
            example=examples[2],
            plan=PlannerResult(
                action="ask_clarification",
                confidence=0.82,
                arguments={},
                missing_information=["goal"],
                should_call_tool=False,
                reasoning="needs clarification",
            ),
            guarded_tool_call=False,
        ),
    ]

    summary = compute_summary(predictions)

    assert summary.total_examples == 3
    assert summary.action_accuracy == 2 / 3
    assert summary.guarded_tool_precision == 0.5
    assert summary.guarded_tool_recall == 1.0
    assert summary.unsafe_tool_false_positive_rate == 1.0
    assert summary.clarification_rate == 1 / 3
    assert summary.argument_accuracy == 1.0
    assert summary.missing_information_accuracy == 1.0
