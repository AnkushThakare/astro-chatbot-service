from __future__ import annotations

from pathlib import Path

from finetune.product_recommendation_eval import (
    ProductRecommendationEvalExample,
    ProductRecommendationEvalPrediction,
    compute_summary,
    load_examples,
)


def test_load_examples_parses_product_recommendation_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "product_recommendation_eval.jsonl"
    dataset.write_text(
        (
            '{"id":"soft-1","message":"career delay","action":"respond_only",'
            '"policy_allows_product":true,"expected_should_recommend":true,'
            '"expected_query":"rudraksha career"}\n'
        ),
        encoding="utf-8",
    )

    examples = load_examples(dataset)

    assert len(examples) == 1
    assert examples[0].id == "soft-1"
    assert examples[0].expected_query == "rudraksha career"


def test_compute_summary_tracks_soft_recommendation_metrics() -> None:
    examples = [
        ProductRecommendationEvalExample(
            id="yes_hit",
            message="career delay",
            expected_should_recommend=True,
            expected_query="rudraksha career",
        ),
        ProductRecommendationEvalExample(
            id="no_hit",
            message="one practical step only",
            expected_should_recommend=False,
        ),
        ProductRecommendationEvalExample(
            id="query_miss",
            message="need peace",
            expected_should_recommend=True,
            expected_query="rudraksha peace",
        ),
    ]

    predictions = [
        ProductRecommendationEvalPrediction(
            example=examples[0],
            decision={"allowed": True, "reason": "policy_and_context_match", "query": "rudraksha career"},
        ),
        ProductRecommendationEvalPrediction(
            example=examples[1],
            decision={"allowed": False, "reason": "single_step_requested", "query": None},
        ),
        ProductRecommendationEvalPrediction(
            example=examples[2],
            decision={"allowed": True, "reason": "policy_and_context_match", "query": "bracelet protection"},
        ),
    ]

    summary = compute_summary(predictions)

    assert summary.total_examples == 3
    assert summary.recommendation_accuracy == 1.0
    assert summary.recommendation_precision == 1.0
    assert summary.recommendation_recall == 1.0
    assert summary.query_match_rate == 0.5
