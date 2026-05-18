from __future__ import annotations

from pathlib import Path

from finetune.retrieval_eval import (
    RetrievalEvalExample,
    RetrievalEvalPrediction,
    compare_summaries,
    compute_summary,
    evaluate_examples,
    load_examples,
    render_comparison_report,
    render_text_report,
)


def test_load_examples_parses_retrieval_jsonl_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "retrieval_eval.jsonl"
    dataset.write_text(
        (
            '{"id":"ex1","query":"career delay","action":"respond_only",'
            '"chart_context":{"current_mahadasha":"Saturn"},'
            '"expected_knowledge_domains":["general_guidance"]}\n'
        ),
        encoding="utf-8",
    )

    examples = load_examples(dataset)

    assert len(examples) == 1
    assert examples[0].id == "ex1"
    assert examples[0].chart_context == {"current_mahadasha": "Saturn"}
    assert examples[0].expected_knowledge_domains == ["general_guidance"]


def test_compute_summary_tracks_knowledge_and_policy_recall() -> None:
    examples = [
        RetrievalEvalExample(
            id="knowledge_hit",
            query="career delay",
            expected_knowledge_domains=["general_guidance"],
        ),
        RetrievalEvalExample(
            id="policy_hit",
            query="rudraksha focus",
            action="recommend_product",
            expected_policy_domains=["product_policy"],
        ),
        RetrievalEvalExample(
            id="miss",
            query="book puja",
            action="book_pooja",
            expected_policy_domains=["booking_guidance"],
        ),
    ]

    predictions = [
        RetrievalEvalPrediction(
            example=examples[0],
            payload={"knowledge_chunks": [{"metadata": {"domain": "general_guidance"}}], "policy_chunks": []},
        ),
        RetrievalEvalPrediction(
            example=examples[1],
            payload={"knowledge_chunks": [], "policy_chunks": [{"metadata": {"domain": "product_policy"}}]},
        ),
        RetrievalEvalPrediction(
            example=examples[2],
            payload={"knowledge_chunks": [], "policy_chunks": [{"metadata": {"domain": "remedy_guidance"}}]},
        ),
    ]

    summary = compute_summary(predictions)

    assert summary.total_examples == 3
    assert summary.knowledge_domain_recall == 1.0
    assert summary.policy_domain_recall == 0.5
    assert summary.full_bundle_match_rate == 2 / 3
    assert summary.any_expected_match_rate == 2 / 3


def test_compute_summary_tracks_reranker_metadata() -> None:
    example = RetrievalEvalExample(id="ex1", query="career delay")
    predictions = [
        RetrievalEvalPrediction(
            example=example,
            payload={
                "knowledge_chunks": [{"metadata": {"domain": "general_guidance"}}],
                "policy_chunks": [],
                "retrieval_metadata": {
                    "reranker_provider": "heuristic",
                    "reranker_model": "heuristic-v1",
                },
            },
        )
    ]

    summary = compute_summary(predictions)

    assert summary.reranker_provider == "heuristic"
    assert summary.reranker_model == "heuristic-v1"


def test_evaluate_examples_passes_chart_context_into_rag() -> None:
    examples = [
        RetrievalEvalExample(
            id="ex1",
            query="job change soon",
            chart_context={"current_mahadasha": "Saturn"},
        )
    ]
    captured: dict[str, object] = {}

    class _RagStub:
        def retrieve_context_bundle(self, *args, **kwargs):  # noqa: ANN202
            captured["kwargs"] = kwargs
            return {"knowledge_chunks": [], "policy_chunks": [], "retrieval_metadata": {}}

    evaluate_examples(examples, _RagStub())  # type: ignore[arg-type]

    assert captured["kwargs"]["chart_context"] == {"current_mahadasha": "Saturn"}


def test_render_text_report_includes_reranker_and_chart_flags() -> None:
    example = RetrievalEvalExample(id="ex1", query="career delay")
    predictions = [
        RetrievalEvalPrediction(
            example=example,
            payload={
                "chunks": [
                    {
                        "source": "Saturn Career Note",
                        "metadata": {"domain": "general_guidance", "chart_score": 6},
                    }
                ],
                "knowledge_chunks": [{"metadata": {"domain": "general_guidance", "chart_score": 6}}],
                "policy_chunks": [],
                "retrieval_metadata": {
                    "provider": "embedding_store",
                    "reranker_provider": "heuristic",
                    "chart_context_used": True,
                },
            },
        )
    ]
    summary = compute_summary(predictions)

    report = render_text_report(summary, predictions)

    assert "Reranker provider: heuristic" in report
    assert "chart_context_used=True" in report
    assert "chart_context_hit=False" in report
    assert "top_sources=['Saturn Career Note']" in report


def test_render_text_report_marks_chart_context_hit_when_example_and_scores_exist() -> None:
    example = RetrievalEvalExample(
        id="ex1",
        query="career delay",
        chart_context={"current_mahadasha": "Saturn"},
    )
    predictions = [
        RetrievalEvalPrediction(
            example=example,
            payload={
                "chunks": [
                    {
                        "source": "Saturn Career Note",
                        "metadata": {"domain": "general_guidance", "chart_score": 6},
                    }
                ],
                "knowledge_chunks": [{"metadata": {"domain": "general_guidance", "chart_score": 6}}],
                "policy_chunks": [],
                "retrieval_metadata": {"reranker_provider": "heuristic", "chart_context_used": True},
            },
        )
    ]

    report = render_text_report(compute_summary(predictions), predictions)

    assert "chart_context_hit=True" in report


def test_compare_summaries_tracks_top_source_changes() -> None:
    example = RetrievalEvalExample(
        id="ex1",
        query="career delay",
        expected_knowledge_domains=["general_guidance"],
    )
    primary_predictions = [
        RetrievalEvalPrediction(
            example=example,
            payload={
                "chunks": [{"source": "General Guidance", "metadata": {"domain": "general_guidance"}}],
                "knowledge_chunks": [{"metadata": {"domain": "general_guidance"}}],
                "policy_chunks": [],
                "retrieval_metadata": {"reranker_provider": "heuristic"},
            },
        )
    ]
    challenger_predictions = [
        RetrievalEvalPrediction(
            example=example,
            payload={
                "chunks": [{"source": "Saturn Career Note", "metadata": {"domain": "general_guidance"}}],
                "knowledge_chunks": [{"metadata": {"domain": "general_guidance"}}],
                "policy_chunks": [],
                "retrieval_metadata": {"reranker_provider": "groq_listwise"},
            },
        )
    ]

    primary_summary = compute_summary(primary_predictions)
    challenger_summary = compute_summary(challenger_predictions)
    comparison = compare_summaries(
        primary_summary,
        challenger_summary,
        primary_predictions,
        challenger_predictions,
    )

    assert comparison.primary_reranker_provider == "heuristic"
    assert comparison.challenger_reranker_provider == "groq_listwise"
    assert comparison.changed_top_source_count == 1

    report = render_comparison_report(comparison, primary_predictions, challenger_predictions)

    assert "Changed top source count: 1" in report
    assert "primary=['General Guidance'] challenger=['Saturn Career Note']" in report
