from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.core.rag import RAGService
from src.db.session import configure_database, get_db


class RetrievalEvalExample(BaseModel):
    id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    action: str | None = None
    planner_query: str | None = None
    chart_context: dict[str, Any] | None = None
    top_k: int = Field(default=3, ge=1, le=10)
    expected_knowledge_domains: list[str] = Field(default_factory=list)
    expected_policy_domains: list[str] = Field(default_factory=list)


@dataclass
class RetrievalEvalPrediction:
    example: RetrievalEvalExample
    payload: dict[str, Any]


class RetrievalEvalSummary(BaseModel):
    total_examples: int
    knowledge_domain_recall: float
    policy_domain_recall: float
    full_bundle_match_rate: float
    any_expected_match_rate: float
    reranker_provider: str | None = None
    reranker_model: str | None = None


class RetrievalEvalComparisonSummary(BaseModel):
    primary_reranker_provider: str
    challenger_reranker_provider: str
    primary_full_bundle_match_rate: float
    challenger_full_bundle_match_rate: float
    primary_any_expected_match_rate: float
    challenger_any_expected_match_rate: float
    primary_knowledge_domain_recall: float
    challenger_knowledge_domain_recall: float
    primary_policy_domain_recall: float
    challenger_policy_domain_recall: float
    full_bundle_delta: float
    any_expected_delta: float
    knowledge_domain_delta: float
    policy_domain_delta: float
    changed_top_source_count: int


def load_examples(dataset_path: Path) -> list[RetrievalEvalExample]:
    examples: list[RetrievalEvalExample] = []
    for line_number, raw_line in enumerate(dataset_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parsed = json.loads(line)
        try:
            examples.append(RetrievalEvalExample.model_validate(parsed))
        except Exception as exc:
            raise ValueError(f"Invalid example at line {line_number}: {exc}") from exc
    return examples


def _observed_domains(matches: list[dict[str, Any]]) -> set[str]:
    domains: set[str] = set()
    for match in matches:
        metadata = match.get("metadata") or {}
        domain = metadata.get("domain")
        if isinstance(domain, str) and domain:
            domains.add(domain)
    return domains


def _all_matches(payload: dict[str, Any]) -> list[dict[str, Any]]:
    combined = list(payload.get("chunks") or [])
    if combined:
        return combined
    return [*list(payload.get("knowledge_chunks") or []), *list(payload.get("policy_chunks") or [])]


def _top_sources(payload: dict[str, Any], limit: int = 3) -> list[str]:
    sources: list[str] = []
    seen: set[str] = set()
    for match in _all_matches(payload):
        source = match.get("source") or match.get("title")
        if not isinstance(source, str) or not source.strip():
            continue
        normalized = source.strip()
        if normalized in seen:
            continue
        seen.add(normalized)
        sources.append(normalized)
        if len(sources) >= limit:
            break
    return sources


def _chart_context_hit(example: RetrievalEvalExample, payload: dict[str, Any]) -> bool:
    if not example.chart_context:
        return False
    for match in _all_matches(payload):
        metadata = match.get("metadata") or {}
        raw_score = metadata.get("chart_score")
        if isinstance(raw_score, (int, float)) and float(raw_score) > 0:
            return True
    return False


def _example_match_flags(prediction: RetrievalEvalPrediction) -> tuple[bool, bool, bool, bool]:
    example = prediction.example
    payload = prediction.payload
    observed_knowledge = _observed_domains(list(payload.get("knowledge_chunks") or []))
    observed_policy = _observed_domains(list(payload.get("policy_chunks") or []))
    expected_knowledge = set(example.expected_knowledge_domains)
    expected_policy = set(example.expected_policy_domains)

    knowledge_ok = not expected_knowledge or expected_knowledge.issubset(observed_knowledge)
    policy_ok = not expected_policy or expected_policy.issubset(observed_policy)
    full_ok = knowledge_ok and policy_ok
    any_ok = (
        bool(expected_knowledge and knowledge_ok)
        or bool(expected_policy and policy_ok)
        or (not expected_knowledge and not expected_policy)
    )
    return knowledge_ok, policy_ok, full_ok, any_ok


def compute_summary(predictions: list[RetrievalEvalPrediction]) -> RetrievalEvalSummary:
    total = len(predictions)
    if total == 0:
        raise ValueError("Cannot compute metrics for an empty evaluation set")

    knowledge_population = 0
    knowledge_hits = 0
    policy_population = 0
    policy_hits = 0
    full_matches = 0
    any_matches = 0

    for prediction in predictions:
        example = prediction.example
        payload = prediction.payload
        observed_knowledge = _observed_domains(list(payload.get("knowledge_chunks") or []))
        observed_policy = _observed_domains(list(payload.get("policy_chunks") or []))
        expected_knowledge = set(example.expected_knowledge_domains)
        expected_policy = set(example.expected_policy_domains)

        knowledge_ok = True
        policy_ok = True
        any_ok = False

        if expected_knowledge:
            knowledge_population += 1
            knowledge_ok = expected_knowledge.issubset(observed_knowledge)
            if knowledge_ok:
                knowledge_hits += 1
                any_ok = True

        if expected_policy:
            policy_population += 1
            policy_ok = expected_policy.issubset(observed_policy)
            if policy_ok:
                policy_hits += 1
                any_ok = True

        if knowledge_ok and policy_ok:
            full_matches += 1
        if any_ok or (not expected_knowledge and not expected_policy):
            any_matches += 1

    reranker_provider: str | None = None
    reranker_model: str | None = None
    for prediction in predictions:
        retrieval_metadata = prediction.payload.get("retrieval_metadata") or {}
        provider = retrieval_metadata.get("reranker_provider")
        model = retrieval_metadata.get("reranker_model")
        if isinstance(provider, str) and provider and reranker_provider is None:
            reranker_provider = provider
        if isinstance(model, str) and model and reranker_model is None:
            reranker_model = model

    return RetrievalEvalSummary(
        total_examples=total,
        knowledge_domain_recall=(
            knowledge_hits / knowledge_population if knowledge_population else 0.0
        ),
        policy_domain_recall=(policy_hits / policy_population if policy_population else 0.0),
        full_bundle_match_rate=full_matches / total,
        any_expected_match_rate=any_matches / total,
        reranker_provider=reranker_provider,
        reranker_model=reranker_model,
    )


def compare_summaries(
    primary_summary: RetrievalEvalSummary,
    challenger_summary: RetrievalEvalSummary,
    primary_predictions: list[RetrievalEvalPrediction],
    challenger_predictions: list[RetrievalEvalPrediction],
) -> RetrievalEvalComparisonSummary:
    primary_by_id = {prediction.example.id: prediction for prediction in primary_predictions}
    challenger_by_id = {prediction.example.id: prediction for prediction in challenger_predictions}
    common_ids = sorted(set(primary_by_id) & set(challenger_by_id))
    changed_top_source_count = 0
    for example_id in common_ids:
        if _top_sources(primary_by_id[example_id].payload, limit=1) != _top_sources(
            challenger_by_id[example_id].payload,
            limit=1,
        ):
            changed_top_source_count += 1

    return RetrievalEvalComparisonSummary(
        primary_reranker_provider=primary_summary.reranker_provider or "unknown",
        challenger_reranker_provider=challenger_summary.reranker_provider or "unknown",
        primary_full_bundle_match_rate=primary_summary.full_bundle_match_rate,
        challenger_full_bundle_match_rate=challenger_summary.full_bundle_match_rate,
        primary_any_expected_match_rate=primary_summary.any_expected_match_rate,
        challenger_any_expected_match_rate=challenger_summary.any_expected_match_rate,
        primary_knowledge_domain_recall=primary_summary.knowledge_domain_recall,
        challenger_knowledge_domain_recall=challenger_summary.knowledge_domain_recall,
        primary_policy_domain_recall=primary_summary.policy_domain_recall,
        challenger_policy_domain_recall=challenger_summary.policy_domain_recall,
        full_bundle_delta=challenger_summary.full_bundle_match_rate - primary_summary.full_bundle_match_rate,
        any_expected_delta=challenger_summary.any_expected_match_rate - primary_summary.any_expected_match_rate,
        knowledge_domain_delta=challenger_summary.knowledge_domain_recall - primary_summary.knowledge_domain_recall,
        policy_domain_delta=challenger_summary.policy_domain_recall - primary_summary.policy_domain_recall,
        changed_top_source_count=changed_top_source_count,
    )


def evaluate_examples(
    examples: list[RetrievalEvalExample],
    rag_service: RAGService,
) -> list[RetrievalEvalPrediction]:
    predictions: list[RetrievalEvalPrediction] = []
    for example in examples:
        payload = rag_service.retrieve_context_bundle(
            example.query,
            example.top_k,
            action=example.action,
            planner_query=example.planner_query,
            chart_context=example.chart_context,
        )
        predictions.append(RetrievalEvalPrediction(example=example, payload=payload))
    return predictions


def render_text_report(
    summary: RetrievalEvalSummary,
    predictions: list[RetrievalEvalPrediction],
) -> str:
    lines = [
        "Retrieval Evaluation Summary",
        f"Total examples: {summary.total_examples}",
        f"Knowledge-domain recall: {summary.knowledge_domain_recall:.3f}",
        f"Policy-domain recall: {summary.policy_domain_recall:.3f}",
        f"Full bundle match rate: {summary.full_bundle_match_rate:.3f}",
        f"Any expected match rate: {summary.any_expected_match_rate:.3f}",
        f"Reranker provider: {summary.reranker_provider or 'unknown'}",
        f"Reranker model: {summary.reranker_model or 'unknown'}",
        "",
        "Per-example results:",
    ]
    for prediction in predictions:
        payload = prediction.payload
        top_sources = _top_sources(payload)
        chart_context_hit = _chart_context_hit(prediction.example, payload)
        knowledge_ok, policy_ok, full_ok, any_ok = _example_match_flags(prediction)
        lines.append(
            (
                f"- {prediction.example.id}: "
                f"knowledge={sorted(_observed_domains(list(payload.get('knowledge_chunks') or [])))}, "
                f"policy={sorted(_observed_domains(list(payload.get('policy_chunks') or [])))}, "
                f"provider={payload.get('retrieval_metadata', {}).get('provider')}, "
                f"reranker={payload.get('retrieval_metadata', {}).get('reranker_provider')}, "
                f"chart_context_used={payload.get('retrieval_metadata', {}).get('chart_context_used')}, "
                f"chart_context_hit={chart_context_hit}, "
                f"top_sources={top_sources}, "
                f"knowledge_ok={knowledge_ok}, policy_ok={policy_ok}, full_ok={full_ok}, any_ok={any_ok}"
            )
        )
    return "\n".join(lines)


def render_comparison_report(
    comparison: RetrievalEvalComparisonSummary,
    primary_predictions: list[RetrievalEvalPrediction],
    challenger_predictions: list[RetrievalEvalPrediction],
) -> str:
    primary_by_id = {prediction.example.id: prediction for prediction in primary_predictions}
    challenger_by_id = {prediction.example.id: prediction for prediction in challenger_predictions}
    lines = [
        "Retrieval Reranker Comparison",
        f"Primary reranker: {comparison.primary_reranker_provider}",
        f"Challenger reranker: {comparison.challenger_reranker_provider}",
        f"Full bundle delta: {comparison.full_bundle_delta:+.3f}",
        f"Any expected delta: {comparison.any_expected_delta:+.3f}",
        f"Knowledge recall delta: {comparison.knowledge_domain_delta:+.3f}",
        f"Policy recall delta: {comparison.policy_domain_delta:+.3f}",
        f"Changed top source count: {comparison.changed_top_source_count}",
        "",
        "Per-example top-source changes:",
    ]
    for example_id in sorted(set(primary_by_id) & set(challenger_by_id)):
        primary_sources = _top_sources(primary_by_id[example_id].payload)
        challenger_sources = _top_sources(challenger_by_id[example_id].payload)
        if primary_sources != challenger_sources:
            lines.append(
                f"- {example_id}: primary={primary_sources} challenger={challenger_sources}"
            )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval on labeled examples.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/retrieval_eval_examples.jsonl"),
        help="Path to the labeled retrieval evaluation dataset.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write the summary JSON.",
    )
    parser.add_argument(
        "--output-predictions-json",
        type=Path,
        default=None,
        help="Optional path to write per-example predictions JSON.",
    )
    parser.add_argument(
        "--reranker-provider",
        type=str,
        default=None,
        help="Optional reranker provider override, for example heuristic or groq_listwise.",
    )
    parser.add_argument(
        "--reranker-model",
        type=str,
        default=None,
        help="Optional reranker model override.",
    )
    parser.add_argument(
        "--compare-rerankers",
        type=str,
        default=None,
        help="Optional comma-separated reranker providers to compare, for example heuristic,groq_listwise.",
    )
    return parser.parse_args()


def _build_rag_service_for_eval(
    db: Any,
    *,
    reranker_provider: str | None = None,
    reranker_model: str | None = None,
) -> RAGService:
    rag_service = RAGService(db)
    if reranker_provider:
        rag_service.settings.RAG_RERANKER_PROVIDER = reranker_provider
        if reranker_model:
            rag_service.settings.RAG_RERANKER_MODEL = reranker_model
        from src.core.reranker import get_reranker_provider

        rag_service.reranker = get_reranker_provider(rag_service.settings)
    return rag_service


def main() -> None:
    args = parse_args()
    examples = load_examples(args.dataset)

    configure_database()
    db = next(get_db())
    try:
        if args.compare_rerankers:
            compare_targets = [target.strip() for target in args.compare_rerankers.split(",") if target.strip()]
            if len(compare_targets) != 2:
                raise ValueError("--compare-rerankers expects exactly two providers separated by a comma")

            primary_predictions = evaluate_examples(
                examples,
                _build_rag_service_for_eval(
                    db,
                    reranker_provider=compare_targets[0],
                    reranker_model=args.reranker_model,
                ),
            )
            challenger_predictions = evaluate_examples(
                examples,
                _build_rag_service_for_eval(
                    db,
                    reranker_provider=compare_targets[1],
                    reranker_model=args.reranker_model,
                ),
            )
            primary_summary = compute_summary(primary_predictions)
            challenger_summary = compute_summary(challenger_predictions)
            comparison = compare_summaries(
                primary_summary,
                challenger_summary,
                primary_predictions,
                challenger_predictions,
            )

            print(render_text_report(primary_summary, primary_predictions))
            print()
            print(render_text_report(challenger_summary, challenger_predictions))
            print()
            print(render_comparison_report(comparison, primary_predictions, challenger_predictions))
            return

        rag_service = _build_rag_service_for_eval(
            db,
            reranker_provider=args.reranker_provider,
            reranker_model=args.reranker_model,
        )
        predictions = evaluate_examples(examples, rag_service)
    finally:
        db.close()

    summary = compute_summary(predictions)
    print(render_text_report(summary, predictions))

    if args.output_json is not None:
        args.output_json.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    if args.output_predictions_json is not None:
        payload = [
            {
                "id": prediction.example.id,
                "query": prediction.example.query,
                "payload": prediction.payload,
            }
            for prediction in predictions
        ]
        args.output_predictions_json.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
