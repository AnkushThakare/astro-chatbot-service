from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from src.core.chat_service import ChatService
from src.core.planner import PlannerResult


class ProductRecommendationEvalExample(BaseModel):
    id: str = Field(min_length=1)
    message: str = Field(min_length=1)
    action: str = Field(default="respond_only")
    kundali_summary: str | None = None
    policy_allows_product: bool = True
    expected_should_recommend: bool
    expected_query: str | None = None


@dataclass
class ProductRecommendationEvalPrediction:
    example: ProductRecommendationEvalExample
    decision: dict[str, object]


class ProductRecommendationEvalSummary(BaseModel):
    total_examples: int
    recommendation_accuracy: float
    recommendation_precision: float
    recommendation_recall: float
    query_match_rate: float


def load_examples(dataset_path: Path) -> list[ProductRecommendationEvalExample]:
    examples: list[ProductRecommendationEvalExample] = []
    for line_number, raw_line in enumerate(dataset_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parsed = json.loads(line)
        try:
            examples.append(ProductRecommendationEvalExample.model_validate(parsed))
        except Exception as exc:
            raise ValueError(f"Invalid example at line {line_number}: {exc}") from exc
    return examples


def _policy_matches(policy_allows_product: bool) -> list[dict[str, object]]:
    if not policy_allows_product:
        return [{"metadata": {"domain": "general_guidance", "allowed_actions": ["explain_only"]}}]
    return [{"metadata": {"domain": "product_policy", "allowed_actions": ["recommend_product"]}}]


def evaluate_examples(
    examples: list[ProductRecommendationEvalExample],
) -> list[ProductRecommendationEvalPrediction]:
    predictions: list[ProductRecommendationEvalPrediction] = []
    for example in examples:
        plan = PlannerResult(
            action=example.action,  # type: ignore[arg-type]
            confidence=0.9,
            arguments={},
            missing_information=[],
            should_call_tool=example.action == "recommend_product",
            reasoning="offline eval",
        )
        decision = ChatService._soft_product_decision(
            message=example.message,
            plan=plan,
            retrieval_policy_matches=_policy_matches(example.policy_allows_product),
            kundali_summary=example.kundali_summary,
        )
        predictions.append(ProductRecommendationEvalPrediction(example=example, decision=decision))
    return predictions


def compute_summary(
    predictions: list[ProductRecommendationEvalPrediction],
) -> ProductRecommendationEvalSummary:
    total = len(predictions)
    if total == 0:
        raise ValueError("Cannot compute metrics for an empty evaluation set")

    correct = 0
    predicted_positive = 0
    expected_positive = 0
    true_positive = 0
    query_matches = 0
    query_population = 0

    for prediction in predictions:
        expected = prediction.example.expected_should_recommend
        observed = bool(prediction.decision.get("allowed"))
        if observed == expected:
            correct += 1
        if observed:
            predicted_positive += 1
        if expected:
            expected_positive += 1
        if observed and expected:
            true_positive += 1

        expected_query = prediction.example.expected_query
        if expected_query:
            query_population += 1
            observed_query = prediction.decision.get("query")
            if observed_query == expected_query:
                query_matches += 1

    return ProductRecommendationEvalSummary(
        total_examples=total,
        recommendation_accuracy=correct / total,
        recommendation_precision=(
            true_positive / predicted_positive if predicted_positive else 0.0
        ),
        recommendation_recall=(true_positive / expected_positive if expected_positive else 0.0),
        query_match_rate=(query_matches / query_population if query_population else 0.0),
    )


def render_text_report(
    summary: ProductRecommendationEvalSummary,
    predictions: list[ProductRecommendationEvalPrediction],
) -> str:
    lines = [
        "Soft Product Recommendation Evaluation Summary",
        f"Total examples: {summary.total_examples}",
        f"Recommendation accuracy: {summary.recommendation_accuracy:.3f}",
        f"Recommendation precision: {summary.recommendation_precision:.3f}",
        f"Recommendation recall: {summary.recommendation_recall:.3f}",
        f"Query match rate: {summary.query_match_rate:.3f}",
        "",
        "Per-example decisions:",
    ]
    for prediction in predictions:
        lines.append(
            (
                f"- {prediction.example.id}: allowed={prediction.decision.get('allowed')}, "
                f"reason={prediction.decision.get('reason')}, query={prediction.decision.get('query')}"
            )
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate soft product recommendation decisions.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/product_recommendation_eval_examples.jsonl"),
        help="Path to the labeled soft product recommendation evaluation dataset.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write the summary JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = load_examples(args.dataset)
    predictions = evaluate_examples(examples)
    summary = compute_summary(predictions)
    print(render_text_report(summary, predictions))
    if args.output_json is not None:
        args.output_json.write_text(summary.model_dump_json(indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
