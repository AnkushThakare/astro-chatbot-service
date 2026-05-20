from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.core.chat_service import ChatService
from src.core.config import settings
from src.core.llm import GroqClient
from src.core.planner import ConversationPlanner, PlannerAction, PlannerResult


class PlannerEvalExample(BaseModel):
    id: str = Field(min_length=1)
    message: str = Field(min_length=1)
    expected_action: PlannerAction
    expected_should_call_tool: bool
    expected_arguments: dict[str, Any] = Field(default_factory=dict)
    expected_missing_information: list[str] = Field(default_factory=list)
    has_birth_details: bool = False
    has_matchmaking_details: bool = False
    is_authenticated: bool = False
    unsafe_to_call_tool: bool = False


@dataclass
class PlannerEvalPrediction:
    example: PlannerEvalExample
    plan: PlannerResult
    guarded_tool_call: bool


class PlannerEvalSummary(BaseModel):
    total_examples: int
    action_accuracy: float
    guarded_tool_precision: float
    guarded_tool_recall: float
    unsafe_tool_false_positive_rate: float
    clarification_rate: float
    argument_accuracy: float
    missing_information_accuracy: float
    raw_tool_call_rate: float
    guarded_tool_call_rate: float


def load_examples(dataset_path: Path) -> list[PlannerEvalExample]:
    examples: list[PlannerEvalExample] = []
    for line_number, raw_line in enumerate(dataset_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parsed = json.loads(line)
        try:
            examples.append(PlannerEvalExample.model_validate(parsed))
        except Exception as exc:
            raise ValueError(f"Invalid example at line {line_number}: {exc}") from exc
    return examples


def _subset_arguments_match(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    for key, expected_value in expected.items():
        if actual.get(key) != expected_value:
            return False
    return True


def compute_summary(predictions: list[PlannerEvalPrediction]) -> PlannerEvalSummary:
    total = len(predictions)
    if total == 0:
        raise ValueError("Cannot compute metrics for an empty evaluation set")

    action_correct = 0
    tool_tp = 0
    tool_fp = 0
    tool_fn = 0
    unsafe_population = 0
    unsafe_false_positives = 0
    clarification_count = 0
    argument_population = 0
    argument_correct = 0
    missing_population = 0
    missing_correct = 0
    raw_tool_call_count = 0
    guarded_tool_call_count = 0

    for prediction in predictions:
        example = prediction.example
        plan = prediction.plan

        if plan.action == example.expected_action:
            action_correct += 1
        if plan.action == "ask_clarification":
            clarification_count += 1
        if plan.should_call_tool:
            raw_tool_call_count += 1
        if prediction.guarded_tool_call:
            guarded_tool_call_count += 1

        if example.expected_should_call_tool and prediction.guarded_tool_call:
            tool_tp += 1
        elif not example.expected_should_call_tool and prediction.guarded_tool_call:
            tool_fp += 1
        elif example.expected_should_call_tool and not prediction.guarded_tool_call:
            tool_fn += 1

        if example.unsafe_to_call_tool:
            unsafe_population += 1
            if prediction.guarded_tool_call:
                unsafe_false_positives += 1

        if example.expected_arguments:
            argument_population += 1
            if _subset_arguments_match(example.expected_arguments, plan.arguments):
                argument_correct += 1

        if example.expected_missing_information:
            missing_population += 1
            if set(example.expected_missing_information) == set(plan.missing_information):
                missing_correct += 1

    precision_denominator = tool_tp + tool_fp
    recall_denominator = tool_tp + tool_fn

    return PlannerEvalSummary(
        total_examples=total,
        action_accuracy=action_correct / total,
        guarded_tool_precision=(tool_tp / precision_denominator) if precision_denominator else 0.0,
        guarded_tool_recall=(tool_tp / recall_denominator) if recall_denominator else 0.0,
        unsafe_tool_false_positive_rate=(
            unsafe_false_positives / unsafe_population if unsafe_population else 0.0
        ),
        clarification_rate=clarification_count / total,
        argument_accuracy=(argument_correct / argument_population) if argument_population else 0.0,
        missing_information_accuracy=(
            missing_correct / missing_population if missing_population else 0.0
        ),
        raw_tool_call_rate=raw_tool_call_count / total,
        guarded_tool_call_rate=guarded_tool_call_count / total,
    )


async def evaluate_examples(
    examples: list[PlannerEvalExample],
    planner: ConversationPlanner,
    *,
    sleep_seconds: float,
) -> list[PlannerEvalPrediction]:
    predictions: list[PlannerEvalPrediction] = []
    for example in examples:
        plan = await planner.plan(
            message=example.message,
            has_birth_details=example.has_birth_details,
            has_matchmaking_details=example.has_matchmaking_details,
            is_authenticated=example.is_authenticated,
        )
        guarded_tool_call = ChatService._should_execute_tool(
            plan,
            birth_details={} if example.has_birth_details else None,
            matchmaking_details={} if example.has_matchmaking_details else None,
        )
        predictions.append(
            PlannerEvalPrediction(
                example=example,
                plan=plan,
                guarded_tool_call=guarded_tool_call,
            )
        )
        await asyncio.sleep(sleep_seconds)
    return predictions


def render_text_report(
    summary: PlannerEvalSummary,
    predictions: list[PlannerEvalPrediction],
) -> str:
    lines = [
        "Planner Evaluation Summary",
        f"Total examples: {summary.total_examples}",
        f"Action accuracy: {summary.action_accuracy:.3f}",
        f"Guarded tool precision: {summary.guarded_tool_precision:.3f}",
        f"Guarded tool recall: {summary.guarded_tool_recall:.3f}",
        f"Unsafe tool false positive rate: {summary.unsafe_tool_false_positive_rate:.3f}",
        f"Clarification rate: {summary.clarification_rate:.3f}",
        f"Argument accuracy: {summary.argument_accuracy:.3f}",
        f"Missing-information accuracy: {summary.missing_information_accuracy:.3f}",
        f"Raw tool call rate: {summary.raw_tool_call_rate:.3f}",
        f"Guarded tool call rate: {summary.guarded_tool_call_rate:.3f}",
        "",
        "Per-example results:",
    ]
    for prediction in predictions:
        example = prediction.example
        plan = prediction.plan
        lines.append(
            (
                f"- {example.id}: expected_action={example.expected_action}, "
                f"actual_action={plan.action}, "
                f"expected_tool={str(example.expected_should_call_tool).lower()}, "
                f"guarded_tool={str(prediction.guarded_tool_call).lower()}, "
                f"confidence={plan.confidence:.2f}"
            )
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate planner accuracy on labeled examples.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/planner_eval_examples.jsonl"),
        help="Path to the labeled planner evaluation dataset.",
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
        "--offset",
        type=int,
        default=0,
        help="Start evaluation from this zero-based example offset.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate at most this many examples.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.25,
        help="Delay between live planner calls to reduce rate limiting.",
    )
    return parser.parse_args()


def _planner_settings_for_eval() -> tuple[Any, str]:
    eval_mode = settings.EVAL_MODE or os.getenv("EVAL_MODE") == "1"
    run_id = f"planner-eval-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    os.environ["EVAL_RUN_ID"] = run_id
    if not eval_mode:
        return settings, run_id

    eval_api_key = settings.GROQ_API_KEY_EVAL or settings.GROQ_API_KEY
    return settings.model_copy(
        update={
            "EVAL_MODE": True,
            "GROQ_API_KEY": eval_api_key,
        }
    ), run_id


async def _main() -> int:
    args = parse_args()
    examples = load_examples(args.dataset)
    if args.offset < 0:
        raise ValueError("--offset must be >= 0")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be > 0 when provided")
    if args.offset:
        examples = examples[args.offset :]
    if args.limit is not None:
        examples = examples[: args.limit]
    eval_settings, run_id = _planner_settings_for_eval()
    planner = ConversationPlanner(GroqClient(eval_settings), eval_settings.GROQ_PLANNER_MODEL)
    if not eval_settings.GROQ_API_KEY:
        print(
            "Warning: GROQ_API_KEY is not configured. The planner will fall back to "
            "respond_only, so accuracy metrics will not reflect live model behavior."
        )
    elif eval_settings.EVAL_MODE:
        print(f"Running planner eval in isolated mode with run_id={run_id}")
    predictions = await evaluate_examples(examples, planner, sleep_seconds=args.sleep_seconds)
    summary = compute_summary(predictions)

    print(render_text_report(summary, predictions))

    if args.output_json is not None:
        args.output_json.write_text(summary.model_dump_json(indent=2), encoding="utf-8")

    if args.output_predictions_json is not None:
        payload = [
            {
                "example": prediction.example.model_dump(mode="json"),
                "plan": prediction.plan.model_dump(mode="json"),
                "guarded_tool_call": prediction.guarded_tool_call,
            }
            for prediction in predictions
        ]
        args.output_predictions_json.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
