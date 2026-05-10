from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from common import DEFAULT_BASE_MODEL, format_chat_example, load_jsonl, score_interpretation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare base model vs LoRA adapter on held-out astrology examples."
    )
    parser.add_argument(
        "--eval_file",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "eval.jsonl",
        help="Path to eval.jsonl",
    )
    parser.add_argument(
        "--adapter_dir",
        type=Path,
        required=True,
        help="Directory containing the saved LoRA adapter.",
    )
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--report_file",
        type=Path,
        default=None,
        help="Optional markdown report path.",
    )
    return parser.parse_args()


def check_environment() -> tuple[Any, Any, Any]:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is not installed. Install requirements.txt first.") from exc

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA / GPU is unavailable. This comparison script expects an NVIDIA GPU."
        )

    try:
        from peft import PeftModel
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise SystemExit(
            "Missing inference dependencies. Install packages/finetuning/requirements.txt."
        ) from exc

    return torch, PeftModel, FastLanguageModel


def build_report_path(args: argparse.Namespace) -> Path:
    if args.report_file is not None:
        return args.report_file
    run_dir = Path(__file__).resolve().parents[1] / "outputs" / "comparison_runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return run_dir / f"{timestamp}-base-vs-adapter.md"


def generate_one(
    *,
    torch: Any,
    tokenizer: Any,
    model: Any,
    example: dict[str, Any],
    max_new_tokens: int,
    temperature: float,
) -> str:
    prompt_text = format_chat_example(tokenizer, example, include_output=False)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )
    output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    if output_text.startswith(prompt_text):
        output_text = output_text[len(prompt_text) :].strip()
    return output_text.strip()


def main() -> int:
    args = parse_args()
    if not args.adapter_dir.exists():
        raise SystemExit(f"Adapter directory not found: {args.adapter_dir}")

    torch, peft_model_cls, fast_model_cls = check_environment()
    rows = load_jsonl(args.eval_file)
    if args.limit > 0:
        rows = rows[: args.limit]

    print(f"Loading base model: {args.base_model}")
    model, tokenizer = fast_model_cls.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    fast_model_cls.for_inference(model)

    base_outputs: list[str] = []
    for row in rows:
        base_outputs.append(
            generate_one(
                torch=torch,
                tokenizer=tokenizer,
                model=model,
                example=row,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
        )

    print(f"Applying adapter: {args.adapter_dir}")
    adapted_model = peft_model_cls.from_pretrained(model, str(args.adapter_dir))
    fast_model_cls.for_inference(adapted_model)

    adapter_outputs: list[str] = []
    for row in rows:
        adapter_outputs.append(
            generate_one(
                torch=torch,
                tokenizer=tokenizer,
                model=adapted_model,
                example=row,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
        )

    report_file = build_report_path(args)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with report_file.open("w", encoding="utf-8") as handle:
        handle.write("# Base vs Adapter Comparison\n\n")
        handle.write(f"- Base model: `{args.base_model}`\n")
        handle.write(f"- Adapter dir: `{args.adapter_dir}`\n")
        handle.write(f"- Eval file: `{args.eval_file}`\n\n")

        for index, row in enumerate(rows, start=1):
            base_output = base_outputs[index - 1]
            adapter_output = adapter_outputs[index - 1]
            base_score = score_interpretation(base_output)
            adapter_score = score_interpretation(adapter_output)

            handle.write(f"## Example {index}: {row['instruction']}\n\n")
            handle.write("### Input\n\n")
            handle.write("```json\n")
            handle.write(f"{row['input']}\n")
            handle.write("```\n\n")
            handle.write("### Reference\n\n")
            handle.write(f"{row['output']}\n\n")
            handle.write("### Base Output\n\n")
            handle.write(f"{base_output}\n\n")
            handle.write("### Adapter Output\n\n")
            handle.write(f"{adapter_output}\n\n")
            handle.write("### Heuristic Snapshot\n\n")
            handle.write(
                f"- Base sections: {base_score['section_count']}/6, "
                f"disclaimer={base_score['disclaimer_present']}, "
                f"confidence={base_score['confidence_present']}\n"
            )
            handle.write(
                f"- Adapter sections: {adapter_score['section_count']}/6, "
                f"disclaimer={adapter_score['disclaimer_present']}, "
                f"confidence={adapter_score['confidence_present']}\n\n"
            )

    print(f"Saved comparison report to: {report_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
