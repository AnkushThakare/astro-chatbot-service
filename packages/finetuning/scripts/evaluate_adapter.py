from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from common import DEFAULT_BASE_MODEL, format_chat_example, load_jsonl, score_interpretation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a held-out evaluation pass with a base model or LoRA adapter."
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
        default=None,
        help="Optional LoRA adapter directory. Leave unset to evaluate the base model only.",
    )
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=0, help="Optional max rows to evaluate.")
    parser.add_argument(
        "--output_file",
        type=Path,
        default=None,
        help="Optional JSONL file for saving row-by-row predictions.",
    )
    return parser.parse_args()


def check_environment() -> tuple[Any, Any, Any]:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is not installed. Install requirements.txt first.") from exc

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA / GPU is unavailable. This evaluation script expects an NVIDIA GPU."
        )

    try:
        from peft import PeftModel
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise SystemExit(
            "Missing inference dependencies. Install packages/finetuning/requirements.txt."
        ) from exc

    return torch, PeftModel, FastLanguageModel


def build_output_path(args: argparse.Namespace) -> Path:
    if args.output_file is not None:
        return args.output_file
    run_dir = Path(__file__).resolve().parents[1] / "outputs" / "eval_runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    mode = "adapter" if args.adapter_dir else "base"
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return run_dir / f"{timestamp}-{mode}.jsonl"


def load_model(args: argparse.Namespace, fast_model_cls: Any, peft_model_cls: Any) -> tuple[Any, Any]:
    print(f"Loading base model: {args.base_model}")
    model, tokenizer = fast_model_cls.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    if args.adapter_dir:
        if not args.adapter_dir.exists():
            raise SystemExit(f"Adapter directory not found: {args.adapter_dir}")
        print(f"Loading LoRA adapter: {args.adapter_dir}")
        model = peft_model_cls.from_pretrained(model, str(args.adapter_dir))
    fast_model_cls.for_inference(model)
    return model, tokenizer


def generate_one(
    *,
    torch: Any,
    tokenizer: Any,
    model: Any,
    example: dict[str, Any],
    max_new_tokens: int,
    temperature: float,
) -> tuple[str, float]:
    prompt_text = format_chat_example(tokenizer, example, include_output=False)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    start = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )
    latency_ms = (time.perf_counter() - start) * 1000

    output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    if output_text.startswith(prompt_text):
        output_text = output_text[len(prompt_text) :].strip()
    return output_text.strip(), latency_ms


def main() -> int:
    args = parse_args()
    torch, peft_model_cls, fast_model_cls = check_environment()
    rows = load_jsonl(args.eval_file)
    if args.limit > 0:
        rows = rows[: args.limit]

    output_file = build_output_path(args)
    model, tokenizer = load_model(args, fast_model_cls, peft_model_cls)

    total_latency_ms = 0.0
    section_complete_count = 0
    disclaimer_count = 0
    confidence_count = 0

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            prediction, latency_ms = generate_one(
                torch=torch,
                tokenizer=tokenizer,
                model=model,
                example=row,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            heuristics = score_interpretation(prediction)
            total_latency_ms += latency_ms
            section_complete_count += int(heuristics["all_sections_present"])
            disclaimer_count += int(heuristics["disclaimer_present"])
            confidence_count += int(heuristics["confidence_present"])

            result_row = {
                "index": index,
                "instruction": row["instruction"],
                "reference_output": row["output"],
                "prediction": prediction,
                "latency_ms": round(latency_ms, 2),
                "heuristics": heuristics,
            }
            handle.write(json.dumps(result_row, ensure_ascii=False) + "\n")
            print(
                f"[{index}/{len(rows)}] latency={latency_ms:.2f} ms "
                f"sections={heuristics['section_count']}/6"
            )

    count = max(len(rows), 1)
    print("\nEvaluation Summary")
    print(f"  Rows evaluated           : {len(rows)}")
    print(f"  Avg latency (ms)         : {total_latency_ms / count:.2f}")
    print(f"  All sections present     : {section_complete_count}/{len(rows)}")
    print(f"  Confidence section found : {confidence_count}/{len(rows)}")
    print(f"  Disclaimer found         : {disclaimer_count}/{len(rows)}")
    print(f"  Saved predictions        : {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
