from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run the full beginner fine-tuning workflow: validate, train, test, eval, compare."
    )
    parser.add_argument(
        "--train_file",
        type=Path,
        default=root / "data" / "train.jsonl",
        help="Path to train.jsonl",
    )
    parser.add_argument(
        "--eval_file",
        type=Path,
        default=root / "data" / "eval.jsonl",
        help="Path to eval.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=root / "outputs" / "astro-lora-v1",
        help="Where the trained adapter will be saved.",
    )
    parser.add_argument(
        "--model_name",
        default="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        help="Base model name.",
    )
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--skip_compare",
        action="store_true",
        help="Skip the base-vs-adapter comparison step.",
    )
    parser.add_argument(
        "--save_merged_16bit",
        action="store_true",
        help="Also save a merged 16-bit model during training.",
    )
    return parser.parse_args()


def run_step(title: str, command: list[str]) -> None:
    print(f"\n=== {title} ===")
    print(" ".join(command))
    subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    scripts_dir = Path(__file__).resolve().parent

    validate_cmd = [
        sys.executable,
        str(scripts_dir / "validate_dataset.py"),
        "--train",
        str(args.train_file),
        "--eval",
        str(args.eval_file),
    ]
    train_cmd = [
        sys.executable,
        str(scripts_dir / "train_unsloth.py"),
        "--model_name",
        args.model_name,
        "--train_file",
        str(args.train_file),
        "--eval_file",
        str(args.eval_file),
        "--output_dir",
        str(args.output_dir),
        "--max_seq_length",
        str(args.max_seq_length),
        "--epochs",
        str(args.epochs),
        "--learning_rate",
        str(args.learning_rate),
        "--batch_size",
        str(args.batch_size),
        "--grad_accum",
        str(args.grad_accum),
    ]
    if args.save_merged_16bit:
        train_cmd.append("--save_merged_16bit")

    test_cmd = [
        sys.executable,
        str(scripts_dir / "test_adapter.py"),
        "--adapter_dir",
        str(args.output_dir),
        "--base_model",
        args.model_name,
        "--max_seq_length",
        str(args.max_seq_length),
        "--max_new_tokens",
        str(args.max_new_tokens),
    ]
    eval_cmd = [
        sys.executable,
        str(scripts_dir / "evaluate_adapter.py"),
        "--eval_file",
        str(args.eval_file),
        "--adapter_dir",
        str(args.output_dir),
        "--base_model",
        args.model_name,
        "--max_seq_length",
        str(args.max_seq_length),
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
    ]
    compare_cmd = [
        sys.executable,
        str(scripts_dir / "compare_models.py"),
        "--eval_file",
        str(args.eval_file),
        "--adapter_dir",
        str(args.output_dir),
        "--base_model",
        args.model_name,
        "--max_seq_length",
        str(args.max_seq_length),
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
    ]

    run_step("Validate Dataset", validate_cmd)
    run_step("Train Adapter", train_cmd)
    run_step("Test Adapter", test_cmd)
    run_step("Evaluate Adapter", eval_cmd)
    if not args.skip_compare:
        run_step("Compare Base vs Adapter", compare_cmd)

    print("\nFull fine-tuning workflow completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
