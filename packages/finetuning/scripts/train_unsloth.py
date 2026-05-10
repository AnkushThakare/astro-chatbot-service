from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import DEFAULT_BASE_MODEL, format_chat_example, load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a small instruct model with Unsloth QLoRA on astrology examples."
    )
    parser.add_argument(
        "--model_name",
        default=DEFAULT_BASE_MODEL,
        help="Base model name.",
    )
    parser.add_argument(
        "--train_file",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "train.jsonl",
        help="Path to train.jsonl",
    )
    parser.add_argument(
        "--eval_file",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "eval.jsonl",
        help="Path to eval.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "astro-lora-v1",
        help="Where the trained LoRA adapter will be saved.",
    )
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument(
        "--save_merged_16bit",
        action="store_true",
        help="Also save a merged 16-bit model for downstream inference testing.",
    )
    return parser.parse_args()


def check_environment() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is not installed. Install packages/finetuning/requirements.txt first."
        ) from exc

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA / GPU is unavailable. This beginner QLoRA script expects an NVIDIA GPU. "
            "Use Colab, Kaggle, or another CUDA-enabled environment."
        )

    try:
        from datasets import Dataset
        from transformers import TrainingArguments
        from trl import SFTTrainer
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise SystemExit(
            "Training dependencies are missing. Install packages/finetuning/requirements.txt."
        ) from exc

    return torch, Dataset, TrainingArguments, SFTTrainer, FastLanguageModel
def build_dataset(dataset_cls: Any, tokenizer: Any, rows: list[dict[str, Any]]) -> Any:
    formatted = [{"text": format_chat_example(tokenizer, row, include_output=True)} for row in rows]
    return dataset_cls.from_list(formatted)


def main() -> int:
    args = parse_args()
    torch, dataset_cls, training_args_cls, trainer_cls, fast_model_cls = check_environment()

    train_rows = load_jsonl(args.train_file)
    eval_rows = load_jsonl(args.eval_file)

    print(f"Loading base model: {args.model_name}")
    model, tokenizer = fast_model_cls.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    model = fast_model_cls.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        use_rslora=False,
        loftq_config=None,
    )

    train_dataset = build_dataset(dataset_cls, tokenizer, train_rows)
    eval_dataset = build_dataset(dataset_cls, tokenizer, eval_rows)

    training_args = training_args_cls(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=1,
        save_strategy="epoch",
        eval_strategy="epoch",
        report_to="none",
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        warmup_ratio=0.05,
        seed=3407,
    )

    trainer = trainer_cls(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,
        args=training_args,
    )

    print("Starting supervised fine-tuning...")
    trainer.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"Saved LoRA adapter to: {args.output_dir}")

    if args.save_merged_16bit:
        merged_dir = args.output_dir / "merged-16bit"
        print(f"Saving merged 16-bit model to: {merged_dir}")
        model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
