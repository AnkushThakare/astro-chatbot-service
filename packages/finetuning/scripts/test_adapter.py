from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from common import DEFAULT_BASE_MODEL, build_messages

SAMPLE_PROMPTS = [
    {
        "name": "career",
        "instruction": "Analyze career from this Vedic chart.",
        "input": {
            "lagna": "Capricorn",
            "moon_sign": "Taurus",
            "houses": {
                "1": {"sign": "Capricorn", "planets": ["Saturn"]},
                "10": {"sign": "Libra", "planets": ["Mercury", "Sun"]},
            },
        },
    },
    {
        "name": "marriage",
        "instruction": "Analyze marriage and relationship dynamics from this Vedic chart.",
        "input": {
            "lagna": "Cancer",
            "moon_sign": "Libra",
            "houses": {
                "5": {"sign": "Scorpio", "planets": ["Venus"]},
                "7": {"sign": "Capricorn", "planets": ["Mars"]},
            },
        },
    },
    {
        "name": "spirituality",
        "instruction": "Analyze spirituality and inner growth from this Vedic chart.",
        "input": {
            "lagna": "Pisces",
            "moon_sign": "Cancer",
            "houses": {
                "9": {"sign": "Scorpio", "planets": ["Ketu"]},
                "12": {"sign": "Aquarius", "planets": ["Jupiter", "Venus"]},
            },
        },
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a base model + LoRA adapter and print sample astrology outputs."
    )
    parser.add_argument(
        "--adapter_dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "astro-lora-v1",
        help="Directory containing the saved LoRA adapter.",
    )
    parser.add_argument(
        "--base_model",
        default=DEFAULT_BASE_MODEL,
        help="Base model name used for inference before applying the adapter.",
    )
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=350)
    return parser.parse_args()


def check_environment() -> tuple[Any, Any, Any]:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is not installed. Install requirements.txt first.") from exc

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA / GPU is unavailable. This simple adapter test script expects an NVIDIA GPU."
        )

    try:
        from peft import PeftModel
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise SystemExit(
            "Missing inference dependencies. Install packages/finetuning/requirements.txt."
        ) from exc

    return torch, PeftModel, FastLanguageModel
def main() -> int:
    args = parse_args()
    torch, peft_model_cls, fast_model_cls = check_environment()

    if not args.adapter_dir.exists():
        raise SystemExit(f"Adapter directory not found: {args.adapter_dir}")

    print(f"Loading base model: {args.base_model}")
    model, tokenizer = fast_model_cls.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    model = peft_model_cls.from_pretrained(model, str(args.adapter_dir))
    fast_model_cls.for_inference(model)

    for sample in SAMPLE_PROMPTS:
        prompt_text = tokenizer.apply_chat_template(
            build_messages(sample, include_output=False),
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        start = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=0.7,
                do_sample=True,
            )
        latency_ms = (time.perf_counter() - start) * 1000

        output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        print("=" * 80)
        print(f"Sample: {sample['name']}")
        print(f"Latency: {latency_ms:.2f} ms")
        print(output_text)
        print()

    print("Adapter test run complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
