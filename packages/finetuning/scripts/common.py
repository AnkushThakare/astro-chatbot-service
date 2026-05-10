from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_BASE_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
SYSTEM_PROMPT = (
    "You are a careful Vedic astrology assistant. Use only the provided chart data. "
    "Do not claim certainty. Do not give medical, legal, or financial guarantees."
)
REQUIRED_SECTIONS = (
    "1. Direct Summary",
    "2. Chart Evidence",
    "3. Interpretation",
    "4. Practical Guidance",
    "5. Confidence Level",
    "6. Disclaimer",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Dataset file not found: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSON ({exc.msg})") from exc
            if not isinstance(parsed, dict):
                raise SystemExit(f"{path}:{line_number}: each row must be a JSON object")
            for key in ("instruction", "input", "output"):
                if key not in parsed:
                    raise SystemExit(f"{path}:{line_number}: missing required key '{key}'")
            rows.append(parsed)

    if not rows:
        raise SystemExit(f"No rows found in {path}")
    return rows


def normalize_input(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, indent=2)


def build_messages(example: dict[str, Any], *, include_output: bool) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"{str(example['instruction']).strip()}\n\n{normalize_input(example['input'])}",
        },
    ]
    if include_output:
        messages.append({"role": "assistant", "content": str(example["output"]).strip()})
    return messages


def format_chat_example(tokenizer: Any, example: dict[str, Any], *, include_output: bool) -> str:
    return tokenizer.apply_chat_template(
        build_messages(example, include_output=include_output),
        tokenize=False,
        add_generation_prompt=not include_output,
    )


def score_interpretation(text: str) -> dict[str, Any]:
    lowered = text.lower()
    present_sections = [section for section in REQUIRED_SECTIONS if section.lower() in lowered]
    disclaimer_present = "disclaimer" in lowered or "not" in lowered and "advice" in lowered
    confidence_present = "confidence level" in lowered
    guidance_present = "practical guidance" in lowered
    return {
        "section_count": len(present_sections),
        "all_sections_present": len(present_sections) == len(REQUIRED_SECTIONS),
        "missing_sections": [section for section in REQUIRED_SECTIONS if section not in present_sections],
        "disclaimer_present": disclaimer_present,
        "confidence_present": confidence_present,
        "guidance_present": guidance_present,
        "char_count": len(text.strip()),
        "word_count": len(text.split()),
    }
