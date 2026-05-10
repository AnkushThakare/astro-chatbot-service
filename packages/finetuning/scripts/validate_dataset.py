from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REQUIRED_KEYS = ("instruction", "input", "output")
SHORT_OUTPUT_THRESHOLD = 240


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate train/eval JSONL files for astrology QLoRA fine-tuning."
    )
    parser.add_argument("--train", required=True, type=Path, help="Path to train.jsonl")
    parser.add_argument("--eval", required=True, type=Path, help="Path to eval.jsonl")
    return parser.parse_args()


def is_valid_value(value: Any) -> tuple[bool, str | None]:
    if isinstance(value, str):
        if value.strip():
            return True, None
        return False, "must be a non-empty string"

    if value is None:
        return False, "must not be null"

    try:
        json.dumps(value, ensure_ascii=False)
    except TypeError:
        return False, "must be a non-empty string or JSON-serializable object"

    if isinstance(value, (list, dict)) and not value:
        return False, "must not be an empty object or list"

    return True, None


def normalize_row(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def validate_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    stats: dict[str, Any] = {
        "path": path,
        "total_rows": 0,
        "valid_rows": 0,
        "invalid_rows": 0,
        "duplicate_count": 0,
        "warnings": [],
        "errors": [],
    }
    seen_rows: Counter[str] = Counter()

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            stats["total_rows"] += 1

            if not line:
                stats["invalid_rows"] += 1
                stats["errors"].append(f"{path}:{line_number}: empty lines are not allowed")
                continue

            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                stats["invalid_rows"] += 1
                stats["errors"].append(f"{path}:{line_number}: invalid JSON ({exc.msg})")
                continue

            if not isinstance(parsed, dict):
                stats["invalid_rows"] += 1
                stats["errors"].append(f"{path}:{line_number}: each row must be a JSON object")
                continue

            row_is_valid = True
            for key in REQUIRED_KEYS:
                if key not in parsed:
                    row_is_valid = False
                    stats["errors"].append(f"{path}:{line_number}: missing required key '{key}'")
                    continue

                ok, reason = is_valid_value(parsed[key])
                if not ok:
                    row_is_valid = False
                    stats["errors"].append(f"{path}:{line_number}: key '{key}' {reason}")

            if not row_is_valid:
                stats["invalid_rows"] += 1
                continue

            normalized = normalize_row(parsed)
            seen_rows[normalized] += 1
            if seen_rows[normalized] > 1:
                stats["duplicate_count"] += 1

            output_value = parsed["output"]
            output_text = output_value if isinstance(output_value, str) else json.dumps(output_value)
            if len(output_text.strip()) < SHORT_OUTPUT_THRESHOLD:
                stats["warnings"].append(
                    f"{path}:{line_number}: output looks short ({len(output_text.strip())} chars)"
                )

            stats["valid_rows"] += 1

    return stats


def print_report(stats: dict[str, Any]) -> None:
    print(f"Dataset: {stats['path']}")
    print(f"  Total rows     : {stats['total_rows']}")
    print(f"  Valid rows     : {stats['valid_rows']}")
    print(f"  Invalid rows   : {stats['invalid_rows']}")
    print(f"  Duplicate count: {stats['duplicate_count']}")
    print(f"  Warnings       : {len(stats['warnings'])}")

    for warning in stats["warnings"]:
        print(f"    WARNING: {warning}")
    for error in stats["errors"]:
        print(f"    ERROR: {error}")


def main() -> int:
    args = parse_args()
    train_stats = validate_file(args.train)
    eval_stats = validate_file(args.eval)

    print_report(train_stats)
    print_report(eval_stats)

    total_rows = train_stats["total_rows"] + eval_stats["total_rows"]
    valid_rows = train_stats["valid_rows"] + eval_stats["valid_rows"]
    duplicate_count = train_stats["duplicate_count"] + eval_stats["duplicate_count"]
    warning_count = len(train_stats["warnings"]) + len(eval_stats["warnings"])
    invalid_rows = train_stats["invalid_rows"] + eval_stats["invalid_rows"]

    print("\nSummary")
    print(f"  Total rows     : {total_rows}")
    print(f"  Valid rows     : {valid_rows}")
    print(f"  Duplicate count: {duplicate_count}")
    print(f"  Warnings       : {warning_count}")

    if invalid_rows:
        print(f"  Invalid rows   : {invalid_rows}")
        print("Validation failed because invalid rows were found.", file=sys.stderr)
        return 1

    print("Validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
