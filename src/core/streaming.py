from __future__ import annotations

import json
import re
from typing import Any


def sse_event(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


def chunk_text(text: str, chunk_size: int = 120) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return [""]

    sentence_parts = re.findall(r".+?(?:[.!?]+(?:\s+|$)|$)", normalized)
    chunks: list[str] = []

    for sentence in sentence_parts:
        has_trailing_space = sentence.endswith(" ")
        stripped_sentence = sentence.rstrip()
        if not stripped_sentence:
            continue
        if len(stripped_sentence) <= chunk_size:
            chunks.append(stripped_sentence + (" " if has_trailing_space else ""))
            continue

        clause_parts = re.split(r"(?<=[,;:])\s+", stripped_sentence)
        current = ""
        for clause in clause_parts:
            clause = clause.strip()
            if not clause:
                continue
            candidate = clause if not current else f"{current} {clause}"
            if len(candidate) <= chunk_size:
                current = candidate
                continue
            if current:
                chunks.append(current)
            if len(clause) <= chunk_size:
                current = clause
                continue
            for index in range(0, len(clause), chunk_size):
                piece = clause[index : index + chunk_size].strip()
                if piece:
                    chunks.append(piece)
            current = ""
        if current:
            chunks.append(current + (" " if has_trailing_space else ""))

    return chunks or [normalized]
