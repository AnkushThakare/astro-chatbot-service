"""Ingest astrology text files into the Embedding table for RAG retrieval.

Usage:
    python -m scripts.embed_astrology_texts

Reads all .txt and .md files from data/astrology_texts/ and stores them
in the embeddings table. Each file is split into chunks of ~500 words.

Currently stores content for keyword search only. To add vector embeddings,
set EMBEDDING_API_KEY and uncomment the embedding call below.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import settings
from src.core.embeddings import EmbeddingService
from src.db.session import configure_database, get_db


CHUNK_SIZE_WORDS = 500
CHUNK_OVERLAP_WORDS = 50


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_WORDS, overlap: int = CHUNK_OVERLAP_WORDS) -> list[str]:
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
    return chunks


def main() -> None:
    data_dir = settings.rag_data_dir
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        print("Create it and add .txt or .md files with astrology knowledge.")
        return

    files = sorted(
        p for p in data_dir.glob("**/*")
        if p.is_file() and p.suffix.lower() in {".txt", ".md"}
    )
    if not files:
        print(f"No .txt or .md files found in {data_dir}")
        return

    configure_database(settings)
    db = next(get_db())
    service = EmbeddingService(db)
    total_chunks = 0

    for file_path in files:
        content = file_path.read_text(encoding="utf-8").strip()
        if not content:
            continue

        rel_path = file_path.relative_to(data_dir)
        chunks = chunk_text(content)

        for i, chunk in enumerate(chunks):
            source_id = f"{rel_path}:chunk_{i}"
            service.upsert_embedding(
                source_type="astrology_text",
                source_id=source_id,
                content=chunk,
            )
            total_chunks += 1

        print(f"  {rel_path}: {len(chunks)} chunk(s)")

    print(f"\nIngested {total_chunks} chunks from {len(files)} files.")
    print("Keyword search is now available via EmbeddingService.search_by_keyword().")


if __name__ == "__main__":
    main()
