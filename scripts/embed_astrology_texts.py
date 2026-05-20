"""Ingest astrology text files into the Embedding table for RAG retrieval.

Usage:
    python -m scripts.embed_astrology_texts

Reads all .txt and .md files from data/astrology_texts/ and stores them
in the embeddings table. Files are split with the same paragraph-aware,
astrology-aware chunker used by RAGService.

Stores chunk content and vectors in the embeddings table. `RAGService`
now prefers this DB-backed corpus when rows are available, and falls back
to filesystem RAG otherwise.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import settings
from src.core.embeddings import EmbeddingService, get_embedding_provider
from src.core.rag import RAGService
from src.db.session import configure_database, get_db


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

    configure_database(
        sync_database_url=settings.sync_database_url,
        async_database_url=settings.async_database_url,
    )
    db = next(get_db())
    service = EmbeddingService(db)
    embedding_provider = get_embedding_provider(settings)
    total_chunks = 0

    for file_path in files:
        content = file_path.read_text(encoding="utf-8").strip()
        if not content:
            continue

        rel_path = file_path.relative_to(data_dir)
        chunks = RAGService._chunk_text(
            content,
            chunk_size_words=settings.RAG_CHUNK_SIZE_WORDS,
            overlap_words=settings.RAG_CHUNK_OVERLAP_WORDS,
        )

        for i, chunk in enumerate(chunks):
            source_id = f"{rel_path}:chunk_{i}"
            service.upsert_embedding(
                source_type="astrology_text",
                source_id=source_id,
                content=chunk,
                vector=embedding_provider.embed_text(chunk),
                model=embedding_provider.model_name,
            )
            total_chunks += 1

        print(f"  {rel_path}: {len(chunks)} chunk(s)")

    print(f"\nIngested {total_chunks} chunks from {len(files)} files.")
    print(
        "DB-backed retrieval is now available to RAGService via the embeddings table "
        f"using provider={embedding_provider.provider_name} model={embedding_provider.model_name}."
    )


if __name__ == "__main__":
    main()
