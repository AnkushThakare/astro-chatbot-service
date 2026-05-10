from __future__ import annotations

import json
import math
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.core.config import settings
from src.core.logging import get_logger
from src.db.models import Embedding

logger = get_logger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


class EmbeddingService:
    """Lightweight embedding store using the Embedding table.

    Supports two modes:
    1. Pre-computed embeddings loaded via ingest script (vector_json populated)
    2. Keyword fallback when no embeddings are available

    To use real vector search, run the ingestion script first:
        python -m scripts.ingest_astrology_texts
    """

    def __init__(self, db: Session) -> None:
        self.db = db

    def upsert_embedding(
        self,
        source_type: str,
        source_id: str,
        content: str,
        vector: list[float] | None = None,
        model: str | None = None,
    ) -> Embedding:
        statement = select(Embedding).where(
            Embedding.source_type == source_type,
            Embedding.source_id == source_id,
        )
        row = self.db.execute(statement).scalar_one_or_none()
        if row is None:
            row = Embedding(
                source_type=source_type,
                source_id=source_id,
                content=content,
                embedding_model=model,
                vector_json=json.dumps(vector) if vector else None,
            )
            self.db.add(row)
        else:
            row.content = content
            if vector:
                row.vector_json = json.dumps(vector)
                row.embedding_model = model
        self.db.commit()
        self.db.refresh(row)
        return row

    def search_by_vector(
        self,
        query_vector: list[float],
        source_type: str | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Search stored embeddings by cosine similarity."""
        statement = select(Embedding).where(Embedding.vector_json.isnot(None))
        if source_type:
            statement = statement.where(Embedding.source_type == source_type)
        rows = self.db.execute(statement).scalars().all()

        scored: list[tuple[float, Embedding]] = []
        for row in rows:
            try:
                stored_vector = json.loads(row.vector_json)
            except (json.JSONDecodeError, TypeError):
                continue
            score = _cosine_similarity(query_vector, stored_vector)
            scored.append((score, row))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "source_type": row.source_type,
                "source_id": row.source_id,
                "content": row.content,
                "score": round(score, 4),
            }
            for score, row in scored[:top_k]
        ]

    def search_by_keyword(
        self,
        query: str,
        source_type: str | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Fallback keyword search when vectors are not available."""
        statement = select(Embedding)
        if source_type:
            statement = statement.where(Embedding.source_type == source_type)
        rows = self.db.execute(statement).scalars().all()

        terms = [t.lower() for t in query.split() if len(t) > 2]
        scored: list[tuple[int, Embedding]] = []
        for row in rows:
            haystack = row.content.lower()
            score = sum(haystack.count(term) for term in terms)
            if score > 0:
                scored.append((score, row))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "source_type": row.source_type,
                "source_id": row.source_id,
                "content": row.content[:300],
                "score": score,
            }
            for score, row in scored[:top_k]
        ]
