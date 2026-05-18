from __future__ import annotations

import json
import math
import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol
from typing import Any

import httpx
from sqlalchemy import func, literal, select
from sqlalchemy.orm import Session

from src.core.config import Settings, settings
from src.core.logging import get_logger
from src.db.models import Embedding

logger = get_logger(__name__)

try:
    from pgvector.sqlalchemy import Vector as PgVector  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency guard
    PgVector = None

SEMANTIC_EQUIVALENTS = {
    "job": "career",
    "jobs": "career",
    "profession": "career",
    "work": "career",
    "promotion": "career",
    "salary": "money",
    "income": "money",
    "wealth": "money",
    "finance": "money",
    "finances": "money",
    "lover": "relationship",
    "partner": "relationship",
    "marriage": "relationship",
    "spouse": "relationship",
    "compatibility": "relationship",
    "puja": "booking",
    "pooja": "booking",
    "temple": "booking",
    "pandit": "consultant",
    "astrologer": "consultant",
    "consultation": "consultant",
    "consultant": "consultant",
    "rudraksha": "remedy",
    "bracelet": "remedy",
    "mala": "remedy",
    "yantra": "remedy",
    "gemstone": "remedy",
    "kundli": "chart",
    "kundali": "chart",
    "birth": "chart",
    "planet": "chart",
    "houses": "chart",
}


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def normalize_embedding_tokens(text: str) -> list[str]:
    tokens = [token for token in re.findall(r"[a-zA-Z0-9]+", text.lower()) if len(token) > 2]
    normalized: list[str] = []
    for token in tokens:
        normalized.append(SEMANTIC_EQUIVALENTS.get(token, token))
    return normalized


def _build_local_hash_embedding_vector(text: str, dimensions: int) -> list[float]:
    vector = [0.0] * dimensions
    for token in normalize_embedding_tokens(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    return vector


class EmbeddingProvider(Protocol):
    provider_name: str
    model_name: str

    def embed_text(self, text: str) -> list[float]:
        ...

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...


@dataclass(frozen=True)
class LocalHashEmbeddingProvider:
    dimensions: int
    model_name: str = "local-hash-v1"
    provider_name: str = "local_hash"

    def embed_text(self, text: str) -> list[float]:
        return _build_local_hash_embedding_vector(text, self.dimensions)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]


@dataclass(frozen=True)
class OpenAICompatibleEmbeddingProvider:
    api_key: str
    base_url: str
    timeout_seconds: int
    model_name: str
    provider_name: str = "openai_compatible"

    def _endpoint(self) -> str:
        return f"{self.base_url.rstrip('/')}/embeddings"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        payload = {
            "model": self.model_name,
            "input": texts,
        }
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(
                self._endpoint(),
                json=payload,
                headers=self._headers(),
            )
            response.raise_for_status()
            parsed = response.json()

        raw_data = parsed.get("data")
        if not isinstance(raw_data, list):
            raise ValueError("Embedding response did not include a 'data' list")

        indexed_vectors: dict[int, list[float]] = {}
        for item in raw_data:
            if not isinstance(item, dict):
                continue
            index = item.get("index")
            embedding = item.get("embedding")
            if not isinstance(index, int) or not isinstance(embedding, list):
                continue
            vector = [float(value) for value in embedding if isinstance(value, (int, float))]
            if not vector:
                continue
            indexed_vectors[index] = vector

        if len(indexed_vectors) != len(texts):
            raise ValueError("Embedding response count did not match request count")

        return [indexed_vectors[index] for index in range(len(texts))]


@lru_cache
def _provider_from_config(
    provider_name: str,
    model_name: str,
    dimensions: int,
    base_url: str,
    api_key: str | None,
    timeout_seconds: int,
) -> EmbeddingProvider:
    normalized_provider = provider_name.strip().lower().replace("-", "_")
    if normalized_provider == "local_hash":
        return LocalHashEmbeddingProvider(dimensions=dimensions, model_name=model_name)
    if normalized_provider == "openai_compatible":
        if not api_key:
            raise ValueError("RAG_EMBEDDING_API_KEY is required for openai_compatible provider")
        return OpenAICompatibleEmbeddingProvider(
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            model_name=model_name,
        )
    raise ValueError(f"Unsupported RAG embedding provider: {provider_name}")


def get_embedding_provider(resolved_settings: Settings | None = None) -> EmbeddingProvider:
    settings_obj = resolved_settings or settings
    return _provider_from_config(
        settings_obj.RAG_EMBEDDING_PROVIDER,
        settings_obj.RAG_EMBEDDING_MODEL,
        settings_obj.RAG_EMBEDDING_DIMENSIONS,
        settings_obj.RAG_EMBEDDING_BASE_URL,
        settings_obj.RAG_EMBEDDING_API_KEY,
        settings_obj.RAG_EMBEDDING_TIMEOUT_SECONDS,
    )


def build_local_embedding_vector(text: str, dimensions: int | None = None) -> list[float]:
    dims = dimensions or settings.RAG_EMBEDDING_DIMENSIONS
    provider = LocalHashEmbeddingProvider(dimensions=dims, model_name="local-hash-v1")
    return provider.embed_text(text)


def resolve_vector_backend(
    db: Session | None,
    resolved_settings: Settings | None = None,
) -> str:
    settings_obj = resolved_settings or settings
    configured = settings_obj.RAG_VECTOR_BACKEND.strip().lower().replace("-", "_")
    if configured == "json_scan":
        return "json_scan"

    dialect_name = ""
    if db is not None and getattr(db, "bind", None) is not None and db.bind.dialect is not None:
        dialect_name = str(db.bind.dialect.name or "")

    pgvector_available = PgVector is not None and dialect_name == "postgresql"
    if configured == "pgvector":
        if not pgvector_available:
            raise ValueError("pgvector backend requires PostgreSQL and the pgvector package")
        return "pgvector"
    if configured == "auto" and pgvector_available:
        return "pgvector"
    return "json_scan"


def resolve_keyword_backend(
    db: Session | None,
    resolved_settings: Settings | None = None,
) -> str:
    del resolved_settings
    dialect_name = ""
    if db is not None and getattr(db, "bind", None) is not None and db.bind.dialect is not None:
        dialect_name = str(db.bind.dialect.name or "")
    if dialect_name == "postgresql":
        return "postgres_fts"
    return "keyword_scan"


class EmbeddingService:
    """Lightweight embedding store using the Embedding table.

    Supports two modes:
    1. Pre-computed embeddings loaded via ingest script (vector_json populated)
    2. Keyword fallback when no embeddings are available

    To use real vector search, run the ingestion script first:
        python -m scripts.ingest_astrology_texts
    """

    def __init__(self, db: Session, resolved_settings: Settings | None = None) -> None:
        self.db = db
        self.settings = resolved_settings or settings

    def upsert_embedding(
        self,
        source_type: str,
        source_id: str,
        content: str,
        vector: list[float] | None = None,
        model: str | None = None,
    ) -> Embedding:
        vector_backend = resolve_vector_backend(self.db, self.settings)
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
                vector_pg=vector if vector and vector_backend == "pgvector" else None,
            )
            self.db.add(row)
        else:
            row.content = content
            if vector:
                row.vector_json = json.dumps(vector)
                row.embedding_model = model
                row.vector_pg = vector if vector_backend == "pgvector" else None
        self.db.commit()
        self.db.refresh(row)
        return row

    def search_by_vector(
        self,
        query_vector: list[float],
        source_type: str | None = None,
        top_k: int = 3,
        embedding_model: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search stored embeddings by cosine similarity."""
        vector_backend = resolve_vector_backend(self.db, self.settings)
        if vector_backend == "pgvector":
            distance = Embedding.vector_pg.cosine_distance(query_vector)
            statement = select(Embedding, distance.label("distance")).where(Embedding.vector_pg.isnot(None))
            if source_type:
                statement = statement.where(Embedding.source_type == source_type)
            if embedding_model:
                statement = statement.where(Embedding.embedding_model == embedding_model)
            statement = statement.order_by(distance).limit(top_k)
            rows = self.db.execute(statement).all()
            return [
                {
                    "source_type": row.source_type,
                    "source_id": row.source_id,
                    "content": row.content,
                    "score": round(max(0.0, 1.0 - float(distance_value)), 4),
                    "embedding_model": row.embedding_model,
                    "vector_json": row.vector_json,
                    "vector_backend": vector_backend,
                }
                for row, distance_value in rows
            ]

        statement = select(Embedding).where(Embedding.vector_json.isnot(None))
        if source_type:
            statement = statement.where(Embedding.source_type == source_type)
        if embedding_model:
            statement = statement.where(Embedding.embedding_model == embedding_model)
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
                "embedding_model": row.embedding_model,
                "vector_json": row.vector_json,
                "vector_backend": vector_backend,
            }
            for score, row in scored[:top_k]
        ]

    def search_by_keyword(
        self,
        query: str,
        source_type: str | None = None,
        top_k: int = 3,
        embedding_model: str | None = None,
    ) -> list[dict[str, Any]]:
        """Keyword retrieval using PostgreSQL full-text search when available."""
        normalized_query = " ".join(normalize_embedding_tokens(query)) or query
        keyword_backend = resolve_keyword_backend(self.db, self.settings)
        if keyword_backend == "postgres_fts":
            text_search_config = self.settings.RAG_TEXT_SEARCH_CONFIG
            tsvector = func.to_tsvector(literal(text_search_config), Embedding.content)
            tsquery = func.websearch_to_tsquery(literal(text_search_config), normalized_query)
            rank = func.ts_rank_cd(tsvector, tsquery)
            statement = (
                select(Embedding, rank.label("rank"))
                .where(tsvector.op("@@")(tsquery))
            )
            if source_type:
                statement = statement.where(Embedding.source_type == source_type)
            if embedding_model:
                statement = statement.where(Embedding.embedding_model == embedding_model)
            statement = statement.order_by(rank.desc()).limit(top_k)
            rows = self.db.execute(statement).all()
            return [
                {
                    "source_type": row.source_type,
                    "source_id": row.source_id,
                    "content": row.content,
                    "score": round(float(rank_value), 4),
                    "embedding_model": row.embedding_model,
                    "vector_json": row.vector_json,
                    "vector_backend": resolve_vector_backend(self.db, self.settings),
                    "keyword_backend": keyword_backend,
                }
                for row, rank_value in rows
            ]

        statement = select(Embedding)
        if source_type:
            statement = statement.where(Embedding.source_type == source_type)
        if embedding_model:
            statement = statement.where(Embedding.embedding_model == embedding_model)
        rows = self.db.execute(statement).scalars().all()

        terms = [t.lower() for t in normalized_query.split() if len(t) > 2]
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
                "content": row.content,
                "score": score,
                "embedding_model": row.embedding_model,
                "vector_json": row.vector_json,
                "vector_backend": resolve_vector_backend(self.db, self.settings),
                "keyword_backend": keyword_backend,
            }
            for score, row in scored[:top_k]
        ]
