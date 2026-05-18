from __future__ import annotations

import pytest

from src.core.config import settings
from src.core.embeddings import (
    LocalHashEmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
    get_embedding_provider,
    resolve_keyword_backend,
    resolve_vector_backend,
)


def test_get_embedding_provider_uses_local_hash_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "RAG_EMBEDDING_PROVIDER", "local_hash")
    monkeypatch.setattr(settings, "RAG_EMBEDDING_MODEL", "local-hash-v1")
    monkeypatch.setattr(settings, "RAG_EMBEDDING_DIMENSIONS", 64)

    provider = get_embedding_provider(settings)

    assert isinstance(provider, LocalHashEmbeddingProvider)
    assert provider.provider_name == "local_hash"
    assert provider.model_name == "local-hash-v1"
    assert len(provider.embed_text("career guidance")) == 64


def test_get_embedding_provider_rejects_unknown_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "RAG_EMBEDDING_PROVIDER", "unknown-provider")
    monkeypatch.setattr(settings, "RAG_EMBEDDING_MODEL", "unknown-v1")

    with pytest.raises(ValueError, match="Unsupported RAG embedding provider"):
        get_embedding_provider(settings)


def test_get_embedding_provider_uses_openai_compatible_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "RAG_EMBEDDING_PROVIDER", "openai_compatible")
    monkeypatch.setattr(settings, "RAG_EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setattr(settings, "RAG_EMBEDDING_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setattr(settings, "RAG_EMBEDDING_API_KEY", "secret-key")
    monkeypatch.setattr(settings, "RAG_EMBEDDING_TIMEOUT_SECONDS", 15)

    provider = get_embedding_provider(settings)

    assert isinstance(provider, OpenAICompatibleEmbeddingProvider)
    assert provider.provider_name == "openai_compatible"
    assert provider.model_name == "text-embedding-3-small"
    assert provider.base_url == "https://api.example.com/v1"
    assert provider.timeout_seconds == 15


def test_openai_compatible_provider_embeds_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "data": [
                    {"index": 0, "embedding": [0.1, 0.2]},
                    {"index": 1, "embedding": [0.3, 0.4]},
                ]
            }

    class _Client:
        def __init__(self, *, timeout: int):  # noqa: ANN204
            captured["timeout"] = timeout

        def __enter__(self):  # noqa: ANN204
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001, ANN204
            del exc_type, exc, tb
            return None

        def post(self, url: str, *, json: dict[str, object], headers: dict[str, str]):  # noqa: ANN204
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            return _Response()

    monkeypatch.setattr("src.core.embeddings.httpx.Client", _Client)

    provider = OpenAICompatibleEmbeddingProvider(
        api_key="secret-key",
        base_url="https://api.example.com/v1",
        timeout_seconds=12,
        model_name="text-embedding-3-small",
    )

    vectors = provider.embed_texts(["career", "remedy"])

    assert captured["timeout"] == 12
    assert captured["url"] == "https://api.example.com/v1/embeddings"
    assert captured["json"] == {
        "model": "text-embedding-3-small",
        "input": ["career", "remedy"],
    }
    assert captured["headers"] == {
        "Authorization": "Bearer secret-key",
        "Content-Type": "application/json",
    }
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]


def test_resolve_vector_backend_defaults_to_json_scan_without_postgres(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "RAG_VECTOR_BACKEND", "auto")

    assert resolve_vector_backend(db=None, resolved_settings=settings) == "json_scan"


def test_resolve_vector_backend_rejects_pgvector_without_postgres(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "RAG_VECTOR_BACKEND", "pgvector")

    with pytest.raises(ValueError, match="pgvector backend requires PostgreSQL"):
        resolve_vector_backend(db=None, resolved_settings=settings)


def test_resolve_keyword_backend_defaults_to_keyword_scan_without_postgres() -> None:
    assert resolve_keyword_backend(db=None, resolved_settings=settings) == "keyword_scan"
