from __future__ import annotations

import pytest

from src.core.config import settings
from src.core.reranker import (
    GroqListwiseRerankerProvider,
    HeuristicRerankerProvider,
    RerankItem,
    get_reranker_provider,
)


def test_get_reranker_provider_uses_heuristic_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "RAG_RERANKER_PROVIDER", "heuristic")
    monkeypatch.setattr(settings, "RAG_RERANKER_MODEL", "heuristic-v1")

    provider = get_reranker_provider(settings)

    assert isinstance(provider, HeuristicRerankerProvider)
    assert provider.provider_name == "heuristic"
    assert provider.model_name == "heuristic-v1"


def test_get_reranker_provider_uses_groq_listwise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "RAG_RERANKER_PROVIDER", "groq_listwise")
    monkeypatch.setattr(settings, "RAG_RERANKER_MODEL", "llama-3.1-8b-instant")
    monkeypatch.setattr(settings, "RAG_RERANKER_BASE_URL", "https://api.groq.com/openai/v1")
    monkeypatch.setattr(settings, "RAG_RERANKER_API_KEY", "secret-key")
    monkeypatch.setattr(settings, "RAG_RERANKER_TIMEOUT_SECONDS", 12)

    provider = get_reranker_provider(settings)

    assert isinstance(provider, GroqListwiseRerankerProvider)
    assert provider.provider_name == "groq_listwise"
    assert provider.model_name == "llama-3.1-8b-instant"
    assert provider.timeout_seconds == 12


def test_heuristic_reranker_prefers_chart_and_entity_aligned_item() -> None:
    provider = HeuristicRerankerProvider()
    items = [
        RerankItem(
            item_id="saturn",
            title="saturn tenth house",
            text="Saturn in the 10th house can delay career but build lasting status.",
            metadata={
                "base_score": 15.0,
                "semantic_score": 0.8,
                "lexical_score": 4,
                "entity_score": 12,
                "chart_score": 8,
            },
        ),
        RerankItem(
            item_id="venus",
            title="venus seventh house",
            text="Venus in the 7th house supports love and relationship harmony.",
            metadata={
                "base_score": 15.5,
                "semantic_score": 0.82,
                "lexical_score": 4,
                "entity_score": 2,
                "chart_score": 0,
            },
        ),
    ]

    ranked = provider.rerank("career delay saturn 10th house", items, top_k=2)

    assert ranked[0].item_id == "saturn"
