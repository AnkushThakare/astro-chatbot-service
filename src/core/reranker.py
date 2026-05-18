from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Any, Protocol

import httpx

from src.core.config import Settings, settings


@dataclass(frozen=True)
class RerankItem:
    item_id: str
    title: str
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RerankResult:
    item_id: str
    score: float
    rank: int


class RerankerProvider(Protocol):
    provider_name: str
    model_name: str

    def rerank(
        self,
        query: str,
        items: list[RerankItem],
        *,
        top_k: int,
    ) -> list[RerankResult]:
        ...


def _normalize_tokens(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-zA-Z0-9]+", text.lower()) if len(token) > 2]


@dataclass(frozen=True)
class HeuristicRerankerProvider:
    model_name: str = "heuristic-v1"
    provider_name: str = "heuristic"

    def rerank(
        self,
        query: str,
        items: list[RerankItem],
        *,
        top_k: int,
    ) -> list[RerankResult]:
        query_tokens = _normalize_tokens(query)
        query_token_set = set(query_tokens)
        query_phrase = " ".join(query_tokens)
        scored: list[tuple[float, RerankItem]] = []

        for item in items:
            text = f"{item.title}\n{item.text}".lower()
            text_tokens = _normalize_tokens(text)
            token_overlap = len(query_token_set & set(text_tokens))
            phrase_bonus = 3 if query_phrase and query_phrase in text else 0
            metadata = item.metadata
            base_score = float(metadata.get("base_score", 0.0))
            semantic_score = float(metadata.get("semantic_score", 0.0))
            lexical_score = float(metadata.get("lexical_score", 0.0))
            entity_score = float(metadata.get("entity_score", 0.0))
            chart_score = float(metadata.get("chart_score", 0.0))
            rerank_score = (
                base_score
                + semantic_score * 3
                + lexical_score
                + entity_score * 2
                + chart_score * 2
                + token_overlap
                + phrase_bonus
            )
            scored.append((rerank_score, item))

        scored.sort(key=lambda entry: (entry[0], entry[1].title), reverse=True)
        return [
            RerankResult(item_id=item.item_id, score=round(score, 4), rank=rank)
            for rank, (score, item) in enumerate(scored[:top_k], start=1)
        ]


@dataclass(frozen=True)
class GroqListwiseRerankerProvider:
    api_key: str
    base_url: str
    timeout_seconds: int
    model_name: str
    provider_name: str = "groq_listwise"

    def _endpoint(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are a reranker for Vedic astrology retrieval. "
            "Rank candidates only by relevance to the query. "
            "Prefer exact astrology entity matches such as planet, house, sign, dasha, nakshatra, remedy, and transit. "
            "Return strict JSON with this shape: "
            '{"ranked_ids":[{"id":"candidate-id","score":0.0}]}.'
        )

    def _user_prompt(self, query: str, items: list[RerankItem], top_k: int) -> str:
        serialized_items = [
            {
                "id": item.item_id,
                "title": item.title,
                "text": item.text[:500],
                "metadata": {
                    "type": item.metadata.get("type"),
                    "domain": item.metadata.get("domain"),
                    "source_citation": item.metadata.get("source_citation"),
                    "astro_entities": item.metadata.get("astro_entities"),
                },
            }
            for item in items
        ]
        return json.dumps(
            {
                "query": query,
                "top_k": top_k,
                "candidates": serialized_items,
            },
            ensure_ascii=True,
        )

    def rerank(
        self,
        query: str,
        items: list[RerankItem],
        *,
        top_k: int,
    ) -> list[RerankResult]:
        if not items:
            return []

        payload = {
            "model": self.model_name,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": self._user_prompt(query, items, top_k)},
            ],
        }
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(self._endpoint(), json=payload, headers=self._headers())
            response.raise_for_status()
            parsed = response.json()

        content = (((parsed.get("choices") or [{}])[0].get("message") or {}).get("content")) or "{}"
        rerank_payload = json.loads(content)
        ranked_ids = rerank_payload.get("ranked_ids")
        if not isinstance(ranked_ids, list):
            raise ValueError("Reranker response did not include ranked_ids")

        valid_ids = {item.item_id for item in items}
        results: list[RerankResult] = []
        for rank, entry in enumerate(ranked_ids[:top_k], start=1):
            if not isinstance(entry, dict):
                continue
            item_id = entry.get("id")
            if not isinstance(item_id, str) or item_id not in valid_ids:
                continue
            raw_score = entry.get("score")
            if isinstance(raw_score, (int, float)):
                score = float(raw_score)
            else:
                score = max(0.0, 1.0 - (rank - 1) / max(len(items), 1))
            results.append(RerankResult(item_id=item_id, score=round(score, 4), rank=rank))
        if not results:
            raise ValueError("Reranker response did not include any valid ids")
        return results


@lru_cache
def _provider_from_config(
    provider_name: str,
    model_name: str,
    base_url: str,
    api_key: str | None,
    timeout_seconds: int,
) -> RerankerProvider:
    normalized_provider = provider_name.strip().lower().replace("-", "_")
    if normalized_provider == "heuristic":
        return HeuristicRerankerProvider(model_name=model_name or "heuristic-v1")
    if normalized_provider == "groq_listwise":
        resolved_key = api_key or settings.GROQ_API_KEY
        resolved_base_url = base_url or settings.GROQ_BASE_URL
        if not resolved_key:
            raise ValueError("RAG_RERANKER_API_KEY or GROQ_API_KEY is required for groq_listwise reranker")
        return GroqListwiseRerankerProvider(
            api_key=resolved_key,
            base_url=resolved_base_url,
            timeout_seconds=timeout_seconds,
            model_name=model_name,
        )
    raise ValueError(f"Unsupported RAG reranker provider: {provider_name}")


def get_reranker_provider(resolved_settings: Settings | None = None) -> RerankerProvider:
    settings_obj = resolved_settings or settings
    return _provider_from_config(
        settings_obj.RAG_RERANKER_PROVIDER,
        settings_obj.RAG_RERANKER_MODEL,
        settings_obj.RAG_RERANKER_BASE_URL,
        settings_obj.RAG_RERANKER_API_KEY,
        settings_obj.RAG_RERANKER_TIMEOUT_SECONDS,
    )
