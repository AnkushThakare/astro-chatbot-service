from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from src.core.config import settings


@dataclass
class RetrievalHit:
    title: str
    excerpt: str
    path: str
    score: int


class RAGService:
    @staticmethod
    @lru_cache
    def _documents() -> list[tuple[str, str, str]]:
        documents: list[tuple[str, str, str]] = []
        data_dir = settings.rag_data_dir
        if not data_dir.exists():
            return documents

        for path in sorted(data_dir.glob("**/*")):
            if path.is_file() and path.suffix.lower() in {".txt", ".md"}:
                content = path.read_text(encoding="utf-8").strip()
                documents.append((path.stem.replace("_", " "), str(path), content))
        return documents

    def retrieve(self, query: str, top_k: int) -> list[dict[str, str | int]]:
        query_terms = [term for term in re.findall(r"[a-zA-Z0-9]+", query.lower()) if len(term) > 2]
        scored_hits: list[RetrievalHit] = []

        for title, path, content in self._documents():
            haystack = f"{title}\n{content}".lower()
            score = sum(haystack.count(term) for term in query_terms)
            if score > 0:
                scored_hits.append(
                    RetrievalHit(
                        title=title,
                        excerpt=content[:220],
                        path=path,
                        score=score,
                    )
                )

        scored_hits.sort(key=lambda hit: (hit.score, hit.title), reverse=True)
        return [
            {
                "title": hit.title,
                "excerpt": hit.excerpt,
                "path": hit.path,
                "score": hit.score,
            }
            for hit in scored_hits[:top_k]
        ]
