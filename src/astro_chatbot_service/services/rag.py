from __future__ import annotations

import re

from sqlalchemy import select
from sqlalchemy.orm import Session

from astro_chatbot_service.models.database import KnowledgeDocument
from astro_chatbot_service.models.schemas import KnowledgeDocumentCreate, RetrievalMatch


class RAGService:
    def __init__(self, db: Session) -> None:
        self.db = db

    def ingest(self, documents: list[KnowledgeDocumentCreate]) -> int:
        for document in documents:
            row = KnowledgeDocument(
                source=document.source,
                title=document.title,
                content=document.content,
                tags=",".join(document.tags),
            )
            self.db.add(row)
        self.db.commit()
        return len(documents)

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalMatch]:
        rows = self.db.execute(select(KnowledgeDocument)).scalars().all()
        query_terms = self._tokenize(query)
        scored: list[tuple[int, KnowledgeDocument]] = []
        for row in rows:
            haystack = f"{row.title} {row.content} {row.tags}".lower()
            score = sum(haystack.count(term) for term in query_terms)
            if score > 0:
                scored.append((score, row))
        scored.sort(key=lambda item: (item[0], item[1].id), reverse=True)

        matches: list[RetrievalMatch] = []
        for score, row in scored[: top_k or 3]:
            matches.append(
                RetrievalMatch(
                    id=row.id,
                    source=row.source,
                    title=row.title,
                    excerpt=row.content[:200],
                    score=score,
                    tags=[tag for tag in row.tags.split(",") if tag],
                )
            )
        return matches

    @staticmethod
    def _tokenize(value: str) -> list[str]:
        return [token for token in re.findall(r"[a-zA-Z0-9]+", value.lower()) if len(token) > 2]

