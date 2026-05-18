from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
import time
from typing import Any

from src.core.config import settings
from src.core.embeddings import (
    EmbeddingProvider,
    EmbeddingService,
    get_embedding_provider,
    normalize_embedding_tokens,
    resolve_keyword_backend,
    resolve_vector_backend,
)
from src.core.reranker import RerankItem, get_reranker_provider
from src.db.models import Embedding
from sqlalchemy import select
from sqlalchemy.orm import Session


@dataclass
class RetrievalHit:
    title: str
    excerpt: str
    path: str
    score: float
    metadata: dict[str, Any]


@dataclass
class RetrievalDocument:
    title: str
    path: str
    content: str
    metadata: dict[str, Any]
    vector: list[float]


@dataclass
class RetrievalCandidate:
    document: RetrievalDocument
    lexical_score: int
    semantic_score: float
    title_score: int
    action_score: int
    topic_score: int
    domain_score: int
    entity_score: int
    chunk_type_score: int
    chart_score: int

    @property
    def key(self) -> tuple[str, Any, str]:
        return (
            self.document.path,
            self.document.metadata.get("chunk_index"),
            self.document.title,
        )

    @property
    def rerank_id(self) -> str:
        return f"{self.document.path}::{self.document.metadata.get('chunk_index')}::{self.document.title}"


@dataclass
class RetrievalCorpus:
    documents: list[RetrievalDocument]
    metadata: dict[str, Any]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


class RAGService:
    _query_cache: dict[tuple[str, int, str | None, str], tuple[float, dict[str, Any]]] = {}
    EMBEDDING_SOURCE_TYPE = "astrology_text"
    KNOWLEDGE_DOMAINS = {"astrology_reference", "general_guidance"}
    POLICY_DOMAINS = {"product_policy", "booking_guidance", "remedy_guidance"}
    ACTION_DOMAIN_HINTS = {
        "recommend_product": {"product_policy", "remedy_guidance", "general_guidance"},
        "book_pooja": {"booking_guidance", "general_guidance"},
        "suggest_consultant": {"booking_guidance", "general_guidance"},
        "show_kundali": {"astrology_reference", "general_guidance"},
        "matchmaking": {"astrology_reference", "general_guidance"},
        "respond_only": {
            "general_guidance",
            "astrology_reference",
            "remedy_guidance",
            "product_policy",
            "booking_guidance",
        },
        "ask_clarification": {
            "general_guidance",
            "astrology_reference",
            "remedy_guidance",
            "product_policy",
            "booking_guidance",
        },
    }
    TOPIC_KEYWORDS = {
        "career": {"career", "job", "profession", "promotion", "work"},
        "finance": {"finance", "money", "wealth", "income", "business"},
        "health": {"health", "stress", "sleep", "routine", "wellbeing"},
        "love": {"love", "relationship", "marriage", "partner", "compatibility"},
        "spirituality": {"spirituality", "meditation", "inner", "retreat", "prayer"},
        "remedies": {"remedy", "remedies", "rudraksha", "bracelet", "mala", "upay"},
        "booking": {"puja", "pooja", "temple", "havan", "homam", "pandit"},
        "kundali": {"kundali", "kundli", "chart", "birth", "house", "planet"},
    }
    TOPIC_ASTRO_EXPANSIONS = {
        "career": {"houses": {10, 6}, "terms": {"10th house", "career", "profession", "saturn", "sun"}},
        "finance": {"houses": {2, 11}, "terms": {"2nd house", "11th house", "wealth", "income", "jupiter", "venus"}},
        "health": {"houses": {1, 6, 8}, "terms": {"6th house", "health", "routine", "mars", "saturn"}},
        "love": {"houses": {5, 7}, "terms": {"5th house", "7th house", "marriage", "partner", "venus", "moon"}},
        "spirituality": {"houses": {9, 12}, "terms": {"9th house", "12th house", "moksha", "guru", "ketu"}},
        "remedies": {"houses": set(), "terms": {"remedy", "upay", "mantra", "rudraksha", "donation"}},
        "booking": {"houses": set(), "terms": {"puja", "pooja", "pandit", "temple", "havan"}},
        "kundali": {"houses": {1}, "terms": {"kundali", "chart", "lagna", "ascendant"}},
    }
    PLANET_ALIASES = {
        "sun": "sun",
        "surya": "sun",
        "moon": "moon",
        "chandra": "moon",
        "mars": "mars",
        "mangal": "mars",
        "kuja": "mars",
        "mercury": "mercury",
        "budh": "mercury",
        "jupiter": "jupiter",
        "guru": "jupiter",
        "brihaspati": "jupiter",
        "venus": "venus",
        "shukra": "venus",
        "saturn": "saturn",
        "shani": "saturn",
        "rahu": "rahu",
        "ketu": "ketu",
    }
    SIGN_ALIASES = {
        "aries": "aries",
        "mesh": "aries",
        "taurus": "taurus",
        "vrishabha": "taurus",
        "gemini": "gemini",
        "mithuna": "gemini",
        "cancer": "cancer",
        "karka": "cancer",
        "leo": "leo",
        "simha": "leo",
        "virgo": "virgo",
        "kanya": "virgo",
        "libra": "libra",
        "tula": "libra",
        "scorpio": "scorpio",
        "vrischika": "scorpio",
        "sagittarius": "sagittarius",
        "dhanu": "sagittarius",
        "capricorn": "capricorn",
        "makara": "capricorn",
        "aquarius": "aquarius",
        "kumbha": "aquarius",
        "pisces": "pisces",
        "meena": "pisces",
    }
    NAKSHATRA_ALIASES = {
        "ashwini": "ashwini",
        "bharani": "bharani",
        "krittika": "krittika",
        "rohini": "rohini",
        "mrigashira": "mrigashira",
        "ardra": "ardra",
        "punarvasu": "punarvasu",
        "pushya": "pushya",
        "ashlesha": "ashlesha",
        "magha": "magha",
        "purva phalguni": "purva phalguni",
        "uttara phalguni": "uttara phalguni",
        "hasta": "hasta",
        "chitra": "chitra",
        "swati": "swati",
        "vishakha": "vishakha",
        "anuradha": "anuradha",
        "jyeshtha": "jyeshtha",
        "mula": "mula",
        "purva ashadha": "purva ashadha",
        "uttara ashadha": "uttara ashadha",
        "shravana": "shravana",
        "dhanishta": "dhanishta",
        "shatabhisha": "shatabhisha",
        "purva bhadrapada": "purva bhadrapada",
        "uttara bhadrapada": "uttara bhadrapada",
        "revati": "revati",
    }
    ASTRO_CHUNK_TYPE_HINTS = {
        "dasha_period": {"dasha", "mahadasha", "antardasha"},
        "nakshatra": {"nakshatra", "pada"},
        "yoga": {"yoga", "dosha", "sade sati", "dhaiya"},
        "transit": {"transit", "gochar"},
        "aspect": {"aspect", "drishti"},
        "conjunction": {"conjunction", "together"},
        "house_lord_in_house": {"lord"},
        "remedy": {"remedy", "upay", "mantra", "rudraksha", "bracelet", "gemstone"},
    }

    def __init__(self, db: Session | None = None) -> None:
        self.db = db
        self.embedding_provider = get_embedding_provider()
        self.reranker = get_reranker_provider()

    def _retrieval_metadata(self, corpus_metadata: dict[str, Any], *, chart_context: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            **corpus_metadata,
            "chart_context_used": bool(chart_context),
            "reranker_provider": self.reranker.provider_name,
            "reranker_model": self.reranker.model_name,
        }

    @staticmethod
    def _normalized_text(text: str) -> str:
        return re.sub(r"\s+", " ", text.lower()).strip()

    @classmethod
    def _extract_canonical_terms(
        cls,
        text: str,
        alias_map: dict[str, str],
    ) -> list[str]:
        normalized = cls._normalized_text(text)
        found: set[str] = set()
        for alias, canonical in alias_map.items():
            if re.search(rf"\b{re.escape(alias)}\b", normalized):
                found.add(canonical)
        return sorted(found)

    @classmethod
    def _extract_houses(cls, text: str) -> list[int]:
        normalized = cls._normalized_text(text)
        houses: set[int] = set()
        for match in re.findall(r"\b([1-9]|1[0-2])(?:st|nd|rd|th)?\s+house\b", normalized):
            try:
                houses.add(int(match))
            except ValueError:
                continue
        if re.search(r"\b(lagna|ascendant|first house)\b", normalized):
            houses.add(1)
        return sorted(houses)

    @classmethod
    def _extract_named_patterns(
        cls,
        text: str,
        pattern: str,
    ) -> list[str]:
        normalized = cls._normalized_text(text)
        found = {
            match.strip()
            for match in re.findall(pattern, normalized)
            if isinstance(match, str) and match.strip()
        }
        return sorted(found)

    @classmethod
    def _extract_astro_entities(cls, text: str) -> dict[str, Any]:
        normalized = cls._normalized_text(text)
        query_terms = normalize_embedding_tokens(text)
        topics = sorted(cls._query_topics(query_terms))
        remedies = sorted(
            token
            for token in {"rudraksha", "bracelet", "mala", "gemstone", "yantra", "mantra", "donation", "upay"}
            if re.search(rf"\b{re.escape(token)}\b", normalized)
        )
        dashas: set[str] = set()
        for alias, canonical in cls.PLANET_ALIASES.items():
            if re.search(rf"\b{re.escape(alias)}\s+(?:mahadasha|antardasha)\b", normalized):
                dashas.add(canonical)
        if re.search(r"\bmahadasha\b", normalized):
            dashas.add("mahadasha")
        if re.search(r"\bantardasha\b", normalized):
            dashas.add("antardasha")
        transit_terms = sorted(
            term for term in {"transit", "gochar", "sade sati", "dhaiya"} if term in normalized
        )
        yogas = cls._extract_named_patterns(
            text,
            r"\b([a-z][a-z\s-]{1,50}?(?:yoga|dosha|sade sati|dhaiya))\b",
        )
        return {
            "planets": cls._extract_canonical_terms(text, cls.PLANET_ALIASES),
            "houses": cls._extract_houses(text),
            "signs": cls._extract_canonical_terms(text, cls.SIGN_ALIASES),
            "nakshatras": cls._extract_canonical_terms(text, cls.NAKSHATRA_ALIASES),
            "dashas": sorted(dashas),
            "yogas": yogas,
            "remedies": remedies,
            "transits": transit_terms,
            "topics": topics,
        }

    @classmethod
    def _infer_chunk_type(
        cls,
        content: str,
        *,
        domain: str,
        astro_entities: dict[str, Any],
    ) -> str:
        normalized = cls._normalized_text(content)
        if astro_entities["dashas"]:
            return "dasha_period"
        if astro_entities["nakshatras"]:
            return "nakshatra"
        if astro_entities["transits"]:
            return "transit"
        if astro_entities["yogas"]:
            return "yoga"
        if domain in {"remedy_guidance", "product_policy"} and astro_entities["remedies"]:
            return "remedy"
        if len(astro_entities["planets"]) >= 2 and re.search(r"\b(conjunction|together)\b", normalized):
            return "conjunction"
        if astro_entities["planets"] and astro_entities["houses"]:
            return "planet_in_house"
        if astro_entities["planets"] and astro_entities["signs"]:
            return "planet_in_sign"
        if astro_entities["planets"] and re.search(r"\b(aspect|drishti)\b", normalized):
            return "aspect"
        if len(astro_entities["houses"]) >= 2 and "lord" in normalized:
            return "house_lord_in_house"
        return "explanation"

    @classmethod
    def _source_citation(cls, title: str, filename: str) -> str:
        base = title.strip() or Path(filename).stem.replace("_", " ")
        return base

    @staticmethod
    def _chunk_text(
        text: str,
        *,
        chunk_size_words: int,
        overlap_words: int,
    ) -> list[str]:
        normalized_text = text.strip()
        words = normalized_text.split()
        if len(words) <= chunk_size_words:
            return [normalized_text]

        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", normalized_text) if part.strip()]
        if len(paragraphs) <= 1:
            chunks: list[str] = []
            start = 0
            while start < len(words):
                end = min(start + chunk_size_words, len(words))
                chunks.append(" ".join(words[start:end]))
                if end >= len(words):
                    break
                start = max(end - overlap_words, start + 1)
            return chunks

        chunks: list[str] = []
        current_parts: list[str] = []
        current_word_count = 0
        for paragraph in paragraphs:
            paragraph_words = paragraph.split()
            if len(paragraph_words) > chunk_size_words:
                if current_parts:
                    chunks.append("\n\n".join(current_parts))
                    current_parts = []
                    current_word_count = 0
                start = 0
                while start < len(paragraph_words):
                    end = min(start + chunk_size_words, len(paragraph_words))
                    chunks.append(" ".join(paragraph_words[start:end]))
                    if end >= len(paragraph_words):
                        break
                    start = max(end - overlap_words, start + 1)
                continue

            if current_parts and current_word_count + len(paragraph_words) > chunk_size_words:
                chunks.append("\n\n".join(current_parts))
                current_parts = []
                current_word_count = 0

            current_parts.append(paragraph)
            current_word_count += len(paragraph_words)

        if current_parts:
            chunks.append("\n\n".join(current_parts))
        return chunks

    @staticmethod
    @lru_cache
    def _documents() -> list[RetrievalDocument]:
        documents: list[RetrievalDocument] = []
        data_dir = settings.rag_data_dir
        if not data_dir.exists():
            return documents
        embedding_provider = get_embedding_provider()

        for path in sorted(data_dir.glob("**/*")):
            if not path.is_file() or path.suffix.lower() not in {".txt", ".md"}:
                continue

            content = path.read_text(encoding="utf-8").strip()
            if not content:
                continue

            title = path.stem.replace("_", " ")
            chunks = RAGService._chunk_text(
                content,
                chunk_size_words=settings.RAG_CHUNK_SIZE_WORDS,
                overlap_words=settings.RAG_CHUNK_OVERLAP_WORDS,
            )
            for index, chunk in enumerate(chunks):
                chunk_metadata = RAGService._infer_metadata(path.name, title, chunk)
                documents.append(
                    RetrievalDocument(
                        title=title,
                        path=str(path),
                        content=chunk,
                        metadata={
                            **chunk_metadata,
                            "chunk_index": index,
                            "chunk_count": len(chunks),
                            "source_title": title,
                            "source_citation": RAGService._source_citation(title, path.name),
                            "embedding_model": embedding_provider.model_name,
                        },
                        vector=embedding_provider.embed_text(chunk),
                    )
                )
        return documents

    @classmethod
    def _infer_metadata(cls, filename: str, title: str, content: str) -> dict[str, Any]:
        haystack = f"{filename} {title} {content}".lower()
        action_hints = {"respond_only"}
        domain = "general_guidance"
        risk = "low"
        allowed_actions = {"explain_only"}
        astro_entities = cls._extract_astro_entities(f"{title}\n{content}")

        if any(token in haystack for token in ("rudraksha", "bracelet", "mala")):
            domain = "product_policy"
            action_hints.update({"recommend_product", "respond_only"})
            allowed_actions.update({"can_recommend"})
        elif "remed" in haystack:
            domain = "remedy_guidance"
            action_hints.update({"recommend_product", "respond_only"})
            allowed_actions.update({"can_recommend"})
        elif any(token in haystack for token in ("puja", "temple", "pandit", "havan", "homam")):
            domain = "booking_guidance"
            action_hints.update({"book_pooja", "suggest_consultant", "respond_only"})
            allowed_actions.update({"can_recommend"})
        elif any(token in haystack for token in ("kundali", "kundli", "house", "planet")):
            domain = "astrology_reference"
            action_hints.update({"show_kundali", "matchmaking", "respond_only"})

        if any(token in haystack for token in ("guarantee", "cure", "medical", "enemy", "control", "black magic")):
            risk = "high"
            allowed_actions = {"explain_only", "no_tool"}
        elif domain in {"product_policy", "booking_guidance"}:
            risk = "medium"

        topic_hints = sorted(
            topic
            for topic, keywords in cls.TOPIC_KEYWORDS.items()
            if any(keyword in haystack for keyword in keywords)
        )
        for topic in astro_entities.get("topics", []):
            if topic not in topic_hints:
                topic_hints.append(topic)
        chunk_type = cls._infer_chunk_type(content, domain=domain, astro_entities=astro_entities)

        return {
            "domain": domain,
            "action_hints": sorted(action_hints),
            "topic_hints": sorted(topic_hints),
            "source_type": "knowledge_base",
            "language": "english",
            "type": chunk_type,
            "risk": risk,
            "allowed_actions": sorted(allowed_actions),
            "astro_entities": astro_entities,
            "source_citation": cls._source_citation(title, filename),
        }

    @staticmethod
    def _parse_embedding_source_id(source_id: str) -> tuple[str, int | None]:
        path_part, separator, chunk_part = source_id.rpartition(":chunk_")
        if not separator:
            return source_id, None
        try:
            return path_part, int(chunk_part)
        except ValueError:
            return path_part, None

    @staticmethod
    def _canonical_document_path(path: str) -> str:
        raw_path = Path(path)
        try:
            return raw_path.relative_to(settings.rag_data_dir).as_posix()
        except ValueError:
            return raw_path.as_posix()

    @classmethod
    def _document_merge_key(cls, document: RetrievalDocument) -> tuple[str, int | None, str]:
        return (
            cls._canonical_document_path(document.path),
            document.metadata.get("chunk_index"),
            document.title,
        )

    @classmethod
    def _merge_corpus_documents(
        cls,
        primary_documents: list[RetrievalDocument],
        fallback_documents: list[RetrievalDocument],
    ) -> tuple[list[RetrievalDocument], int]:
        merged: dict[tuple[str, int | None, str], RetrievalDocument] = {}
        for document in primary_documents:
            merged[cls._document_merge_key(document)] = document

        fallback_added = 0
        for document in fallback_documents:
            key = cls._document_merge_key(document)
            if key in merged:
                continue
            merged[key] = document
            fallback_added += 1

        return list(merged.values()), fallback_added

    def _embedding_documents(self) -> list[RetrievalDocument]:
        if self.db is None:
            return []

        statement = select(Embedding).where(
            Embedding.source_type == self.EMBEDDING_SOURCE_TYPE,
            Embedding.vector_json.isnot(None),
            Embedding.embedding_model == self.embedding_provider.model_name,
        )
        rows = self.db.execute(statement).scalars().all()
        documents: list[RetrievalDocument] = []

        for row in rows:
            if not row.vector_json:
                continue
            try:
                raw_vector = json.loads(row.vector_json)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(raw_vector, list):
                continue

            vector = [
                float(value)
                for value in raw_vector
                if isinstance(value, (int, float))
            ]
            if not vector:
                continue

            relative_path, chunk_index = self._parse_embedding_source_id(row.source_id)
            source_path = Path(relative_path)
            title = source_path.stem.replace("_", " ") or row.source_id
            base_metadata = self._infer_metadata(source_path.name or row.source_id, title, row.content)
            documents.append(
                RetrievalDocument(
                    title=title,
                    path=relative_path,
                    content=row.content,
                    metadata={
                        **base_metadata,
                        "chunk_index": chunk_index if chunk_index is not None else 0,
                        "source_title": title,
                        "source_citation": self._source_citation(title, source_path.name or row.source_id),
                        "source_type": row.source_type,
                        "embedding_model": row.embedding_model or "unknown",
                    },
                    vector=vector,
                )
            )

        return documents

    def _embedding_documents_for_queries(
        self,
        queries: list[str],
        *,
        top_k: int,
    ) -> list[RetrievalDocument]:
        if self.db is None:
            return []

        embedding_service = EmbeddingService(self.db, settings)
        candidate_limit = self._candidate_limit(top_k)
        documents_by_source: dict[str, RetrievalDocument] = {}

        for query in queries:
            normalized_query = query.strip()
            if not normalized_query:
                continue

            query_vector = self.embedding_provider.embed_text(normalized_query)
            vector_hits = embedding_service.search_by_vector(
                query_vector,
                source_type=self.EMBEDDING_SOURCE_TYPE,
                top_k=candidate_limit,
                embedding_model=self.embedding_provider.model_name,
            )
            keyword_hits = embedding_service.search_by_keyword(
                normalized_query,
                source_type=self.EMBEDDING_SOURCE_TYPE,
                top_k=candidate_limit,
                embedding_model=self.embedding_provider.model_name,
            )

            for hit in [*vector_hits, *keyword_hits]:
                source_id = str(hit.get("source_id") or "")
                content = hit.get("content")
                if not source_id or not isinstance(content, str) or not content.strip():
                    continue
                vector_json = hit.get("vector_json")
                vector: list[float] | None = None
                if isinstance(vector_json, str):
                    try:
                        raw_vector = json.loads(vector_json)
                    except (json.JSONDecodeError, TypeError):
                        raw_vector = None
                    if isinstance(raw_vector, list):
                        parsed_vector = [
                            float(value)
                            for value in raw_vector
                            if isinstance(value, (int, float))
                        ]
                        if parsed_vector:
                            vector = parsed_vector
                if vector is None:
                    vector = self.embedding_provider.embed_text(content)

                relative_path, chunk_index = self._parse_embedding_source_id(source_id)
                source_path = Path(relative_path)
                title = source_path.stem.replace("_", " ") or source_id
                base_metadata = self._infer_metadata(source_path.name or source_id, title, content)
                documents_by_source[source_id] = RetrievalDocument(
                    title=title,
                    path=relative_path,
                    content=content,
                    metadata={
                        **base_metadata,
                        "chunk_index": chunk_index if chunk_index is not None else 0,
                        "source_title": title,
                        "source_citation": self._source_citation(title, source_path.name or source_id),
                        "source_type": self.EMBEDDING_SOURCE_TYPE,
                        "embedding_model": hit.get("embedding_model") or "unknown",
                    },
                    vector=vector,
                )

        return list(documents_by_source.values())

    def _resolve_corpus(
        self,
        *,
        query_seeds: list[str] | None = None,
        top_k: int | None = None,
    ) -> RetrievalCorpus:
        file_documents = self._documents()
        embedding_documents = (
            self._embedding_documents_for_queries(query_seeds or [], top_k=top_k or settings.RAG_TOP_K)
            if query_seeds
            else self._embedding_documents()
        )
        if embedding_documents:
            merged_documents, filesystem_fallback_count = self._merge_corpus_documents(
                embedding_documents,
                file_documents,
            )
            vector_backend = resolve_vector_backend(self.db, settings)
            keyword_backend = resolve_keyword_backend(self.db, settings)
            return RetrievalCorpus(
                documents=merged_documents,
                metadata={
                    "provider": "embedding_store",
                    "retrieval_strategy": "db_embedding_hybrid_v3",
                    "embedding_model": self.embedding_provider.model_name,
                    "embedding_provider": "precomputed",
                    "vector_backend": vector_backend,
                    "keyword_backend": keyword_backend,
                    "document_count": len(merged_documents),
                    "db_document_count": len(embedding_documents),
                    "filesystem_fallback_count": filesystem_fallback_count,
                    "source_type": self.EMBEDDING_SOURCE_TYPE,
                },
            )

        return RetrievalCorpus(
            documents=file_documents,
            metadata={
                "provider": "filesystem",
                "retrieval_strategy": "inprocess_hybrid_v3",
                "embedding_model": self.embedding_provider.model_name,
                "embedding_provider": self.embedding_provider.provider_name,
                "vector_backend": "inprocess",
                "keyword_backend": "keyword_scan",
                "document_count": len(file_documents),
                "source_type": self.EMBEDDING_SOURCE_TYPE,
            },
        )

    @classmethod
    def _query_topics(cls, query_terms: list[str]) -> set[str]:
        query_topics: set[str] = set()
        query_term_set = set(query_terms)
        for topic, keywords in cls.TOPIC_KEYWORDS.items():
            if query_term_set & keywords:
                query_topics.add(topic)
        return query_topics

    @classmethod
    def _expand_astrology_query(
        cls,
        query: str,
        *,
        chart_context: dict[str, Any] | None = None,
    ) -> list[str]:
        normalized_query = query.strip()
        if not normalized_query:
            return []

        expansions: list[str] = [normalized_query]
        entities = cls._extract_astro_entities(normalized_query)
        for topic in entities.get("topics", []):
            expansion = cls.TOPIC_ASTRO_EXPANSIONS.get(topic)
            if not expansion:
                continue
            houses = " ".join(f"{house}th house" for house in sorted(expansion["houses"]))
            terms = " ".join(sorted(expansion["terms"]))
            variant = " ".join(part for part in [normalized_query, houses, terms] if part).strip()
            if variant and variant not in expansions:
                expansions.append(variant)

        planets = entities.get("planets", [])
        houses = entities.get("houses", [])
        if planets and houses:
            for planet in planets[:2]:
                for house in houses[:2]:
                    variant = f"{planet} {house}th house interpretation"
                    if variant not in expansions:
                        expansions.append(variant)

        if entities.get("dashas"):
            dasha_planets = [term for term in entities["dashas"] if term not in {"mahadasha", "antardasha"}]
            for planet in dasha_planets[:2]:
                variant = f"{planet} mahadasha antardasha timing"
                if variant not in expansions:
                    expansions.append(variant)

        chart_entities = cls._chart_entities_for_query(entities, chart_context)
        for planet in chart_entities.get("dashas", [])[:2]:
            variant = f"{normalized_query} {planet} mahadasha antardasha"
            if variant not in expansions:
                expansions.append(variant)
        chart_planets = chart_entities.get("planets", [])
        chart_houses = chart_entities.get("houses", [])
        for planet in chart_planets[:2]:
            for house in chart_houses[:2]:
                variant = f"{normalized_query} {planet} {house}th house"
                if variant not in expansions:
                    expansions.append(variant)

        return expansions[:5]

    @staticmethod
    def _chart_context_cache_key(chart_context: dict[str, Any] | None) -> str:
        if not chart_context:
            return ""
        ascendant = str(chart_context.get("ascendant_sign") or "")
        moon_sign = str(chart_context.get("moon_sign") or "")
        maha = str(chart_context.get("current_mahadasha") or "")
        antara = str(chart_context.get("current_antardasha") or "")
        placements = chart_context.get("placements") or []
        placement_key = "|".join(
            f"{placement.get('planet')}:{placement.get('house')}:{placement.get('sign')}"
            for placement in placements[:6]
            if isinstance(placement, dict)
        )
        return "::".join([ascendant, moon_sign, maha, antara, placement_key])

    @classmethod
    def _chart_entities_for_query(
        cls,
        query_entities: dict[str, Any],
        chart_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not chart_context:
            return {
                "planets": [],
                "houses": [],
                "signs": [],
                "nakshatras": [],
                "dashas": [],
                "yogas": [],
                "remedies": [],
                "transits": [],
                "topics": [],
            }

        chart_entities = chart_context.get("astro_entities") or {}
        placements = chart_context.get("placements") or []
        relevant_houses = set(query_entities.get("houses", []))
        for topic in query_entities.get("topics", []):
            relevant_houses.update(cls.TOPIC_ASTRO_EXPANSIONS.get(topic, {}).get("houses", set()))

        relevant_planets: set[str] = set()
        relevant_signs: set[str] = set()
        placement_houses: set[int] = set()
        for placement in placements:
            if not isinstance(placement, dict):
                continue
            house = placement.get("house")
            planet = placement.get("planet")
            sign = placement.get("sign")
            if relevant_houses and house not in relevant_houses:
                continue
            if isinstance(house, int):
                placement_houses.add(house)
            if isinstance(planet, str) and planet:
                relevant_planets.add(planet.lower())
            if isinstance(sign, str) and sign:
                relevant_signs.add(sign.lower())

        dashas = {
            str(value).lower()
            for value in chart_entities.get("dashas", [])
            if isinstance(value, str) and value
        }
        return {
            "planets": sorted(relevant_planets),
            "houses": sorted(placement_houses or relevant_houses),
            "signs": sorted(relevant_signs),
            "nakshatras": list(chart_entities.get("nakshatras", [])),
            "dashas": sorted(dashas),
            "yogas": [],
            "remedies": [],
            "transits": [],
            "topics": list(query_entities.get("topics", [])),
        }

    @staticmethod
    def _excerpt_for_query(content: str, query_terms: list[str]) -> str:
        if not content:
            return ""
        lowered = content.lower()
        match_index = min(
            (lowered.find(term) for term in query_terms if lowered.find(term) != -1),
            default=-1,
        )
        if match_index == -1:
            return content[:220]

        start = max(match_index - 80, 0)
        end = min(start + 220, len(content))
        excerpt = content[start:end].strip()
        return excerpt if start == 0 else f"...{excerpt}"

    @staticmethod
    def _candidate_limit(top_k: int) -> int:
        return max(top_k * 4, 8)

    @staticmethod
    def _query_phrase(query_terms: list[str]) -> str:
        return " ".join(query_terms).strip()

    @classmethod
    def _astro_entity_score(
        cls,
        query_entities: dict[str, Any],
        document_entities: dict[str, Any],
    ) -> int:
        score = 0
        planet_overlap = len(set(query_entities.get("planets", [])) & set(document_entities.get("planets", [])))
        house_overlap = len(set(query_entities.get("houses", [])) & set(document_entities.get("houses", [])))
        sign_overlap = len(set(query_entities.get("signs", [])) & set(document_entities.get("signs", [])))
        nakshatra_overlap = len(set(query_entities.get("nakshatras", [])) & set(document_entities.get("nakshatras", [])))
        dasha_overlap = len(set(query_entities.get("dashas", [])) & set(document_entities.get("dashas", [])))
        yoga_overlap = len(set(query_entities.get("yogas", [])) & set(document_entities.get("yogas", [])))
        remedy_overlap = len(set(query_entities.get("remedies", [])) & set(document_entities.get("remedies", [])))
        topic_overlap = len(set(query_entities.get("topics", [])) & set(document_entities.get("topics", [])))

        score += planet_overlap * 4
        score += house_overlap * 5
        score += sign_overlap * 3
        score += nakshatra_overlap * 4
        score += dasha_overlap * 4
        score += yoga_overlap * 4
        score += remedy_overlap * 2
        score += topic_overlap * 2
        if planet_overlap and house_overlap:
            score += 6
        return score

    @classmethod
    def _chunk_type_match_score(
        cls,
        query_entities: dict[str, Any],
        document_type: str,
    ) -> int:
        if not document_type:
            return 0
        if query_entities.get("dashas") and document_type == "dasha_period":
            return 4
        if query_entities.get("nakshatras") and document_type == "nakshatra":
            return 4
        if query_entities.get("yogas") and document_type == "yoga":
            return 4
        if query_entities.get("transits") and document_type == "transit":
            return 4
        if query_entities.get("remedies") and document_type == "remedy":
            return 3
        if query_entities.get("planets") and query_entities.get("houses") and document_type == "planet_in_house":
            return 4
        if query_entities.get("planets") and query_entities.get("signs") and document_type == "planet_in_sign":
            return 4
        return 0

    @classmethod
    def _chart_match_score(
        cls,
        chart_entities: dict[str, Any],
        document_entities: dict[str, Any],
    ) -> int:
        if not chart_entities:
            return 0
        score = cls._astro_entity_score(chart_entities, document_entities)
        return max(0, score // 2)

    @classmethod
    def _build_candidates(
        cls,
        *,
        query: str,
        action: str | None,
        domains: set[str] | None,
        documents: list[RetrievalDocument],
        embedding_provider: EmbeddingProvider,
        chart_context: dict[str, Any] | None = None,
    ) -> tuple[list[RetrievalCandidate], set[str]]:
        query_terms = normalize_embedding_tokens(query)
        query_topics = cls._query_topics(query_terms)
        query_entities = cls._extract_astro_entities(query)
        chart_entities = cls._chart_entities_for_query(query_entities, chart_context)
        query_vector = embedding_provider.embed_text(query)
        action_domains = cls.ACTION_DOMAIN_HINTS.get(action or "", set())
        allowed_domains = domains or action_domains
        query_phrase = cls._query_phrase(query_terms)

        candidates: list[RetrievalCandidate] = []
        for document in documents:
            document_domain = str(document.metadata.get("domain") or "")
            if allowed_domains and document_domain not in allowed_domains:
                continue

            haystack = f"{document.title}\n{document.content}".lower()
            lexical_score = sum(haystack.count(term) for term in query_terms)
            title_score = 0
            lowered_title = document.title.lower()
            if query_phrase:
                if query_phrase in lowered_title:
                    title_score += 3
                elif query_phrase in haystack:
                    title_score += 1
            semantic_score = max(0.0, _cosine_similarity(query_vector, document.vector))
            action_score = (
                4
                if action is not None and action in document.metadata.get("action_hints", [])
                else 0
            )
            topic_score = len(query_topics & set(document.metadata.get("topic_hints", []))) * 2
            document_entities = document.metadata.get("astro_entities") or {}
            entity_score = cls._astro_entity_score(query_entities, document_entities)
            chart_score = cls._chart_match_score(chart_entities, document_entities)
            for topic in query_topics:
                astro_expansion = cls.TOPIC_ASTRO_EXPANSIONS.get(topic)
                if not astro_expansion:
                    continue
                if set(document_entities.get("houses", [])) & set(astro_expansion["houses"]):
                    topic_score += 2
                if set(document_entities.get("planets", [])) & set(
                    astro_expansion["terms"]
                ):
                    topic_score += 1
            chunk_type_score = cls._chunk_type_match_score(
                query_entities,
                str(document.metadata.get("type") or ""),
            )
            domain_score = 2 if document_domain in action_domains and action_domains else 0

            if (
                lexical_score <= 0
                and title_score <= 0
                and semantic_score <= 0
                and action_score <= 0
                and topic_score <= 0
                and domain_score <= 0
                and entity_score <= 0
                and chunk_type_score <= 0
                and chart_score <= 0
            ):
                continue

            candidates.append(
                RetrievalCandidate(
                    document=document,
                    lexical_score=lexical_score,
                    semantic_score=semantic_score,
                    title_score=title_score,
                    action_score=action_score,
                    topic_score=topic_score,
                    domain_score=domain_score,
                    entity_score=entity_score,
                    chunk_type_score=chunk_type_score,
                    chart_score=chart_score,
                )
            )

        return candidates, action_domains

    def _rerank_candidates(
        self,
        candidates: list[RetrievalCandidate],
        *,
        query: str,
        query_terms: list[str],
        top_k: int,
    ) -> list[RetrievalHit]:
        if not candidates:
            return []

        candidate_limit = self._candidate_limit(top_k)
        lexical_ranked = sorted(
            (
                candidate
                for candidate in candidates
                if candidate.lexical_score > 0 or candidate.title_score > 0
            ),
            key=lambda candidate: (
                candidate.lexical_score + candidate.title_score,
                candidate.document.title,
            ),
            reverse=True,
        )
        semantic_ranked = sorted(
            (candidate for candidate in candidates if candidate.semantic_score > 0),
            key=lambda candidate: (candidate.semantic_score, candidate.document.title),
            reverse=True,
        )
        support_ranked = sorted(
            (
                candidate
                for candidate in candidates
                if candidate.action_score > 0
                or candidate.topic_score > 0
                or candidate.domain_score > 0
                or candidate.entity_score > 0
                or candidate.chunk_type_score > 0
                or candidate.chart_score > 0
            ),
            key=lambda candidate: (
                candidate.action_score
                + candidate.topic_score
                + candidate.domain_score
                + candidate.entity_score
                + candidate.chunk_type_score
                + candidate.chart_score,
                candidate.document.title,
            ),
            reverse=True,
        )

        selected: dict[tuple[str, Any, str], RetrievalCandidate] = {}
        lexical_ranks: dict[tuple[str, Any, str], int] = {}
        semantic_ranks: dict[tuple[str, Any, str], int] = {}
        support_ranks: dict[tuple[str, Any, str], int] = {}

        for rank, candidate in enumerate(lexical_ranked[:candidate_limit], start=1):
            lexical_ranks[candidate.key] = rank
            selected[candidate.key] = candidate
        for rank, candidate in enumerate(semantic_ranked[:candidate_limit], start=1):
            semantic_ranks[candidate.key] = rank
            selected[candidate.key] = candidate
        for rank, candidate in enumerate(support_ranked[:candidate_limit], start=1):
            support_ranks[candidate.key] = rank
            selected[candidate.key] = candidate

        base_scores: dict[str, float] = {}
        selected_items: list[RerankItem] = []
        for candidate in selected.values():
            fusion_score = 0.0
            if candidate.key in lexical_ranks:
                fusion_score += 1.0 / (50 + lexical_ranks[candidate.key])
            if candidate.key in semantic_ranks:
                fusion_score += 1.0 / (50 + semantic_ranks[candidate.key])
            if candidate.key in support_ranks:
                fusion_score += 1.0 / (70 + support_ranks[candidate.key])

            score = (
                fusion_score * 200
                + candidate.lexical_score
                + candidate.title_score
                + candidate.action_score
                + candidate.topic_score
                + candidate.domain_score
                + candidate.entity_score
                + candidate.chunk_type_score
                + candidate.chart_score
                + candidate.semantic_score * 3
            )

            rerank_id = candidate.rerank_id
            rounded_score = round(score, 4)
            base_scores[rerank_id] = rounded_score
            selected_items.append(
                RerankItem(
                    item_id=rerank_id,
                    title=candidate.document.title,
                    text=candidate.document.content,
                    metadata={
                        **candidate.document.metadata,
                        "base_score": rounded_score,
                        "semantic_score": round(candidate.semantic_score, 4),
                        "lexical_score": candidate.lexical_score,
                        "title_score": candidate.title_score,
                        "entity_score": candidate.entity_score,
                        "chunk_type_score": candidate.chunk_type_score,
                        "chart_score": candidate.chart_score,
                    },
                )
            )

        rerank_scores: dict[str, tuple[float, int]] = {}
        try:
            rerank_results = self.reranker.rerank(query, selected_items, top_k=len(selected_items))
            for result in rerank_results:
                rerank_scores[result.item_id] = (result.score, result.rank)
        except Exception:
            rerank_scores = {}

        scored_hits: list[RetrievalHit] = []
        for item in selected_items:
            candidate_score = base_scores[item.item_id]
            rerank_score, rerank_rank = rerank_scores.get(item.item_id, (0.0, 0))
            final_score = candidate_score + rerank_score * 10.0
            metadata = dict(item.metadata)
            metadata["hybrid_score"] = candidate_score
            metadata["reranker_score"] = round(rerank_score, 4)
            metadata["reranker_rank"] = rerank_rank
            metadata["reranker_provider"] = self.reranker.provider_name
            metadata["reranker_model"] = self.reranker.model_name
            scored_hits.append(
                RetrievalHit(
                    title=item.title,
                    excerpt=self._excerpt_for_query(item.text, query_terms),
                    path=str(item.metadata.get("path") or item.item_id.split("::", 1)[0]),
                    score=round(final_score, 4),
                    metadata=metadata,
                )
            )

        scored_hits.sort(key=lambda hit: (hit.score, hit.title), reverse=True)
        return scored_hits[:top_k]

    @classmethod
    def _merge_ranked_matches(
        cls,
        *match_groups: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        deduped: dict[tuple[str, Any, str, str], dict[str, Any]] = {}
        for group in match_groups:
            for match in group:
                metadata = match.get("metadata") or {}
                key = (
                    str(match.get("path") or ""),
                    metadata.get("chunk_index"),
                    str(match.get("title") or match.get("source") or ""),
                    str(match.get("excerpt") or match.get("text") or ""),
                )
                existing = deduped.get(key)
                existing_score = float(existing.get("score", 0)) if existing is not None else float("-inf")
                match_score = float(match.get("score", 0))
                if existing is None or match_score > existing_score:
                    deduped[key] = match

        merged = sorted(
            deduped.values(),
            key=lambda match: (
                float(match.get("score", 0)),
                float((match.get("metadata") or {}).get("semantic_score", 0)),
                str(match.get("title") or match.get("source") or ""),
            ),
            reverse=True,
        )
        return merged[:top_k]

    def retrieve(
        self,
        query: str,
        top_k: int,
        *,
        action: str | None = None,
        domains: set[str] | None = None,
        chart_context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        query_variants = self._expand_astrology_query(query, chart_context=chart_context)
        corpus = self._resolve_corpus(query_seeds=query_variants, top_k=top_k)
        match_groups = [
            self._retrieve_from_corpus(
                variant,
                top_k,
                action=action,
                domains=domains,
                corpus=corpus,
                chart_context=chart_context,
            )
            for variant in query_variants
        ]
        return self._merge_ranked_matches(*match_groups, top_k=top_k)

    def _retrieve_from_corpus(
        self,
        query: str,
        top_k: int,
        *,
        action: str | None = None,
        domains: set[str] | None = None,
        corpus: RetrievalCorpus,
        chart_context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        query_terms = normalize_embedding_tokens(query)
        candidates, _action_domains = self._build_candidates(
            query=query,
            action=action,
            domains=domains,
            documents=corpus.documents,
            embedding_provider=self.embedding_provider,
            chart_context=chart_context,
        )
        scored_hits = self._rerank_candidates(
            candidates,
            query=query,
            query_terms=query_terms,
            top_k=top_k,
        )
        return [
            {
                "title": hit.title,
                "excerpt": hit.excerpt,
                "path": hit.path,
                "score": hit.score,
                "metadata": {
                    **hit.metadata,
                    "retrieval_provider": corpus.metadata.get("provider"),
                    "retrieval_strategy": corpus.metadata.get("retrieval_strategy"),
                    "embedding_provider": corpus.metadata.get("embedding_provider"),
                    "embedding_model": corpus.metadata.get("embedding_model"),
                    "vector_backend": corpus.metadata.get("vector_backend"),
                    "keyword_backend": corpus.metadata.get("keyword_backend"),
                },
            }
            for hit in scored_hits[:top_k]
        ]

    @classmethod
    def _knowledge_domains_for_action(cls, action: str | None) -> set[str]:
        hinted = set(cls.ACTION_DOMAIN_HINTS.get(action or "", set()))
        if not hinted:
            return set(cls.KNOWLEDGE_DOMAINS)
        knowledge_domains = hinted & cls.KNOWLEDGE_DOMAINS
        return knowledge_domains or {"general_guidance"}

    @classmethod
    def _policy_domains_for_action(cls, action: str | None) -> set[str]:
        hinted = set(cls.ACTION_DOMAIN_HINTS.get(action or "", set()))
        return hinted & cls.POLICY_DOMAINS

    @staticmethod
    def _annotate_bucket(
        matches: list[dict[str, Any]],
        *,
        bucket: str,
    ) -> list[dict[str, Any]]:
        annotated: list[dict[str, Any]] = []
        for match in matches:
            annotated.append({**match, "bucket": bucket})
        return annotated

    @staticmethod
    def _dedupe_matches(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deduped: list[dict[str, Any]] = []
        seen: set[tuple[str, Any, str, str]] = set()
        for match in matches:
            metadata = match.get("metadata") or {}
            key = (
                str(match.get("path") or ""),
                metadata.get("chunk_index"),
                str(match.get("title") or match.get("source") or ""),
                str(match.get("excerpt") or match.get("text") or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(match)
        return deduped

    def retrieve_context(
        self,
        query: str,
        top_k: int,
        *,
        action: str | None = None,
        domains: set[str] | None = None,
        chart_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        domain_key = ",".join(sorted(domains or []))
        chart_key = self._chart_context_cache_key(chart_context)
        cache_key = (query.strip().lower(), top_k, action, f"{domain_key}|{chart_key}")
        cached = self._query_cache.get(cache_key)
        now = time.time()
        if cached is not None and now - cached[0] < 120:
            return cached[1]

        query_variants = self._expand_astrology_query(query, chart_context=chart_context)
        corpus = self._resolve_corpus(query_seeds=query_variants, top_k=top_k)
        match_groups = [
            self._retrieve_from_corpus(
                variant,
                top_k,
                action=action,
                domains=domains,
                corpus=corpus,
                chart_context=chart_context,
            )
            for variant in query_variants
        ]
        matches = self._merge_ranked_matches(*match_groups, top_k=top_k)
        payload = {
            "retrieval_metadata": self._retrieval_metadata(corpus.metadata, chart_context=chart_context),
            "chunks": [
                {
                    "text": match["excerpt"],
                    "source": match["title"],
                    "type": match.get("metadata", {}).get("type", "explanation"),
                    "risk": match.get("metadata", {}).get("risk", "low"),
                    "allowed_actions": match.get("metadata", {}).get("allowed_actions", ["explain_only"]),
                    "metadata": match.get("metadata", {}),
                    "score": match.get("score"),
                    "path": match.get("path"),
                }
                for match in matches
            ]
        }
        self._query_cache[cache_key] = (now, payload)
        return payload

    def retrieve_context_bundle(
        self,
        query: str,
        top_k: int,
        *,
        action: str | None = None,
        planner_query: str | None = None,
        chart_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized_query = query.strip()
        normalized_planner_query = (planner_query or "").strip()
        knowledge_queries = self._expand_astrology_query(
            normalized_query,
            chart_context=chart_context,
        )
        planner_queries = (
            self._expand_astrology_query(normalized_planner_query, chart_context=chart_context)
            if normalized_planner_query
            else []
        )
        corpus = self._resolve_corpus(
            query_seeds=[
                seed
                for seed in [*knowledge_queries, *planner_queries]
                if seed
            ],
            top_k=top_k,
        )
        knowledge_domains = self._knowledge_domains_for_action(action)
        knowledge_match_groups = [
            self._retrieve_from_corpus(
                knowledge_query,
                top_k,
                action=action,
                domains=knowledge_domains,
                corpus=corpus,
                chart_context=chart_context,
            )
            for knowledge_query in knowledge_queries
        ]
        if planner_queries and normalized_planner_query != normalized_query:
            knowledge_match_groups.extend(
                self._retrieve_from_corpus(
                    planner_query_variant,
                    top_k,
                    action=action,
                    domains=knowledge_domains,
                    corpus=corpus,
                    chart_context=chart_context,
                )
                for planner_query_variant in planner_queries
            )
        knowledge_matches = self._merge_ranked_matches(*knowledge_match_groups, top_k=top_k)
        if not knowledge_matches and knowledge_domains != self.KNOWLEDGE_DOMAINS:
            knowledge_matches = self._retrieve_from_corpus(
                normalized_query,
                top_k,
                action=action,
                domains=set(self.KNOWLEDGE_DOMAINS),
                corpus=corpus,
                chart_context=chart_context,
            )

        policy_matches: list[dict[str, Any]] = []
        policy_domains = self._policy_domains_for_action(action)
        if policy_domains:
            policy_queries = planner_queries or knowledge_queries
            policy_match_groups = [
                self._retrieve_from_corpus(
                    policy_query,
                    top_k,
                    action=action,
                    domains=policy_domains,
                    corpus=corpus,
                    chart_context=chart_context,
                )
                for policy_query in policy_queries
            ]
            if normalized_planner_query and normalized_planner_query != normalized_query:
                policy_match_groups.extend(
                    self._retrieve_from_corpus(
                        knowledge_query,
                        top_k,
                        action=action,
                        domains=policy_domains,
                        corpus=corpus,
                        chart_context=chart_context,
                    )
                    for knowledge_query in knowledge_queries
                )
            policy_matches = self._merge_ranked_matches(*policy_match_groups, top_k=top_k)
            if not policy_matches and policy_domains != self.POLICY_DOMAINS:
                policy_matches = self._retrieve_from_corpus(
                    normalized_query,
                    top_k,
                    action=action,
                    domains=set(self.POLICY_DOMAINS),
                    corpus=corpus,
                    chart_context=chart_context,
                )

        annotated_knowledge = self._annotate_bucket(knowledge_matches, bucket="knowledge")
        annotated_policy = self._annotate_bucket(policy_matches, bucket="policy")
        combined = self._dedupe_matches([*annotated_policy, *annotated_knowledge])
        corpus_metadata = corpus.metadata

        return {
            "retrieval_metadata": self._retrieval_metadata(corpus_metadata, chart_context=chart_context),
            "chunks": combined,
            "knowledge_chunks": annotated_knowledge,
            "policy_chunks": annotated_policy,
        }
