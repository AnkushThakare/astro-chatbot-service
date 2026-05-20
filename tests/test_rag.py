import pytest

from src.core.embeddings import build_local_embedding_vector
from src.core.rag import RAGService, RetrievalDocument, clear_documents_cache
from src.db.models import Embedding
from src.db.session import SessionLocal
from sqlalchemy import delete


def test_rag_prefers_action_aligned_product_doc() -> None:
    service = RAGService()
    original_documents = RAGService._documents

    docs = [
        RetrievalDocument(
            title="common astrology concepts",
            path="common.txt",
            content="Career growth depends on discipline, timing, and communication.",
            metadata={
                "domain": "general_guidance",
                "action_hints": ["respond_only"],
                "topic_hints": ["career"],
                "source_type": "knowledge_base",
                "language": "english",
            },
            vector=build_local_embedding_vector(
                "Career growth depends on discipline, timing, and communication."
            ),
        ),
        RetrievalDocument(
            title="rudraksha guide",
            path="rudraksha.txt",
            content="Rudraksha options are discussed for focus, stability, and career support.",
            metadata={
                "domain": "product_policy",
                "action_hints": ["recommend_product", "respond_only"],
                "topic_hints": ["career", "remedies"],
                "source_type": "knowledge_base",
                "language": "english",
            },
            vector=build_local_embedding_vector(
                "Rudraksha options are discussed for focus, stability, and career support."
            ),
        ),
    ]

    try:
        clear_documents_cache()
        RAGService._documents = classmethod(lambda cls: docs)  # type: ignore[method-assign]
        hits = service.retrieve("career support", top_k=2, action="recommend_product")
    finally:
        clear_documents_cache()
        RAGService._documents = original_documents  # type: ignore[method-assign]

    assert hits[0]["title"] == "rudraksha guide"
    assert hits[0]["metadata"]["domain"] == "product_policy"


def test_rag_matches_semantic_synonyms_for_career_queries() -> None:
    service = RAGService()
    original_documents = RAGService._documents

    docs = [
        RetrievalDocument(
            title="career pressure note",
            path="career.txt",
            content="Career pressure often rises when work feels directionless and progress slows down.",
            metadata={
                "domain": "general_guidance",
                "action_hints": ["respond_only"],
                "topic_hints": ["career"],
                "source_type": "knowledge_base",
                "language": "english",
            },
            vector=build_local_embedding_vector(
                "Career pressure often rises when work feels directionless and progress slows down."
            ),
        ),
        RetrievalDocument(
            title="relationship note",
            path="love.txt",
            content="Relationship distance can come from unclear expectations and trust issues.",
            metadata={
                "domain": "general_guidance",
                "action_hints": ["respond_only"],
                "topic_hints": ["love"],
                "source_type": "knowledge_base",
                "language": "english",
            },
            vector=build_local_embedding_vector(
                "Relationship distance can come from unclear expectations and trust issues."
            ),
        ),
    ]

    try:
        clear_documents_cache()
        RAGService._documents = classmethod(lambda cls: docs)  # type: ignore[method-assign]
        hits = service.retrieve("job stress", top_k=2, action="respond_only")
    finally:
        clear_documents_cache()
        RAGService._documents = original_documents  # type: ignore[method-assign]

    assert hits
    assert hits[0]["title"] == "career pressure note"


def test_rag_filters_to_booking_domains_for_booking_action() -> None:
    service = RAGService()
    original_documents = RAGService._documents

    docs = [
        RetrievalDocument(
            title="rudraksha guide",
            path="rudraksha.txt",
            content="Rudraksha options are discussed for focus and stability.",
            metadata={
                "domain": "product_policy",
                "action_hints": ["recommend_product", "respond_only"],
                "topic_hints": ["remedies"],
                "source_type": "knowledge_base",
                "language": "english",
            },
            vector=build_local_embedding_vector(
                "Rudraksha options are discussed for focus and stability."
            ),
        ),
        RetrievalDocument(
            title="temple puja guide",
            path="booking.txt",
            content="Temple puja bookings usually depend on service type and location.",
            metadata={
                "domain": "booking_guidance",
                "action_hints": ["book_pooja", "suggest_consultant", "respond_only"],
                "topic_hints": ["booking"],
                "source_type": "knowledge_base",
                "language": "english",
            },
            vector=build_local_embedding_vector(
                "Temple puja bookings usually depend on service type and location."
            ),
        ),
    ]

    try:
        clear_documents_cache()
        RAGService._documents = classmethod(lambda cls: docs)  # type: ignore[method-assign]
        hits = service.retrieve("book puja", top_k=2, action="book_pooja")
    finally:
        clear_documents_cache()
        RAGService._documents = original_documents  # type: ignore[method-assign]

    assert hits
    assert all(hit["metadata"]["domain"] == "booking_guidance" for hit in hits)


def test_rag_bundle_separates_policy_and_knowledge_for_product_action() -> None:
    service = RAGService()
    original_documents = RAGService._documents

    docs = [
        RetrievalDocument(
            title="career note",
            path="career.txt",
            content="Career growth needs patience, discipline, and timing.",
            metadata={
                "domain": "general_guidance",
                "action_hints": ["respond_only", "recommend_product"],
                "topic_hints": ["career"],
                "source_type": "knowledge_base",
                "language": "english",
            },
            vector=build_local_embedding_vector(
                "Career growth needs patience, discipline, and timing."
            ),
        ),
        RetrievalDocument(
            title="rudraksha policy",
            path="products.txt",
            content="Recommend catalog rudraksha options only when the user asks for a remedy or support item.",
            metadata={
                "domain": "product_policy",
                "action_hints": ["recommend_product"],
                "topic_hints": ["career", "remedies"],
                "source_type": "knowledge_base",
                "language": "english",
            },
            vector=build_local_embedding_vector(
                "Recommend catalog rudraksha options only when the user asks for a remedy or support item."
            ),
        ),
    ]

    try:
        clear_documents_cache()
        RAGService._documents = classmethod(lambda cls: docs)  # type: ignore[method-assign]
        payload = service.retrieve_context_bundle(
            "career confusion",
            top_k=3,
            action="recommend_product",
            planner_query="rudraksha career support",
        )
    finally:
        clear_documents_cache()
        RAGService._documents = original_documents  # type: ignore[method-assign]

    assert payload["knowledge_chunks"]
    assert payload["policy_chunks"]
    assert payload["knowledge_chunks"][0]["bucket"] == "knowledge"
    assert payload["policy_chunks"][0]["bucket"] == "policy"
    assert payload["policy_chunks"][0]["metadata"]["domain"] == "product_policy"
    assert payload["knowledge_chunks"][0]["metadata"]["domain"] == "general_guidance"


def test_rag_context_cache_respects_domain_filters() -> None:
    service = RAGService()
    original_documents = RAGService._documents
    original_cache = dict(RAGService._query_cache)
    RAGService._query_cache.clear()

    docs = [
        RetrievalDocument(
            title="career note",
            path="career.txt",
            content="Career growth needs patience and timing.",
            metadata={
                "domain": "general_guidance",
                "action_hints": ["respond_only", "recommend_product"],
                "topic_hints": ["career"],
                "source_type": "knowledge_base",
                "language": "english",
            },
            vector=build_local_embedding_vector("Career growth needs patience and timing."),
        ),
        RetrievalDocument(
            title="product note",
            path="product.txt",
            content="Recommend only catalog rudraksha products for remedy requests.",
            metadata={
                "domain": "product_policy",
                "action_hints": ["recommend_product"],
                "topic_hints": ["remedies"],
                "source_type": "knowledge_base",
                "language": "english",
            },
            vector=build_local_embedding_vector(
                "Recommend only catalog rudraksha products for remedy requests."
            ),
        ),
    ]

    try:
        clear_documents_cache()
        RAGService._documents = classmethod(lambda cls: docs)  # type: ignore[method-assign]
        knowledge_payload = service.retrieve_context(
            "career support",
            2,
            action="recommend_product",
            domains={"general_guidance"},
        )
        policy_payload = service.retrieve_context(
            "career support",
            2,
            action="recommend_product",
            domains={"product_policy"},
        )
    finally:
        clear_documents_cache()
        RAGService._documents = original_documents  # type: ignore[method-assign]
        RAGService._query_cache.clear()
        RAGService._query_cache.update(original_cache)

    assert knowledge_payload["chunks"][0]["metadata"]["domain"] == "general_guidance"
    assert policy_payload["chunks"][0]["metadata"]["domain"] == "product_policy"


def test_rag_bundle_uses_planner_query_to_recover_knowledge_matches() -> None:
    service = RAGService()
    original_documents = RAGService._documents

    docs = [
        RetrievalDocument(
            title="saturn career note",
            path="saturn.txt",
            content="Saturn in career periods can bring delay, duty, and slow but steady growth.",
            metadata={
                "domain": "general_guidance",
                "action_hints": ["respond_only"],
                "topic_hints": ["career"],
                "source_type": "knowledge_base",
                "language": "english",
            },
            vector=build_local_embedding_vector(
                "Saturn in career periods can bring delay, duty, and slow but steady growth."
            ),
        ),
    ]

    try:
        clear_documents_cache()
        RAGService._documents = classmethod(lambda cls: docs)  # type: ignore[method-assign]
        payload = service.retrieve_context_bundle(
            "what should i do",
            top_k=2,
            action="respond_only",
            planner_query="saturn career delay",
        )
    finally:
        clear_documents_cache()
        RAGService._documents = original_documents  # type: ignore[method-assign]

    assert payload["knowledge_chunks"]
    assert payload["knowledge_chunks"][0]["title"] == "saturn career note"


def test_rag_bundle_falls_back_to_user_query_for_policy_matches() -> None:
    service = RAGService()
    original_documents = RAGService._documents

    docs = [
        RetrievalDocument(
            title="booking policy",
            path="booking.txt",
            content="Temple puja bookings should use catalog service listings and location filters.",
            metadata={
                "domain": "booking_guidance",
                "action_hints": ["book_pooja", "suggest_consultant", "respond_only"],
                "topic_hints": ["booking"],
                "source_type": "knowledge_base",
                "language": "english",
            },
            vector=build_local_embedding_vector(
                "Temple puja bookings should use catalog service listings and location filters."
            ),
        ),
    ]

    try:
        clear_documents_cache()
        RAGService._documents = classmethod(lambda cls: docs)  # type: ignore[method-assign]
        payload = service.retrieve_context_bundle(
            "book temple puja near me",
            top_k=2,
            action="book_pooja",
            planner_query="spiritual support",
        )
    finally:
        clear_documents_cache()
        RAGService._documents = original_documents  # type: ignore[method-assign]

    assert payload["policy_chunks"]
    assert payload["policy_chunks"][0]["title"] == "booking policy"


def test_rag_prefers_embedding_store_when_available() -> None:
    db = SessionLocal()
    try:
        db.execute(delete(Embedding))
        db.commit()
        db.add(
            Embedding(
                source_type="astrology_text",
                source_id="rudraksha_guide.txt:chunk_0",
                content="Rudraksha is usually discussed for focus, steadiness, and remedy support.",
                embedding_model="local-hash-v1",
                vector_json="[0.0, 1.0, 0.0]",
            )
        )
        db.commit()

        service = RAGService(db)
        payload = service.retrieve_context_bundle(
            "rudraksha focus",
            top_k=2,
            action="recommend_product",
            planner_query="rudraksha support item",
        )
    finally:
        db.execute(delete(Embedding))
        db.commit()
        db.close()

    assert payload["retrieval_metadata"]["provider"] == "embedding_store"
    assert payload["retrieval_metadata"]["retrieval_strategy"] == "db_embedding_hybrid_v3"
    assert payload["retrieval_metadata"]["embedding_provider"] == "precomputed"
    assert payload["retrieval_metadata"]["embedding_model"] == "local-hash-v1"
    assert payload["retrieval_metadata"]["vector_backend"] == "json_scan"
    assert payload["retrieval_metadata"]["keyword_backend"] == "keyword_scan"
    assert payload["retrieval_metadata"]["reranker_provider"] == "heuristic"
    assert payload["retrieval_metadata"]["reranker_model"] == "heuristic-v1"
    assert payload["policy_chunks"]
    assert payload["policy_chunks"][0]["metadata"]["embedding_provider"] == "precomputed"
    assert payload["policy_chunks"][0]["metadata"]["vector_backend"] == "json_scan"
    assert payload["policy_chunks"][0]["metadata"]["keyword_backend"] == "keyword_scan"
    assert payload["policy_chunks"][0]["metadata"]["reranker_provider"] == "heuristic"
    assert payload["policy_chunks"][0]["metadata"]["retrieval_provider"] == "embedding_store"


def test_rag_ignores_embeddings_from_other_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.core.embeddings.settings.RAG_EMBEDDING_PROVIDER", "local_hash")
    monkeypatch.setattr("src.core.embeddings.settings.RAG_EMBEDDING_MODEL", "active-model-v1")
    monkeypatch.setattr("src.core.rag.settings.RAG_EMBEDDING_PROVIDER", "local_hash")
    monkeypatch.setattr("src.core.rag.settings.RAG_EMBEDDING_MODEL", "active-model-v1")

    original_documents = RAGService._documents
    db = SessionLocal()
    try:
        clear_documents_cache()
        RAGService._documents = classmethod(lambda cls: [])  # type: ignore[method-assign]
        db.execute(delete(Embedding))
        db.commit()
        db.add_all(
            [
                Embedding(
                    source_type="astrology_text",
                    source_id="career_note.txt:chunk_0",
                    content="Career guidance needs patience and steady work.",
                    embedding_model="active-model-v1",
                    vector_json="[1.0, 0.0, 0.0]",
                ),
                Embedding(
                    source_type="astrology_text",
                    source_id="other_model_note.txt:chunk_0",
                    content="Rudraksha catalog support for remedies.",
                    embedding_model="legacy-model-v0",
                    vector_json="[0.0, 1.0, 0.0]",
                ),
            ]
        )
        db.commit()

        service = RAGService(db)
        service.embedding_provider = type(
            "ProviderStub",
            (),
            {
                "model_name": "active-model-v1",
                "provider_name": "local_hash",
                "embed_text": staticmethod(lambda text: [1.0, 0.0, 0.0]),
            },
        )()

        payload = service.retrieve_context_bundle(
            "career guidance",
            top_k=2,
            action="respond_only",
            planner_query="career work support",
        )
    finally:
        clear_documents_cache()
        RAGService._documents = original_documents  # type: ignore[method-assign]
        db.execute(delete(Embedding))
        db.commit()
        db.close()

    assert payload["retrieval_metadata"]["provider"] == "embedding_store"
    assert payload["retrieval_metadata"]["embedding_model"] == "active-model-v1"
    assert payload["retrieval_metadata"]["document_count"] == 1
    assert payload["chunks"]
    assert all(
        chunk["metadata"]["embedding_model"] == "active-model-v1"
        for chunk in payload["chunks"]
    )
    assert all("other_model_note" not in chunk["path"] for chunk in payload["chunks"])


def test_rag_merges_filesystem_fallback_when_embedding_store_is_partial() -> None:
    db = SessionLocal()
    original_documents = RAGService._documents
    try:
        db.execute(delete(Embedding))
        db.commit()
        db.add(
            Embedding(
                source_type="astrology_text",
                source_id="db_only.txt:chunk_0",
                content="Database-backed career guidance chunk.",
                embedding_model="local-hash-v1",
                vector_json="[1.0, 0.0, 0.0]",
            )
        )
        db.commit()

        clear_documents_cache()
        RAGService._documents = classmethod(  # type: ignore[method-assign]
            lambda cls: [
                RetrievalDocument(
                    title="db only",
                    path="db_only.txt",
                    content="Filesystem version of the same DB chunk.",
                    metadata={
                        "domain": "general_guidance",
                        "action_hints": ["respond_only"],
                        "topic_hints": ["career"],
                        "source_type": "knowledge_base",
                        "language": "english",
                        "chunk_index": 0,
                    },
                    vector=build_local_embedding_vector("Filesystem version of the same DB chunk."),
                ),
                RetrievalDocument(
                    title="file only",
                    path="file_only.txt",
                    content="Filesystem-only fallback guidance for patience and timing.",
                    metadata={
                        "domain": "general_guidance",
                        "action_hints": ["respond_only"],
                        "topic_hints": ["career"],
                        "source_type": "knowledge_base",
                        "language": "english",
                        "chunk_index": 0,
                    },
                    vector=build_local_embedding_vector(
                        "Filesystem-only fallback guidance for patience and timing."
                    ),
                ),
            ]
        )

        service = RAGService(db)
        payload = service.retrieve_context_bundle(
            "career guidance",
            top_k=4,
            action="respond_only",
            planner_query="career patience timing",
        )
    finally:
        clear_documents_cache()
        RAGService._documents = original_documents  # type: ignore[method-assign]
        db.execute(delete(Embedding))
        db.commit()
        db.close()

    assert payload["retrieval_metadata"]["provider"] == "embedding_store"
    assert payload["retrieval_metadata"]["db_document_count"] == 1
    assert payload["retrieval_metadata"]["filesystem_fallback_count"] == 1
    assert payload["retrieval_metadata"]["document_count"] == 2
    assert any(chunk["path"] == "db_only.txt" for chunk in payload["chunks"])
    assert any(chunk["path"] == "file_only.txt" for chunk in payload["chunks"])


def test_infer_metadata_extracts_astrology_entities_and_chunk_type() -> None:
    metadata = RAGService._infer_metadata(
        "saturn_career.txt",
        "Saturn Career Note",
        "Saturn in the 10th house during Jupiter mahadasha can bring slow but steady career rise.",
    )

    assert metadata["type"] == "dasha_period"
    assert metadata["astro_entities"]["planets"] == ["jupiter", "saturn"]
    assert metadata["astro_entities"]["houses"] == [10]
    assert "career" in metadata["topic_hints"]
    assert metadata["source_citation"] == "Saturn Career Note"


def test_rag_prefers_exact_planet_house_match_for_astrology_query() -> None:
    service = RAGService()
    original_documents = RAGService._documents

    saturn_content = "Saturn in the 10th house often brings duty, delay, and long-term career stability."
    jupiter_content = "Jupiter in the 10th house often brings guidance, growth, and reputation in profession."
    docs = [
        RetrievalDocument(
            title="saturn tenth house",
            path="saturn_10.txt",
            content=saturn_content,
            metadata=RAGService._infer_metadata("saturn_10.txt", "saturn tenth house", saturn_content),
            vector=build_local_embedding_vector(saturn_content),
        ),
        RetrievalDocument(
            title="jupiter tenth house",
            path="jupiter_10.txt",
            content=jupiter_content,
            metadata=RAGService._infer_metadata("jupiter_10.txt", "jupiter tenth house", jupiter_content),
            vector=build_local_embedding_vector(jupiter_content),
        ),
    ]

    try:
        clear_documents_cache()
        RAGService._documents = classmethod(lambda cls: docs)  # type: ignore[method-assign]
        hits = service.retrieve("What does Saturn in 10th house mean for career?", top_k=2, action="respond_only")
    finally:
        clear_documents_cache()
        RAGService._documents = original_documents  # type: ignore[method-assign]

    assert hits
    assert hits[0]["title"] == "saturn tenth house"
    assert hits[0]["metadata"]["entity_score"] > hits[1]["metadata"]["entity_score"]


def test_rag_expands_career_query_to_tenth_house_context() -> None:
    service = RAGService()
    original_documents = RAGService._documents

    career_house_content = "The 10th house governs profession, status, public work, and career changes."
    unrelated_content = "The 4th house relates to home, mother, and domestic comfort."
    docs = [
        RetrievalDocument(
            title="tenth house profession",
            path="career_house.txt",
            content=career_house_content,
            metadata=RAGService._infer_metadata("career_house.txt", "tenth house profession", career_house_content),
            vector=build_local_embedding_vector(career_house_content),
        ),
        RetrievalDocument(
            title="fourth house home",
            path="home_house.txt",
            content=unrelated_content,
            metadata=RAGService._infer_metadata("home_house.txt", "fourth house home", unrelated_content),
            vector=build_local_embedding_vector(unrelated_content),
        ),
    ]

    try:
        clear_documents_cache()
        RAGService._documents = classmethod(lambda cls: docs)  # type: ignore[method-assign]
        hits = service.retrieve("Will I get a job change soon?", top_k=2, action="respond_only")
    finally:
        clear_documents_cache()
        RAGService._documents = original_documents  # type: ignore[method-assign]

    assert hits
    assert hits[0]["title"] == "tenth house profession"


def test_rag_uses_chart_context_to_prefer_chart_aligned_match() -> None:
    service = RAGService()
    original_documents = RAGService._documents

    saturn_content = "Saturn in the 10th house can bring disciplined career growth after delays."
    venus_content = "Venus in the 7th house supports relationship harmony and attraction."
    docs = [
        RetrievalDocument(
            title="saturn tenth house",
            path="saturn_10.txt",
            content=saturn_content,
            metadata=RAGService._infer_metadata("saturn_10.txt", "saturn tenth house", saturn_content),
            vector=build_local_embedding_vector(saturn_content),
        ),
        RetrievalDocument(
            title="venus seventh house",
            path="venus_7.txt",
            content=venus_content,
            metadata=RAGService._infer_metadata("venus_7.txt", "venus seventh house", venus_content),
            vector=build_local_embedding_vector(venus_content),
        ),
    ]
    chart_context = {
        "ascendant_sign": "Aries",
        "moon_sign": "Cancer",
        "current_mahadasha": "Saturn",
        "current_antardasha": "Jupiter",
        "placements": [
            {"planet": "saturn", "house": 10, "sign": "capricorn"},
            {"planet": "venus", "house": 7, "sign": "libra"},
        ],
        "astro_entities": {
            "planets": ["saturn", "venus"],
            "houses": [7, 10],
            "signs": ["capricorn", "libra"],
            "nakshatras": [],
            "dashas": ["saturn", "jupiter"],
        },
    }

    try:
        clear_documents_cache()
        RAGService._documents = classmethod(lambda cls: docs)  # type: ignore[method-assign]
        hits = service.retrieve(
            "Will I get a job change soon?",
            top_k=2,
            action="respond_only",
            chart_context=chart_context,
        )
    finally:
        clear_documents_cache()
        RAGService._documents = original_documents  # type: ignore[method-assign]

    assert hits
    assert hits[0]["title"] == "saturn tenth house"
    assert hits[0]["metadata"]["chart_score"] >= hits[1]["metadata"]["chart_score"]
