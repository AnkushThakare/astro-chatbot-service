from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from astro_chatbot_service.models.schemas import (
    KnowledgeDocumentBatchCreate,
    RetrievalResponse,
    SearchRequest,
)
from astro_chatbot_service.services.rag import RAGService

router = APIRouter(prefix="/retrieval")


@router.post("/documents")
def ingest_documents(
    request: KnowledgeDocumentBatchCreate,
    db: Session = Depends(get_db),
) -> dict[str, int]:
    service = RAGService(db)
    created = service.ingest(request.documents)
    return {"created": created}


@router.post("/search", response_model=RetrievalResponse)
def search_documents(
    request: SearchRequest,
    db: Session = Depends(get_db),
) -> RetrievalResponse:
    service = RAGService(db)
    matches = service.retrieve(request.query, request.top_k)
    return RetrievalResponse(matches=matches)

