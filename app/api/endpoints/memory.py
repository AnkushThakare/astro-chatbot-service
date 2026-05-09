from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from astro_chatbot_service.models.schemas import MemoryListResponse
from astro_chatbot_service.services.memory import MemoryService

router = APIRouter()


@router.get("/memory/{session_id}", response_model=MemoryListResponse)
def list_memory(
    session_id: str,
    db: Session = Depends(get_db),
) -> MemoryListResponse:
    service = MemoryService(db)
    turns = service.list_session(session_id)
    return MemoryListResponse(session_id=session_id, turns=turns)

