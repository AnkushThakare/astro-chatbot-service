from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.config import settings
from astro_chatbot_service.models.schemas import ChatRequest, ChatResponse
from astro_chatbot_service.services.chat_orchestrator import ChatOrchestrator

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    db: Session = Depends(get_db),
) -> ChatResponse:
    orchestrator = ChatOrchestrator(db=db, settings=settings)
    return await orchestrator.generate_reply(request)
