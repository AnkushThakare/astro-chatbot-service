from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import AliasChoices, BaseModel, Field
from sqlalchemy.orm import Session

from src.api.kundali import BirthDetails
from src.api.matchmaking import MatchmakingDetails
from src.auth.jwt import AuthenticatedUser, get_optional_current_user
from src.core.chat_service import ChatService
from src.core.config import settings
from src.core.streaming import sse_event
from src.db.session import get_db

router = APIRouter()


class ChatMessageRequest(BaseModel):
    session_id: str | None = Field(default=None, min_length=1)
    message: str = Field(min_length=1, validation_alias=AliasChoices("message", "text"))
    birth_details: BirthDetails | None = None
    matchmaking_details: MatchmakingDetails | None = None


@router.post("/chat/message")
async def chat_message(
    request: ChatMessageRequest,
    db: Session = Depends(get_db),
    current_user: AuthenticatedUser | None = Depends(get_optional_current_user),
) -> StreamingResponse:
    session_id = request.session_id or (
        f"user-{current_user.user_id}" if current_user is not None else "anonymous-session"
    )
    chat_service = ChatService(db, settings)

    async def event_stream():
        async for event_name, payload in chat_service.stream_reply_events(
            session_id=session_id,
            message=request.message,
            birth_details=request.birth_details.model_dump(mode="json") if request.birth_details else None,
            matchmaking_details=(
                request.matchmaking_details.model_dump(mode="json")
                if request.matchmaking_details
                else None
            ),
            current_user=current_user,
        ):
            payload["session_id"] = request.session_id
            yield sse_event(event_name, payload)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
