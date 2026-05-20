from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import AliasChoices, BaseModel, Field, UUID4, ValidationError
from sqlalchemy.orm import Session

from src.api.kundali import BirthDetails
from src.api.matchmaking import MatchmakingDetails
from src.auth.jwt import AuthenticatedUser, get_optional_current_user, seconds_until_token_expiry
from src.core.chat_service import ChatService
from src.core.config import settings
from src.core.idempotency import chat_idempotency_store
from src.core.llm import GroqClient, create_llm_client
from src.core.logging import get_logger
from src.core.memory import MemoryService
from src.core.rate_limit import check_rate_limit
from src.core.streaming import sse_event
from src.db.session import get_db

router = APIRouter()
logger = get_logger(__name__)


class ChatMessageRequest(BaseModel):
    session_id: str | None = Field(default=None, min_length=1)
    client_message_id: UUID4 | None = None
    message: str = Field(min_length=1, validation_alias=AliasChoices("message", "text"))
    birth_details: BirthDetails | None = None
    matchmaking_details: MatchmakingDetails | None = None


def _idempotency_scope(
    request: Request,
    session_id: str,
    current_user: AuthenticatedUser | None,
) -> str:
    if current_user is not None:
        return f"user:{current_user.user_id}"
    client_host = request.client.host if request.client else "unknown"
    return f"anon:{session_id}:{client_host}"


@router.post("/chat/message", dependencies=[Depends(check_rate_limit)])
async def chat_message(
    http_request: Request,
    body: dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    current_user: AuthenticatedUser | None = Depends(get_optional_current_user),
) -> StreamingResponse:
    try:
        request_payload = ChatMessageRequest.model_validate(body)
    except ValidationError as exc:
        if any("client_message_id" in ".".join(str(part) for part in err["loc"]) for err in exc.errors()):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="client_message_id must be a valid UUID v4",
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid chat request payload",
        ) from exc

    session_id = request_payload.session_id or (
        f"user-{current_user.user_id}" if current_user is not None else "anonymous-session"
    )
    msg_id = request_payload.client_message_id or uuid.uuid4()
    idempotency_key = (
        f"{_idempotency_scope(http_request, session_id, current_user)}:"
        f"{msg_id}"
    )
    chat_idempotency_store.ttl_seconds = settings.IDEMPOTENCY_TTL_SECONDS
    chat_service = ChatService(db, settings)

    token_remaining_seconds = (
        seconds_until_token_expiry(current_user) if current_user is not None else None
    )
    if (
        token_remaining_seconds is not None
        and token_remaining_seconds < settings.JWT_STREAM_MIN_REMAINING_SECONDS
    ):
        async def token_expiring_stream():
            yield sse_event(
                "error",
                {
                    "type": "error",
                    "data": {
                        "code": "token_expiring",
                        "message": "Please refresh your session before sending this message.",
                        "action": "refresh_token",
                    },
                    "session_id": request_payload.session_id,
                },
            )

        return StreamingResponse(
            token_expiring_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    async def event_stream():
        replay, cached_events = await chat_idempotency_store.reserve_or_replay(idempotency_key)
        if replay and cached_events is not None:
            logger.info(
                "replaying_cached_chat_response",
                extra={"extra_fields": {"session_id": session_id}},
            )
            for rendered_event in cached_events:
                yield rendered_event
            return

        rendered_events: list[str] = []
        completed = False
        try:
            async for event_name, payload in chat_service.stream_reply_events(
                session_id=session_id,
                message=request_payload.message,
                birth_details=(
                    request_payload.birth_details.model_dump(mode="json")
                    if request_payload.birth_details
                    else None
                ),
                matchmaking_details=(
                    request_payload.matchmaking_details.model_dump(mode="json")
                    if request_payload.matchmaking_details
                    else None
                ),
                current_user=current_user,
                disconnect_checker=http_request.is_disconnected,
            ):
                payload["session_id"] = request_payload.session_id
                rendered = sse_event(event_name, payload)
                rendered_events.append(rendered)
                if event_name == "done":
                    completed = True
                yield rendered
        except Exception:
            await chat_idempotency_store.abort(idempotency_key)
            raise

        if completed:
            await chat_idempotency_store.complete(idempotency_key, rendered_events)
        else:
            await chat_idempotency_store.abort(idempotency_key)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


class SessionSummaryRequest(BaseModel):
    session_id: str = Field(min_length=1)


@router.post("/chat/summarize")
async def summarize_session(
    request: SessionSummaryRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Generate summary and extract facts for a session."""
    groq = create_llm_client(settings, role="response")
    if not groq.is_configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM not configured",
        )

    memory = MemoryService(db)
    summary = await memory.summarize_conversation(request.session_id, groq)
    facts = await memory.extract_and_store_facts(request.session_id, groq)

    return {
        "session_id": request.session_id,
        "summary": summary,
        "extracted_facts": facts,
    }
