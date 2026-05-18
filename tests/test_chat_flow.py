import time
from uuid import uuid4

from jose import jwt

from src.core.chat_service import ChatService
from src.core.config import settings

def test_chat_flow_without_groq_key(client) -> None:
    session_id = f"test-session-{uuid4()}"

    chat_response = client.post(
        "/api/v1/chat/message",
        json={
            "session_id": session_id,
            "client_message_id": str(uuid4()),
            "text": "Tell me about Saturn remedies",
            "birth_details": {
                "name": "Ada",
                "latitude": 12.9716,
                "longitude": 77.5946,
                "birth_datetime": "1990-01-01T12:00:00+00:00",
            },
        },
    )
    assert chat_response.status_code == 200
    assert chat_response.headers["content-type"].startswith("text/event-stream")
    assert chat_response.headers["X-Correlation-ID"]
    assert "event: meta" in chat_response.text
    assert "event: done" in chat_response.text
    assert "local fallback response" in chat_response.text.lower()


def test_chat_message_replays_cached_response_for_same_client_message_id(client, monkeypatch) -> None:
    call_count = 0

    async def fake_stream(  # noqa: ANN202
        self,
        *,
        session_id: str,
        message: str,
        birth_details=None,
        matchmaking_details=None,
        current_user=None,
        disconnect_checker=None,
    ):
        nonlocal call_count
        del self, session_id, message, birth_details, matchmaking_details, current_user, disconnect_checker
        call_count += 1
        yield ("meta", {"intent": "respond_only"})
        yield ("done", {"reply": "cached reply", "intent": "respond_only", "response": {"metadata": {}}, "metadata": {}})

    monkeypatch.setattr(ChatService, "stream_reply_events", fake_stream)
    payload = {
        "session_id": f"test-session-{uuid4()}",
        "client_message_id": str(uuid4()),
        "text": "hello",
    }

    first = client.post("/api/v1/chat/message", json=payload)
    second = client.post("/api/v1/chat/message", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.text == second.text
    assert call_count == 1


def test_chat_message_rejects_invalid_client_message_id(client) -> None:
    response = client.post(
        "/api/v1/chat/message",
        json={
            "session_id": f"test-session-{uuid4()}",
            "client_message_id": "not-a-uuid",
            "text": "hello",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "client_message_id must be a valid UUID v4"


def test_chat_message_processes_different_client_message_ids_normally(client, monkeypatch) -> None:
    call_count = 0

    async def fake_stream(  # noqa: ANN202
        self,
        *,
        session_id: str,
        message: str,
        birth_details=None,
        matchmaking_details=None,
        current_user=None,
        disconnect_checker=None,
    ):
        nonlocal call_count
        del self, session_id, message, birth_details, matchmaking_details, current_user, disconnect_checker
        call_count += 1
        yield ("meta", {"intent": "respond_only"})
        yield ("done", {"reply": f"reply-{call_count}", "intent": "respond_only", "response": {"metadata": {}}, "metadata": {}})

    monkeypatch.setattr(ChatService, "stream_reply_events", fake_stream)
    session_id = f"test-session-{uuid4()}"

    first = client.post(
        "/api/v1/chat/message",
        json={"session_id": session_id, "client_message_id": str(uuid4()), "text": "hello"},
    )
    second = client.post(
        "/api/v1/chat/message",
        json={"session_id": session_id, "client_message_id": str(uuid4()), "text": "hello"},
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert call_count == 2


def test_chat_message_refuses_stream_when_token_is_near_expiry(client) -> None:
    token = jwt.encode(
        {
            "sub": "user-1",
            "role": "customer",
            "exp": int(time.time()) + 30,
        },
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )

    response = client.post(
        "/api/v1/chat/message",
        json={
            "session_id": f"test-session-{uuid4()}",
            "client_message_id": str(uuid4()),
            "text": "hello",
        },
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert "token_expiring" in response.text
