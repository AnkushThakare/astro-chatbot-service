from uuid import uuid4

def test_chat_flow_without_groq_key(client) -> None:
    session_id = f"test-session-{uuid4()}"

    chat_response = client.post(
        "/api/v1/chat/message",
        json={
            "session_id": session_id,
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
