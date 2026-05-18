import asyncio
import json
import time
from types import SimpleNamespace

import pytest
from jose import jwt

from src.auth.jwt import (
    AuthenticatedUser,
    InvalidSignatureTokenError,
    MalformedTokenError,
    decode_jwt_token,
)
from src.core.chat_service import ChatService
from src.core.config import settings
from src.core.core_service import CoreServiceClient
from src.core.planner import PlannerResult
from src.core.router import ChatRouteDecision
from src.db.repositories.conversations import ConversationRepository
from src.db.session import SessionLocal


def _current_user() -> AuthenticatedUser:
    return AuthenticatedUser(
        user_id="user-123",
        role="customer",
        token_type="access",
        raw_token="token-123",
        raw_claims={},
    )


def test_prepare_base_reply_context_uses_core_service_birth_details() -> None:
    service = ChatService.__new__(ChatService)
    service.settings = settings
    service.memory_service = SimpleNamespace(
        recent_messages=lambda session_id, limit: [],
        repository=SimpleNamespace(get_session_state=lambda session_id: None),
    )
    service._resolve_internal_user_id = lambda current_user: None  # type: ignore[method-assign]

    class _CoreStub:
        async def get_user_birth_profile(self, user_id: str, current_user: AuthenticatedUser):  # noqa: ANN202
            del user_id, current_user
            return None

        async def get_user_birth_details(self, user_id: str, current_user: AuthenticatedUser):  # noqa: ANN202
            del user_id, current_user
            return {
                "name": "Ada",
                "latitude": 12.9716,
                "longitude": 77.5946,
                "birth_datetime": "1990-01-01T12:00:00+00:00",
                "timezone_str": "Etc/UTC",
            }

        async def save_user_birth_details(self, user_id: str, payload: dict, current_user: AuthenticatedUser):  # noqa: ANN202
            del user_id, payload, current_user
            raise AssertionError("save_user_birth_details should not be called")

    service.core_service_client = _CoreStub()

    context = asyncio.run(
        service._prepare_base_reply_context(
            session_id="session-1",
            message="Show my kundali",
            current_user=_current_user(),
        )
    )

    assert context["effective_birth_details"] is not None
    assert context["effective_birth_details"]["latitude"] == 12.9716
    assert context["route_decision"].intent == "show_kundali"
    assert context["needs_birth_details"] is False


def test_prepare_base_reply_context_saves_birth_details_followup_to_core_service() -> None:
    service = ChatService.__new__(ChatService)
    service.settings = settings
    service.memory_service = SimpleNamespace(
        recent_messages=lambda session_id, limit: [
            {
                "role": "assistant",
                "content": "For an exact chart-based answer, I would need your birth details. If you want, share your date, time, and place of birth.",
            }
        ],
        repository=SimpleNamespace(get_session_state=lambda session_id: None),
    )
    service._resolve_internal_user_id = lambda current_user: None  # type: ignore[method-assign]

    saved: dict[str, object] = {}

    class _CoreStub:
        async def get_user_birth_profile(self, user_id: str, current_user: AuthenticatedUser):  # noqa: ANN202
            del user_id, current_user
            return None

        async def get_user_birth_details(self, user_id: str, current_user: AuthenticatedUser):  # noqa: ANN202
            del user_id, current_user
            return None

        async def save_user_birth_details(self, user_id: str, payload: dict, current_user: AuthenticatedUser):  # noqa: ANN202
            saved["user_id"] = user_id
            saved["payload"] = dict(payload)
            del current_user
            return payload

    async def _infer_birth_details(message: str):  # noqa: ANN202
        del message
        return {
            "name": None,
            "latitude": 18.5204,
            "longitude": 73.8567,
            "birth_datetime": "2001-06-20T22:22:00",
            "timezone_str": "Asia/Kolkata",
        }

    service.core_service_client = _CoreStub()
    service._infer_birth_details_from_message = _infer_birth_details  # type: ignore[method-assign]

    context = asyncio.run(
        service._prepare_base_reply_context(
            session_id="session-2",
            message="20.06.2001 time 22.22 place pune",
            current_user=_current_user(),
        )
    )

    assert saved["user_id"] == "user-123"
    assert isinstance(saved["payload"], dict)
    assert saved["payload"]["timezone_str"] == "Asia/Kolkata"
    assert context["route_decision"].intent == "show_kundali"
    assert context["effective_birth_details"]["longitude"] == 73.8567


def test_prepare_base_reply_context_resumes_kundali_from_session_state() -> None:
    service = ChatService.__new__(ChatService)
    service.settings = settings
    service.memory_service = SimpleNamespace(
        recent_messages=lambda session_id, limit: [],
        repository=SimpleNamespace(
            get_session_state=lambda session_id: {
                "active_intent": "show_kundali",
                "birth_details": {
                    "name": "Ada",
                    "latitude": 12.9716,
                    "longitude": 77.5946,
                    "birth_datetime": "1990-01-01T12:00:00+00:00",
                    "timezone_str": "Etc/UTC",
                },
                "pending_slots": [],
                "last_tool": "show_kundali",
            }
        ),
    )
    service._resolve_internal_user_id = lambda current_user: None  # type: ignore[method-assign]

    class _CoreStub:
        async def get_user_birth_details(self, user_id: str, current_user: AuthenticatedUser):  # noqa: ANN202
            del user_id, current_user
            return None

        async def get_user_birth_profile(self, user_id: str, current_user: AuthenticatedUser):  # noqa: ANN202
            del user_id, current_user
            return None

    service.core_service_client = _CoreStub()

    context = asyncio.run(
        service._prepare_base_reply_context(
            session_id="session-kundali-1",
            message="What about my career timing?",
            current_user=_current_user(),
        )
    )

    assert context["route_decision"].route == "TOOL_FLOW"
    assert context["route_decision"].intent == "show_kundali"
    assert context["route_decision"].reason == "cached_kundali_context"


def test_prepare_base_reply_context_keeps_birth_slot_clarification_from_session_state() -> None:
    service = ChatService.__new__(ChatService)
    service.settings = settings
    service.memory_service = SimpleNamespace(
        recent_messages=lambda session_id, limit: [],
        repository=SimpleNamespace(
            get_session_state=lambda session_id: {
                "active_intent": "show_kundali",
                "partial_birth_details": {
                    "date_parts": (6, 6, 2004),
                    "time_parts": (17, 0),
                    "place": None,
                },
                "pending_slots": ["birth_place"],
                "last_tool": "show_kundali",
            }
        ),
    )
    service._resolve_internal_user_id = lambda current_user: None  # type: ignore[method-assign]

    class _CoreStub:
        async def get_user_birth_details(self, user_id: str, current_user: AuthenticatedUser):  # noqa: ANN202
            del user_id, current_user
            return None

        async def get_user_birth_profile(self, user_id: str, current_user: AuthenticatedUser):  # noqa: ANN202
            del user_id, current_user
            return None

    service.core_service_client = _CoreStub()

    context = asyncio.run(
        service._prepare_base_reply_context(
            session_id="session-kundali-2",
            message="Already given",
            current_user=_current_user(),
        )
    )

    assert context["route_decision"].route == "CLARIFICATION"
    assert context["route_decision"].intent == "show_kundali"
    assert context["route_decision"].missing_slots == ["birth_place"]
    assert context["birth_details_capture_pending"] is True


def test_persist_chat_turns_records_metadata_fields() -> None:
    captured: list[dict[str, object]] = []

    class _Repository:
        def add_turn(self, session_id: str, role: str, content: str, **kwargs):  # noqa: ANN202
            captured.append(
                {
                    "session_id": session_id,
                    "role": role,
                    "content": content,
                    **kwargs,
                }
            )

        def save_session_state(self, session_id: str, state: dict, *, user_id=None):  # noqa: ANN001, ANN202
            captured.append(
                {
                    "session_id": session_id,
                    "saved_session_state": state,
                    "user_id": user_id,
                }
            )

    service = ChatService.__new__(ChatService)
    service.memory_service = SimpleNamespace(repository=_Repository())

    context = {
        "plan": PlannerResult(
            action="suggest_consultant",
            confidence=0.92,
            arguments={},
            missing_information=[],
            should_call_tool=True,
            reasoning="needs human help",
        ),
        "route": SimpleNamespace(provider="groq", model="llama-3.3-70b-versatile", reasoning_profile="tool-aware"),
        "route_decision": ChatRouteDecision(
            route="TOOL_FLOW",
            intent="suggest_consultant",
            confidence=0.92,
            risk_level="low",
            reason="consultant_request",
            should_call_tool=True,
        ),
        "session_id": "session-3",
        "message": "I need a pandit consultation",
        "metadata_json": None,
    }
    metadata = {
        "prompt_versions": {"persona": "v1.3.0", "planner": "v1.2.0", "content_hash": "abcd1234"},
        "model_used": "llama-3.3-70b-versatile",
        "route_taken": "TOOL_FLOW",
        "tool_called": "suggest_consultant",
        "variant_id": "base",
        "total_tokens_input": 120,
        "total_tokens_output": 45,
        "latency_ms": 321,
        "retrieval_trace": {
            "match_count": 2,
            "knowledge_match_count": 1,
            "policy_match_count": 1,
            "knowledge": [{"source": "Consultant notes", "domain": "consultant_reference"}],
            "policy": [{"source": "Consultant booking policy", "domain": "booking_guidance"}],
        },
    }

    service._persist_chat_turns(context, "You can speak with a pandit directly.", metadata, partial=True)

    assert len(captured) == 2
    assistant_turn = captured[1]
    assert assistant_turn["role"] == "assistant"
    assert assistant_turn["prompt_versions"] == metadata["prompt_versions"]
    assert assistant_turn["model_used"] == "llama-3.3-70b-versatile"
    assert assistant_turn["route_taken"] == "TOOL_FLOW"
    assert assistant_turn["tool_called"] == "suggest_consultant"
    assert assistant_turn["variant_id"] == "base"
    assert assistant_turn["total_tokens_input"] == 120
    assert assistant_turn["total_tokens_output"] == 45
    assert assistant_turn["latency_ms"] == 321
    assert assistant_turn["partial"] is True
    persisted_metadata = json.loads(str(assistant_turn["metadata_json"]))
    assert persisted_metadata["response_metadata"]["retrieval_trace"] == metadata["retrieval_trace"]
    assert any("saved_session_state" in item for item in captured)


def test_conversation_repository_session_state_round_trip_excludes_long_term_memory() -> None:
    db = SessionLocal()
    try:
        repository = ConversationRepository(db)
        repository.save_session_state(
            "session-state-1",
            {"active_intent": "show_kundali", "pending_slots": ["birth_place"]},
        )
        repository.upsert_fact("session-state-1", "preferred_language", "english")

        session_state = repository.get_session_state("session-state-1")
        facts = repository.list_facts("session-state-1")

        assert session_state == {
            "active_intent": "show_kundali",
            "pending_slots": ["birth_place"],
        }
        assert all(fact.fact_key != repository.SESSION_STATE_FACT_KEY for fact in facts)
        assert any(fact.fact_key == "preferred_language" for fact in facts)
    finally:
        db.close()


def test_core_service_birth_details_cache_avoids_second_request() -> None:
    CoreServiceClient.invalidate_birth_details_cache("user-123")
    client = CoreServiceClient(settings)
    calls = 0

    class _Response:
        def json(self) -> dict[str, object]:
            return {
                "birth_details": {
                    "name": "Ada",
                    "latitude": 12.9716,
                    "longitude": 77.5946,
                    "date_of_birth": "1990-01-01",
                    "time_of_birth": "12:00:00",
                    "timezone_str": "Etc/UTC",
                }
            }

    async def _request(method: str, path: str, *, headers: dict[str, str], params=None, json=None):  # noqa: ANN202
        nonlocal calls
        del method, path, headers, params, json
        calls += 1
        return _Response()

    client._request = _request  # type: ignore[method-assign]

    first = asyncio.run(client.get_user_birth_details("user-123", _current_user()))
    second = asyncio.run(client.get_user_birth_details("user-123", _current_user()))

    assert calls == 1
    assert second == first
    assert first is not None
    assert first["birth_datetime"] == "1990-01-01T12:00:00"


def test_core_service_partial_birth_profile_serialization_round_trip() -> None:
    payload = {
        "date_parts": (20, 6, 2001),
        "time_parts": (22, 30),
        "place": "Pune",
    }

    serialized = CoreServiceClient._serialize_partial_birth_profile(payload)

    assert serialized == {
        "birth_date": "2001-06-20",
        "birth_time": "22:30:00",
        "birth_place": "Pune",
    }

    deserialized = CoreServiceClient._deserialize_partial_birth_profile(serialized)

    assert deserialized == payload


def test_core_service_partial_birth_profile_cache_avoids_second_request() -> None:
    CoreServiceClient.invalidate_birth_details_cache("user-123")
    client = CoreServiceClient(settings)
    calls = 0

    class _Response:
        def json(self) -> dict[str, object]:
            return {
                "birth_date": "2001-06-20",
                "birth_time": "22:30:00",
                "birth_place": "Pune",
            }

    async def _request(method: str, path: str, *, headers: dict[str, str], params=None, json=None):  # noqa: ANN202
        nonlocal calls
        del method, path, headers, params, json
        calls += 1
        return _Response()

    client._request = _request  # type: ignore[method-assign]

    first = asyncio.run(client.get_user_birth_profile("user-123", _current_user()))
    second = asyncio.run(client.get_user_birth_profile("user-123", _current_user()))

    assert calls == 1
    assert first == second
    assert first == {
        "date_parts": (20, 6, 2001),
        "time_parts": (22, 30),
        "place": "Pune",
    }


def test_internal_invalidate_cache_clears_birth_cache(client) -> None:
    CoreServiceClient._birth_details_cache["user-123"] = (time.time(), {"birth_datetime": "1990-01-01T12:00:00"})

    response = client.post(
        "/internal/invalidate-cache",
        json={"user_id": "user-123"},
        headers={"X-Internal-API-Key": settings.INTERNAL_API_KEY},
    )

    assert response.status_code == 200
    assert "user-123" not in CoreServiceClient._birth_details_cache


def test_decode_jwt_token_rejects_malformed_token() -> None:
    with pytest.raises(MalformedTokenError):
        decode_jwt_token("not-a-jwt")


def test_decode_jwt_token_rejects_invalid_signature() -> None:
    token = jwt.encode(
        {
            "sub": "user-1",
            "role": "customer",
            "exp": int(time.time()) + 300,
        },
        "wrong-secret",
        algorithm=settings.JWT_ALGORITHM,
    )

    with pytest.raises(InvalidSignatureTokenError):
        decode_jwt_token(token)
