from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from src.core.energy_flow import EnergyFlowService
from src.db.session import get_db


def test_energy_flow_service_derives_behavior_state_from_real_signals() -> None:
    db = next(get_db())
    try:
        service = EnergyFlowService(db)
        session_id = f"energy-{uuid4()}"

        service.record_events(
            session_id=session_id,
            events=[
                {
                    "event_type": "session_started",
                    "occurred_at": datetime(2026, 5, 18, 1, 10, 0),
                    "source": "client",
                    "payload": {},
                },
                {
                    "event_type": "typing_paused",
                    "occurred_at": datetime(2026, 5, 18, 1, 12, 0),
                    "source": "client",
                    "payload": {"pause_ms": 24000, "pause_count": 3, "chars_typed": 70},
                },
                {
                    "event_type": "app_reopened",
                    "occurred_at": datetime(2026, 5, 18, 1, 20, 0),
                    "source": "client",
                    "payload": {},
                },
            ],
        )
        service.track_message_signal(
            session_id=session_id,
            message="I am confused about my career, stressed about money, and not sure if I am making the wrong move.",
            emotion_label="career_stress",
        )
        snapshot = service.track_message_signal(
            session_id=session_id,
            message="Why does this keep happening? What if I make the wrong decision again?",
            emotion_label="confused",
        )

        assert snapshot.stress_score > 0
        assert snapshot.cognitive_overload_score > 0
        assert snapshot.focus_state in {"scattered", "wavering", "steady"}
        assert snapshot.behavioral_state in {"overthinking_loop", "drained_rhythm", "inconsistent", "grounded"}
        assert snapshot.signals["message_count"] == 2

        prompt_context = service.behavior_prompt_context(
            session_id=session_id,
            current_message="I am still not sure what to do next.",
            current_emotion="confused",
        )
        assert prompt_context is not None
        assert "Energy flow snapshot" in prompt_context
        assert "Current message cue" in prompt_context
    finally:
        db.close()


def test_behavior_events_endpoint_returns_derived_state(client) -> None:
    session_id = f"behavior-{uuid4()}"

    response = client.post(
        "/api/v1/behavior/events",
        json={
            "session_id": session_id,
            "events": [
                {
                    "event_type": "typing_paused",
                    "occurred_at": "2026-05-18T02:00:00",
                    "payload": {"pause_ms": 18000, "pause_count": 2, "chars_typed": 60},
                },
                {
                    "event_type": "message_submitted",
                    "occurred_at": "2026-05-18T02:01:00",
                    "payload": {
                        "token_count": 14,
                        "stress_hits": 2,
                        "uncertainty_hits": 2,
                        "theme": "career",
                        "emotion_label": "career_stress",
                        "question_count": 1,
                    },
                },
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["ingested_count"] == 2
    assert payload["state"]["stress_score"] > 0
    assert payload["state"]["signal_count"] >= 1

    state_response = client.get(f"/api/v1/behavior/state/{session_id}")

    assert state_response.status_code == 200
    state_payload = state_response.json()
    assert state_payload["status"] == "ok"
    assert state_payload["state"]["signals"]["message_count"] == 1
    assert state_payload["state"]["overall_alignment"] >= 0
