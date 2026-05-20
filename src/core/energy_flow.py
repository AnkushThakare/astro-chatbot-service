from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import re
from typing import Any

from sqlalchemy.orm import Session

from src.db.models import BehaviorEvent, BehaviorProfile
from src.db.repositories.behavior import BehaviorRepository

STRESS_KEYWORDS = {
    "anxious",
    "anxiety",
    "confused",
    "delay",
    "fear",
    "heavy",
    "overthinking",
    "panic",
    "pressure",
    "restless",
    "scared",
    "stressed",
    "stuck",
    "tension",
    "uncertain",
    "worried",
}
UNCERTAINTY_KEYWORDS = {
    "can",
    "confused",
    "maybe",
    "not sure",
    "unclear",
    "unknown",
    "unsure",
    "what if",
    "whether",
    "why",
}
THEME_KEYWORDS = {
    "career": {"career", "job", "promotion", "salary", "work"},
    "relationship": {"dating", "love", "marriage", "partner", "relationship"},
    "finance": {"business", "finance", "money", "wealth"},
    "health": {"health", "sleep", "stress", "wellbeing"},
    "spirituality": {"mantra", "meditation", "pooja", "prayer", "spiritual"},
}
LATE_NIGHT_HOURS = {0, 1, 2, 3, 4}


@dataclass
class EnergyFlowSnapshot:
    scope_key: str
    scope_type: str
    overall_alignment: int
    stress_score: int
    focus_score: int
    emotional_drift_score: int
    cognitive_overload_score: int
    clarity_score: int
    behavioral_consistency_score: int
    emotional_state: str
    focus_state: str
    behavioral_state: str
    signal_count: int
    summary_text: str
    signals: dict[str, Any]
    last_event_at: datetime | None


def _clamp(value: float, low: int = 0, high: int = 100) -> int:
    return max(low, min(high, int(round(value))))


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _count_phrases(text: str, phrases: set[str]) -> int:
    lowered = text.lower()
    return sum(1 for phrase in phrases if phrase in lowered)


def _extract_theme(text: str) -> str | None:
    token_set = set(_tokens(text))
    for theme, keywords in THEME_KEYWORDS.items():
        if token_set & keywords:
            return theme
    return None


def _message_event_payload(message: str, emotion_label: str | None) -> dict[str, Any]:
    token_list = _tokens(message)
    token_count = len(token_list)
    stress_hits = sum(1 for token in token_list if token in STRESS_KEYWORDS)
    uncertainty_tokens = {token for token in UNCERTAINTY_KEYWORDS if " " not in token}
    uncertainty_hits = sum(1 for token in token_list if token in uncertainty_tokens)
    uncertainty_hits += _count_phrases(message, {"not sure", "what if", "unclear", "unsure"})
    return {
        "text_length": len(message),
        "token_count": token_count,
        "stress_hits": stress_hits,
        "uncertainty_hits": uncertainty_hits,
        "theme": _extract_theme(message),
        "emotion_label": emotion_label or "unknown",
        "question_count": message.count("?"),
    }


class EnergyFlowService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.repository = BehaviorRepository(db)

    @staticmethod
    def _scope_key(session_id: str, user_id: int | None) -> tuple[str, str]:
        if user_id is not None:
            return f"user:{user_id}", "user"
        return f"session:{session_id}", "session"

    @staticmethod
    def _event_payload(row: BehaviorEvent) -> dict[str, Any]:
        return row.payload_json if isinstance(row.payload_json, dict) else {}

    @classmethod
    def _derive_snapshot(
        cls,
        *,
        scope_key: str,
        scope_type: str,
        events: list[BehaviorEvent],
    ) -> EnergyFlowSnapshot:
        if not events:
            return EnergyFlowSnapshot(
                scope_key=scope_key,
                scope_type=scope_type,
                overall_alignment=50,
                stress_score=0,
                focus_score=50,
                emotional_drift_score=0,
                cognitive_overload_score=0,
                clarity_score=50,
                behavioral_consistency_score=50,
                emotional_state="steady",
                focus_state="neutral",
                behavioral_state="steady",
                signal_count=0,
                summary_text="No behavioral pattern is established yet.",
                signals={"message_count": 0},
                last_event_at=None,
            )

        message_events = [event for event in events if event.event_type == "message_submitted"]
        typing_events = [event for event in events if event.event_type.startswith("typing")]
        reopen_events = [event for event in events if event.event_type in {"app_reopened", "session_started"}]
        message_count = len(message_events)
        total_tokens = 0
        stress_hits = 0
        uncertainty_hits = 0
        question_count = 0
        theme_counter: Counter[str] = Counter()
        emotion_counter: Counter[str] = Counter()
        typing_pause_ms = 0
        typing_chars = 0
        hesitation_events = 0
        late_night_events = 0
        active_days: set[str] = set()

        for event in events:
            active_days.add(event.occurred_at.date().isoformat())
            if event.occurred_at.hour in LATE_NIGHT_HOURS:
                late_night_events += 1

            payload = cls._event_payload(event)
            if event.event_type == "message_submitted":
                total_tokens += int(payload.get("token_count", 0) or 0)
                stress_hits += int(payload.get("stress_hits", 0) or 0)
                uncertainty_hits += int(payload.get("uncertainty_hits", 0) or 0)
                question_count += int(payload.get("question_count", 0) or 0)
                theme = payload.get("theme")
                if isinstance(theme, str) and theme:
                    theme_counter[theme] += 1
                emotion = payload.get("emotion_label")
                if isinstance(emotion, str) and emotion:
                    emotion_counter[emotion] += 1
            if event.event_type in {"typing_paused", "typing_submitted"}:
                typing_pause_ms += int(payload.get("pause_ms", 0) or 0)
                typing_chars += int(payload.get("chars_typed", 0) or 0)
            if event.event_type == "typing_paused":
                hesitation_events += 1
                hesitation_events += int(payload.get("pause_count", 0) or 0)

        avg_tokens = total_tokens / message_count if message_count else 0.0
        stress_density = (stress_hits / max(total_tokens, 1)) * 100
        uncertainty_density = (uncertainty_hits / max(total_tokens, 1)) * 100
        late_night_ratio = late_night_events / max(len(events), 1)
        hesitation_ratio = hesitation_events / max(message_count or len(typing_events), 1)
        theme_switch_ratio = (len(theme_counter) - 1) / max(message_count, 1) if theme_counter else 0.0
        reopen_ratio = len(reopen_events) / max(len(active_days), 1)
        pause_per_char = typing_pause_ms / max(typing_chars, 1)

        stress_score = _clamp(
            (stress_density * 8)
            + (uncertainty_density * 6)
            + (late_night_ratio * 20)
            + (hesitation_ratio * 8)
        )
        focus_score = _clamp(
            75
            - (uncertainty_density * 6)
            - (hesitation_ratio * 8)
            - (pause_per_char / 12)
            - (late_night_ratio * 15)
            + (min(avg_tokens, 45) / 3)
        )
        emotional_drift_score = _clamp((theme_switch_ratio * 45) + (reopen_ratio * 10))
        cognitive_overload_score = _clamp(
            (stress_score * 0.45)
            + (uncertainty_density * 10)
            + (hesitation_ratio * 12)
            + min(question_count * 4, 20)
        )
        clarity_score = _clamp(
            80
            - (uncertainty_density * 9)
            - (hesitation_ratio * 7)
            - (stress_density * 4)
            - (question_count * 2)
        )
        active_day_score = min(len(active_days) * 18, 65)
        steadiness_bonus = max(0, 25 - int(theme_switch_ratio * 20) - int(late_night_ratio * 10))
        behavioral_consistency_score = _clamp(active_day_score + steadiness_bonus)
        overall_alignment = _clamp(
            (focus_score * 0.24)
            + (clarity_score * 0.22)
            + (behavioral_consistency_score * 0.22)
            + ((100 - stress_score) * 0.18)
            + ((100 - cognitive_overload_score) * 0.14)
        )

        if stress_score >= 65:
            emotional_state = "elevated_stress"
        elif uncertainty_density >= 6:
            emotional_state = "uncertain"
        elif emotion_counter.get("devotional", 0) >= 2:
            emotional_state = "reflective"
        else:
            emotional_state = "steady"

        if focus_score >= 70:
            focus_state = "steady"
        elif focus_score >= 45:
            focus_state = "wavering"
        else:
            focus_state = "scattered"

        if cognitive_overload_score >= 70 and uncertainty_hits >= max(2, message_count):
            behavioral_state = "overthinking_loop"
        elif behavioral_consistency_score < 40:
            behavioral_state = "inconsistent"
        elif late_night_ratio >= 0.35 and stress_score >= 50:
            behavioral_state = "drained_rhythm"
        else:
            behavioral_state = "grounded"

        dominant_theme = theme_counter.most_common(1)[0][0] if theme_counter else "general"
        signal_phrases: list[str] = []
        if uncertainty_hits:
            signal_phrases.append("repeated uncertainty language")
        if stress_hits:
            signal_phrases.append("stress-heavy wording")
        if hesitation_events:
            signal_phrases.append("stop-start typing pauses")
        if late_night_ratio >= 0.25:
            signal_phrases.append("late-night activity")
        if reopen_ratio >= 1.5:
            signal_phrases.append("frequent re-open behavior")
        if not signal_phrases:
            signal_phrases.append("steady interaction rhythm")
        summary_text = (
            f"Dominant theme: {dominant_theme}. "
            f"Observed pattern: {', '.join(signal_phrases[:3])}. "
            f"Behavioral read: {behavioral_state.replace('_', ' ')}."
        )

        return EnergyFlowSnapshot(
            scope_key=scope_key,
            scope_type=scope_type,
            overall_alignment=overall_alignment,
            stress_score=stress_score,
            focus_score=focus_score,
            emotional_drift_score=emotional_drift_score,
            cognitive_overload_score=cognitive_overload_score,
            clarity_score=clarity_score,
            behavioral_consistency_score=behavioral_consistency_score,
            emotional_state=emotional_state,
            focus_state=focus_state,
            behavioral_state=behavioral_state,
            signal_count=len(signal_phrases),
            summary_text=summary_text,
            signals={
                "message_count": message_count,
                "dominant_theme": dominant_theme,
                "stress_hits": stress_hits,
                "uncertainty_hits": uncertainty_hits,
                "late_night_ratio": round(late_night_ratio, 3),
                "hesitation_events": hesitation_events,
                "active_days": len(active_days),
                "theme_count": len(theme_counter),
                "signal_phrases": signal_phrases,
            },
            last_event_at=events[-1].occurred_at,
        )

    @staticmethod
    def _snapshot_to_metrics(snapshot: EnergyFlowSnapshot) -> dict[str, Any]:
        return {
            "overall_alignment": snapshot.overall_alignment,
            "stress_score": snapshot.stress_score,
            "focus_score": snapshot.focus_score,
            "emotional_drift_score": snapshot.emotional_drift_score,
            "cognitive_overload_score": snapshot.cognitive_overload_score,
            "clarity_score": snapshot.clarity_score,
            "behavioral_consistency_score": snapshot.behavioral_consistency_score,
            "emotional_state": snapshot.emotional_state,
            "focus_state": snapshot.focus_state,
            "behavioral_state": snapshot.behavioral_state,
            "signal_count": snapshot.signal_count,
            "summary_text": snapshot.summary_text,
            "signals_json": snapshot.signals,
            "last_event_at": snapshot.last_event_at,
        }

    @classmethod
    def _profile_to_snapshot(cls, profile: BehaviorProfile) -> EnergyFlowSnapshot:
        return EnergyFlowSnapshot(
            scope_key=profile.scope_key,
            scope_type=profile.scope_type,
            overall_alignment=profile.overall_alignment,
            stress_score=profile.stress_score,
            focus_score=profile.focus_score,
            emotional_drift_score=profile.emotional_drift_score,
            cognitive_overload_score=profile.cognitive_overload_score,
            clarity_score=profile.clarity_score,
            behavioral_consistency_score=profile.behavioral_consistency_score,
            emotional_state=profile.emotional_state,
            focus_state=profile.focus_state,
            behavioral_state=profile.behavioral_state,
            signal_count=profile.signal_count,
            summary_text=profile.summary_text or "No behavioral pattern is established yet.",
            signals=profile.signals_json if isinstance(profile.signals_json, dict) else {},
            last_event_at=profile.last_event_at,
        )

    def record_events(
        self,
        *,
        session_id: str,
        events: list[dict[str, Any]],
        user_id: int | None = None,
    ) -> EnergyFlowSnapshot:
        if not events:
            return self.refresh_state(session_id=session_id, user_id=user_id)

        conversation_id = self.repository.resolve_conversation_id(session_id)
        self.repository.add_events(
            session_id=session_id,
            events=events,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        return self.refresh_state(session_id=session_id, user_id=user_id)

    def track_message_signal(
        self,
        *,
        session_id: str,
        message: str,
        emotion_label: str,
        user_id: int | None = None,
        occurred_at: datetime | None = None,
    ) -> EnergyFlowSnapshot:
        payload = _message_event_payload(message, emotion_label)
        return self.record_events(
            session_id=session_id,
            user_id=user_id,
            events=[
                {
                    "event_type": "message_submitted",
                    "source": "chat_service",
                    "occurred_at": occurred_at or datetime.now(timezone.utc),
                    "payload": payload,
                }
            ],
        )

    def refresh_state(
        self,
        *,
        session_id: str,
        user_id: int | None = None,
    ) -> EnergyFlowSnapshot:
        scope_key, scope_type = self._scope_key(session_id, user_id)
        events = self.repository.list_recent_events(session_id=session_id, user_id=user_id)
        snapshot = self._derive_snapshot(scope_key=scope_key, scope_type=scope_type, events=events)
        conversation_id = self.repository.resolve_conversation_id(session_id)
        self.repository.upsert_profile(
            scope_key=scope_key,
            scope_type=scope_type,
            session_id=session_id,
            user_id=user_id,
            conversation_id=conversation_id,
            metrics=self._snapshot_to_metrics(snapshot),
        )
        return snapshot

    def get_snapshot(
        self,
        *,
        session_id: str,
        user_id: int | None = None,
    ) -> EnergyFlowSnapshot:
        scope_key, scope_type = self._scope_key(session_id, user_id)
        profile = self.repository.get_profile(scope_key)
        if profile is not None:
            return self._profile_to_snapshot(profile)
        events = self.repository.list_recent_events(session_id=session_id, user_id=user_id)
        if not events:
            return self._derive_snapshot(scope_key=scope_key, scope_type=scope_type, events=[])
        return self.refresh_state(session_id=session_id, user_id=user_id)

    def behavior_prompt_context(
        self,
        *,
        session_id: str,
        user_id: int | None = None,
        current_message: str | None = None,
        current_emotion: str | None = None,
    ) -> str | None:
        snapshot = self.get_snapshot(session_id=session_id, user_id=user_id)
        current_signal: list[str] = []
        if current_message:
            payload = _message_event_payload(current_message, current_emotion)
            if int(payload.get("uncertainty_hits", 0) or 0) > 0:
                current_signal.append("the current message carries uncertainty")
            if int(payload.get("stress_hits", 0) or 0) > 0:
                current_signal.append("the current message is stress-loaded")
            theme = payload.get("theme")
            if isinstance(theme, str) and theme:
                current_signal.append(f"the live theme is {theme}")

        if snapshot.signal_count == 0 and not current_signal:
            return None

        signal_phrases = snapshot.signals.get("signal_phrases") if isinstance(snapshot.signals, dict) else []
        lines = [
            "Energy flow snapshot (derived from real interaction behavior, not random scoring):",
            f"- Overall alignment: {snapshot.overall_alignment}/100",
            f"- Emotional state: {snapshot.emotional_state}",
            f"- Focus state: {snapshot.focus_state}",
            f"- Behavioral state: {snapshot.behavioral_state}",
        ]
        if isinstance(signal_phrases, list) and signal_phrases:
            lines.append("- Pattern signals: " + ", ".join(str(item) for item in signal_phrases[:3]))
        if current_signal:
            lines.append("- Current message cue: " + ", ".join(current_signal[:2]))
        lines.append(
            "- Response guidance: name the pattern once if relevant, stay concise, do not sound therapeutic or preachy."
        )
        return "\n".join(lines)

    @staticmethod
    def serialize_snapshot(snapshot: EnergyFlowSnapshot) -> dict[str, Any]:
        return {
            "scope_key": snapshot.scope_key,
            "scope_type": snapshot.scope_type,
            "overall_alignment": snapshot.overall_alignment,
            "stress_score": snapshot.stress_score,
            "focus_score": snapshot.focus_score,
            "emotional_drift_score": snapshot.emotional_drift_score,
            "cognitive_overload_score": snapshot.cognitive_overload_score,
            "clarity_score": snapshot.clarity_score,
            "behavioral_consistency_score": snapshot.behavioral_consistency_score,
            "emotional_state": snapshot.emotional_state,
            "focus_state": snapshot.focus_state,
            "behavioral_state": snapshot.behavioral_state,
            "signal_count": snapshot.signal_count,
            "summary_text": snapshot.summary_text,
            "signals": snapshot.signals,
            "last_event_at": snapshot.last_event_at.isoformat() if snapshot.last_event_at else None,
        }
