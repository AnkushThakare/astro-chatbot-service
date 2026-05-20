import asyncio
import logging

from src.core.chat_service import ChatService
from src.core.guardrails import final_response_guardrail, pre_scope_guardrail, tool_specific_guardrail
from src.core.planner import PlannerResult
from src.core.response_composer import build_cards, normalize_tool_outputs
from src.core.router import ChatRouteDecision, classify_route
from src.core.streaming import chunk_text


def test_chat_service_is_defined() -> None:
    assert ChatService


def test_infer_response_language_prefers_english_for_plain_english_message() -> None:
    message = "I want one practical next step for my career this week."
    assert ChatService._infer_response_language(message) == "english"


def test_infer_response_language_prefers_hinglish_for_hinglish_message() -> None:
    message = "Mujhe career ke liye ek practical step batayein, main bahut confused hoon."
    assert ChatService._infer_response_language(message) == "hinglish"


def test_fast_greeting_reply_handles_simple_english_greeting() -> None:
    plan = PlannerResult(
        action="respond_only",
        confidence=0.99,
        arguments={},
        missing_information=[],
        should_call_tool=False,
        reasoning="simple greeting",
    )
    reply = ChatService._build_fast_astrology_reply(
        message="hii",
        plan=plan,
        birth_details=None,
        matchmaking_details=None,
    )
    assert reply == "Hello. What would you like guidance on?"


def test_birth_details_followup_detection_handles_plain_text_details() -> None:
    recent_messages = [
        {
            "role": "assistant",
            "content": "For an exact chart-based answer, I would need your birth details. If you want, share your date, time, and place of birth.",
        }
    ]
    assert ChatService._looks_like_birth_details_message("20.06.2001 time 22.22 place pune")
    assert ChatService._is_birth_details_followup(
        "20.06.2001 time 22.22 place pune",
        recent_messages,
    )


def test_birth_details_followup_detection_accepts_partial_birth_detail_message() -> None:
    recent_messages = [
        {
            "role": "assistant",
            "content": "For an exact chart-based answer, I would need your birth details. If you want, share your date, time, and place of birth.",
        }
    ]

    assert ChatService._is_birth_details_followup("20/06/2001", recent_messages)


def test_assistant_requested_birth_details_detects_capture_reply_prompt() -> None:
    recent_messages = [
        {
            "role": "assistant",
            "content": (
                "I have your date and time. "
                "For an exact chart, send the birthplace a bit more specifically, like city, state, country."
            ),
        }
    ]

    assert ChatService._assistant_requested_birth_details(recent_messages) is True


def test_birth_details_parser_handles_combined_kundli_message() -> None:
    parts = ChatService._extract_birth_detail_parts(
        "My dob is 6/6/2004 Pune India 5 pm tell me kundli"
    )

    assert parts == {
        "date_parts": (6, 6, 2004),
        "time_parts": (17, 0),
        "place": "Pune India",
    }


def test_birth_details_followup_ignores_greeting_message() -> None:
    recent_messages = [
        {
            "role": "assistant",
            "content": "I have your birthplace. Please share your date of birth and birth time as well.",
        }
    ]

    assert ChatService._extract_birth_place_text("Hello") is None
    assert ChatService._is_birth_details_followup("Hello", recent_messages) is False


def test_birth_details_followup_ignores_generic_question_message() -> None:
    recent_messages = [
        {
            "role": "assistant",
            "content": "I have your birthplace. Please share your date of birth and birth time as well.",
        }
    ]

    assert ChatService._extract_birth_place_text("What is it?") is None
    assert ChatService._is_birth_details_followup("What is it?", recent_messages) is False
    assert ChatService._message_acknowledges_shared_birth_details("What is it?") is False


def test_birth_details_followup_does_not_treat_acknowledgement_as_place() -> None:
    recent_messages = [
        {
            "role": "assistant",
            "content": "I have your birthplace. Please share your date of birth and birth time as well.",
        }
    ]

    assert ChatService._extract_birth_detail_parts(
        "Already given",
        allow_bare_place=True,
    )["place"] is None
    assert ChatService._is_birth_details_followup("Already given", recent_messages) is False
    assert ChatService._message_acknowledges_shared_birth_details("Already given")


def test_assistant_requested_matchmaking_details_detects_matchmaking_prompt() -> None:
    recent_messages = [
        {
            "role": "assistant",
            "content": "For matchmaking, I would need the birth details of both individuals.",
        }
    ]

    assert ChatService._assistant_requested_matchmaking_details(recent_messages) is True


def test_should_resume_matchmaking_from_context_for_short_followup() -> None:
    recent_messages = [
        {
            "role": "assistant",
            "content": "For matchmaking, I would need the birth details of both individuals.",
        }
    ]
    route_decision = ChatRouteDecision(
        route="FAST_CHAT",
        intent="respond_only",
        confidence=0.64,
        risk_level="low",
        reason="default_fast_chat",
        should_call_tool=False,
    )

    assert ChatService._should_resume_matchmaking_from_context(
        message="check now",
        recent_messages=recent_messages,
        matchmaking_details={"primary": {}, "partner": {}},
        route_decision=route_decision,
        session_state={"active_intent": "matchmaking"},
    ) is True


def test_birth_details_followup_ignores_product_request_message() -> None:
    recent_messages = [
        {
            "role": "assistant",
            "content": "I have your birthplace. Please share your date of birth and birth time as well.",
        }
    ]

    assert ChatService._extract_birth_detail_parts(
        "Show me the product",
        allow_bare_place=True,
    )["place"] is None
    assert ChatService._is_birth_details_followup("Show me the product", recent_messages) is False


def test_birth_details_acknowledgement_uses_cached_partial_slots() -> None:
    reply = ChatService._build_birth_details_capture_reply(
        "Already given",
        {
            "date_parts": (6, 6, 2004),
            "time_parts": (17, 0),
            "place": None,
        },
    )

    assert ChatService._message_acknowledges_shared_birth_details("Already given")
    assert reply == (
        "I have your date and time. "
        "For an exact chart, send the birthplace a bit more specifically, like city, state, country."
    )


def test_should_resume_kundali_from_session_context_for_guidance_followup() -> None:
    route_decision = ChatRouteDecision(
        route="FAST_CHAT",
        intent="respond_only",
        confidence=0.64,
        risk_level="low",
        reason="default_fast_chat",
        should_call_tool=False,
    )

    assert ChatService._should_resume_kundali_from_context(
        message="What about my career timing?",
        birth_details={"birth_datetime": "1990-01-01T10:00:00"},
        route_decision=route_decision,
        session_state={"active_intent": "show_kundali", "last_tool": "show_kundali"},
    ) is True


def test_should_keep_kundali_clarification_from_session_context_for_acknowledgement() -> None:
    route_decision = ChatRouteDecision(
        route="FAST_CHAT",
        intent="respond_only",
        confidence=0.64,
        risk_level="low",
        reason="default_fast_chat",
        should_call_tool=False,
    )

    assert ChatService._should_keep_kundali_clarification_from_context(
        message="Already given",
        partial_birth_details={
            "date_parts": (6, 6, 2004),
            "time_parts": (17, 0),
            "place": None,
        },
        route_decision=route_decision,
        session_state={"active_intent": "show_kundali", "pending_slots": ["birth_place"]},
    ) is True


def test_recommendation_readiness_stays_low_for_simple_guidance_message() -> None:
    readiness = ChatService._recommendation_readiness(
        semantic_understanding={
            "problem": "career",
            "severity": "low",
            "remedy_interest": False,
        },
        conversation_context={"main_concern": None},
        behavior_signals={
            "engagement": 0,
            "remedy_interest": 0,
            "explicit_request": 0,
            "avoidance": 0,
        },
        recent_messages=[],
    )

    assert readiness["score"] < 31
    assert readiness["recommendation_ready"] is False


def test_orchestrate_plan_blocks_product_when_readiness_is_low() -> None:
    plan = PlannerResult(
        action="recommend_product",
        confidence=0.92,
        arguments={"search_query": "career remedy"},
        missing_information=[],
        should_call_tool=True,
        reasoning="planner suggested product support",
    )

    updated, orchestration = ChatService._orchestrate_plan(
        plan=plan,
        message="My efforts are slow",
        session_state={},
        semantic_understanding={"problem": "career"},
        behavior_signals={"avoidance": 0},
        recommendation_readiness={"score": 20},
        birth_details=None,
        matchmaking_details=None,
    )

    assert updated.action == "respond_only"
    assert updated.should_call_tool is False
    assert orchestration["reason"] == "readiness_too_low"


def test_orchestrate_plan_blocks_duplicate_product_without_explicit_request() -> None:
    plan = PlannerResult(
        action="recommend_product",
        confidence=0.92,
        arguments={"search_query": "career remedy"},
        missing_information=[],
        should_call_tool=True,
        reasoning="planner suggested product support",
    )

    updated, orchestration = ChatService._orchestrate_plan(
        plan=plan,
        message="I still feel stuck",
        session_state={
            "main_concern": "career",
            "previous_tools": ["recommend_product"],
            "products_shown": ["prod-1"],
        },
        semantic_understanding={"problem": "career"},
        behavior_signals={"avoidance": 0},
        recommendation_readiness={"score": 45},
        birth_details=None,
        matchmaking_details=None,
    )

    assert updated.action == "respond_only"
    assert updated.should_call_tool is False
    assert orchestration["reason"] in {"recent_tool_repeat", "products_already_shown"}


def test_build_compact_session_state_tracks_recommendation_history() -> None:
    state = ChatService._build_compact_session_state(
        context={
            "plan": PlannerResult(
                action="recommend_product",
                confidence=0.92,
                arguments={"search_query": "career remedy"},
                missing_information=[],
                should_call_tool=True,
                reasoning="planner suggested product support",
            ),
            "effective_birth_details": None,
            "partial_birth_details": None,
            "matchmaking_details": None,
            "tool_outputs": [
                {
                    "tool": "recommend_product",
                    "summary": "Products available.",
                    "items": [{"id": "prod-1"}],
                }
            ],
            "message": "Show me products",
            "session_state": {
                "main_concern": "career",
                "previous_tools": ["suggest_consultant"],
                "products_shown": [],
                "services_shown": [],
                "consultation_history": [],
            },
            "semantic_understanding": {"problem": "career"},
        },
        reply="Here are a few options.",
    )

    assert state["main_concern"] == "career"
    assert "suggest_consultant" in state["previous_tools"]
    assert "recommend_product" in state["previous_tools"]
    assert "prod-1" in state["products_shown"]


def test_normalize_tool_outputs_returns_reasoned_payloads() -> None:
    normalized = normalize_tool_outputs(
        [
            {
                "tool": "recommend_product",
                "search_query": "career remedy",
                "items": [{"id": "prod-1", "name": "5 Mukhi Rudraksha"}],
            }
        ]
    )

    assert normalized == [
        {
            "type": "product",
            "reason": "career remedy",
            "items": [{"id": "prod-1", "name": "5 Mukhi Rudraksha"}],
        }
    ]


def test_build_birth_details_capture_reply_asks_only_for_missing_place() -> None:
    reply = ChatService._build_birth_details_capture_reply(
        "DOB is 20/06/2001 and time is 10:30 pm",
        {
            "date_parts": (20, 6, 2001),
            "time_parts": (22, 30),
            "place": None,
        },
    )

    assert reply == (
        "I have your date and time. "
        "For an exact chart, send the birthplace a bit more specifically, like city, state, country."
    )


def test_response_style_context_for_show_kundali_answers_actual_question() -> None:
    plan = PlannerResult(
        action="show_kundali",
        confidence=0.98,
        arguments={},
        missing_information=[],
        should_call_tool=True,
        reasoning="birth details available",
    )
    style = ChatService._build_response_style_context(
        message="What does astrology say about my career and financial prospects?",
        plan=plan,
        tool_outputs=[{"tool": "show_kundali", "summary": "Ascendant: Aries."}],
    )
    assert "Sound like a calm, traditional Vedic astrologer speaking directly to one person." in style
    assert "Answer the user's actual concern from the chart perspective" in style
    assert "Do not repeat every structured item in prose." in style
    assert "astrological insight about the user's concern" in style


def test_response_style_context_allows_soft_product_after_guidance() -> None:
    plan = PlannerResult(
        action="respond_only",
        confidence=0.88,
        arguments={},
        missing_information=[],
        should_call_tool=False,
        reasoning="general guidance",
    )
    style = ChatService._build_response_style_context(
        message="I am facing career delay and pressure.",
        plan=plan,
        tool_outputs=[{"tool": "recommend_product", "soft_recommendation": True}],
    )

    assert "supportive product option is available" in style
    assert "Do not introduce rudraksha, bracelets, or catalog products on your own." not in style


def test_response_style_context_tolerates_emotion_stub_without_intensity() -> None:
    plan = PlannerResult(
        action="respond_only",
        confidence=0.88,
        arguments={},
        missing_information=[],
        should_call_tool=False,
        reasoning="general guidance",
    )

    style = ChatService._build_response_style_context(
        message="I am facing career delay and pressure.",
        plan=plan,
        tool_outputs=[],
        emotion=type("Emotion", (), {"label": "calm", "emotion": "calm"})(),
    )

    assert "Sound like a calm, traditional Vedic astrologer speaking directly to one person." in style


def test_response_style_context_handles_empty_explicit_product_results() -> None:
    plan = PlannerResult(
        action="recommend_product",
        confidence=0.9,
        arguments={"search_query": "rudraksha career growth"},
        missing_information=[],
        should_call_tool=True,
        reasoning="explicit product request",
    )

    style = ChatService._build_response_style_context(
        message="Suggest a rudraksha for career growth",
        plan=plan,
        tool_outputs=[
            {
                "tool": "recommend_product",
                "search_query": "rudraksha career growth",
                "policy_note": "No matching catalog items were found for this request.",
                "items": [],
            }
        ],
    )

    assert "no exact product items" in style.lower()
    assert "do not imply" in style.lower()
    assert "pandit consultation" in style.lower()


def test_chunk_text_prefers_sentence_like_streaming_chunks() -> None:
    chunks = chunk_text(
        "Saturn is strong in this phase. This usually brings delay, pressure, and discipline. "
        "A simple remedy is steady prayer on Saturdays."
    )

    assert chunks == [
        "Saturn is strong in this phase. ",
        "This usually brings delay, pressure, and discipline. ",
        "A simple remedy is steady prayer on Saturdays.",
    ]


def test_build_fast_astrology_reply_asks_for_birth_details_for_personal_career_query() -> None:
    plan = PlannerResult(
        action="respond_only",
        confidence=0.86,
        arguments={},
        missing_information=[],
        should_call_tool=False,
        reasoning="fast personal astrology guidance",
    )
    reply = ChatService._build_fast_astrology_reply(
        message="What does astrology say about my career and financial prospects?",
        plan=plan,
        birth_details=None,
        matchmaking_details=None,
    )
    assert "10th house" in reply
    assert "birth" in reply.lower()


def test_stream_reply_events_runs_complete_context_for_fast_chat_guidance() -> None:
    plan = PlannerResult(
        action="respond_only",
        confidence=0.91,
        arguments={},
        missing_information=[],
        should_call_tool=False,
        reasoning="career guidance",
    )

    service = ChatService.__new__(ChatService)

    class _GroqStub:
        is_configured = False
        last_usage = None

    class _EmotionStub:
        label = "calm"
        emotion = "calm"

    completed = {"called": False}

    service.groq_client = _GroqStub()
    service._persist_chat_turns = lambda *args, **kwargs: None  # type: ignore[method-assign]
    service._persist_lightweight_memory = lambda *args, **kwargs: None  # type: ignore[method-assign]

    async def _prepare_base(**kwargs):  # noqa: ANN202
        return {
            "plan": plan,
            "route": type("Route", (), {"provider": "groq", "model": "test-model", "reasoning_profile": "fast-answer"})(),
            "emotion": _EmotionStub(),
            "messages": [],
            "tool_outputs": [],
            "retrieval_matches": [],
            "retrieval_knowledge_matches": [],
            "retrieval_policy_matches": [],
            "retrieval_metadata": {},
            "kundali_chart": None,
            "kundali_summary": None,
            "matchmaking_result": None,
            "metadata_json": None,
            "message": kwargs["message"],
            "session_id": kwargs["session_id"],
            "scope_guardrail": {"allowed": True, "reason": "astrology_service_action"},
            "tool_guardrail": {"allowed": False, "reason": "planner_declined_tool"},
            "tool_execution_allowed": False,
            "birth_details_followup": False,
            "birth_details_capture_pending": False,
            "effective_birth_details": None,
            "matchmaking_details": None,
            "recent_messages": [],
            "internal_user_id": None,
            "route_decision": ChatRouteDecision(
                route="FAST_CHAT",
                intent="respond_only",
                confidence=0.91,
                risk_level="low",
                reason="astrology_qa",
                should_call_tool=False,
                needs_planner=False,
            ),
            "normalized_message": kwargs["message"],
            "recommendation_context": {
                "soft_product": {"eligible": False, "reason": "not_evaluated", "query": None}
            },
        }

    async def _complete(context, **kwargs):  # noqa: ANN202
        del kwargs
        completed["called"] = True
        enriched = dict(context)
        enriched.update(
            {
                "messages": [{"role": "system", "content": "rag aware"}],
                "tool_outputs": [],
                "retrieval_matches": [
                    {
                        "title": "Saturn remedies",
                        "excerpt": "Saturn remedies focus on steady discipline and Saturday practices.",
                        "path": "data/astrology_texts/planetary_remedies.txt",
                        "score": 0.91,
                        "metadata": {
                            "domain": "remedy_guidance",
                            "risk": "low",
                            "allowed_actions": ["explain_only"],
                        },
                    }
                ],
                "retrieval_knowledge_matches": [
                    {
                        "title": "Saturn remedies",
                        "excerpt": "Saturn remedies focus on steady discipline and Saturday practices.",
                        "path": "data/astrology_texts/planetary_remedies.txt",
                        "score": 0.91,
                        "metadata": {
                            "domain": "remedy_guidance",
                            "risk": "low",
                            "allowed_actions": ["explain_only"],
                        },
                    }
                ],
                "retrieval_policy_matches": [],
                "retrieval_metadata": {
                    "provider": "embedding_store",
                    "retrieval_strategy": "db_embedding_hybrid_v3",
                    "embedding_provider": "precomputed",
                    "embedding_model": "local-hash-v1",
                    "vector_backend": "json_scan",
                    "keyword_backend": "keyword_scan",
                    "reranker_provider": "heuristic",
                    "reranker_model": "heuristic-v1",
                    "chart_context_used": False,
                    "document_count": 1,
                },
                "kundali_chart": None,
                "kundali_summary": None,
                "matchmaking_result": None,
            }
        )
        return enriched

    service._prepare_base_reply_context = _prepare_base  # type: ignore[method-assign]
    service._complete_reply_context = _complete  # type: ignore[method-assign]

    async def _collect() -> list[tuple[str, dict[str, object]]]:
        events: list[tuple[str, dict[str, object]]] = []
        async for item in service.stream_reply_events(
            session_id="session-1",
            message="I am confused about my career and Saturn remedies.",
        ):
            events.append(item)
        return events

    events = asyncio.run(_collect())
    done_payload = next(payload for name, payload in events if name == "done")

    assert completed["called"] is True
    assert done_payload["metadata"]["retrieval_trace"]["provider"] == "embedding_store"


def test_pre_scope_guardrail_blocks_curse_query() -> None:
    decision = pre_scope_guardrail("Am I cursed?")
    assert decision.allowed is False
    assert decision.reason == "curse_fear"


def test_pre_scope_guardrail_blocks_enemy_pooja_query() -> None:
    decision = pre_scope_guardrail("Which pooja will destroy my enemy?")
    assert decision.allowed is False
    assert decision.reason == "harm_or_manipulation"


def test_pre_scope_guardrail_blocks_medical_claims() -> None:
    decision = pre_scope_guardrail("Can gemstone cure cancer?")
    assert decision.allowed is False
    assert decision.reason == "medical_claim"


def test_pre_scope_guardrail_allows_mantra_for_peace() -> None:
    decision = pre_scope_guardrail("Suggest mantra for peace")
    assert decision.allowed is True
    assert decision.reason == "allowed"


def test_classify_route_fast_chat_for_rahu_question() -> None:
    decision = classify_route(
        message="What is Rahu mahadasha?",
        birth_details=None,
        matchmaking_details=None,
        pre_guardrail=pre_scope_guardrail("What is Rahu mahadasha?"),
    )
    assert decision.route == "FAST_CHAT"
    assert decision.intent == "respond_only"
    assert decision.should_call_tool is False
    assert decision.needs_planner is True


def test_classify_route_default_fast_chat_defers_to_planner() -> None:
    decision = classify_route(
        message="Please guide me about what is happening in life.",
        birth_details=None,
        matchmaking_details=None,
        pre_guardrail=pre_scope_guardrail("Please guide me about what is happening in life."),
    )
    assert decision.route == "FAST_CHAT"
    assert decision.intent == "respond_only"
    assert decision.needs_planner is True


def test_classify_route_tool_flow_for_show_kundali_with_birth_details() -> None:
    decision = classify_route(
        message="Show my kundali",
        birth_details={"birth_datetime": "2001-06-20T22:22:00"},
        matchmaking_details=None,
        pre_guardrail=pre_scope_guardrail("Show my kundali"),
    )
    assert decision.route == "TOOL_FLOW"
    assert decision.intent == "show_kundali"
    assert decision.should_call_tool is True


def test_classify_route_clarification_for_show_kundali_without_birth_details() -> None:
    decision = classify_route(
        message="Show my kundali",
        birth_details=None,
        matchmaking_details=None,
        pre_guardrail=pre_scope_guardrail("Show my kundali"),
    )
    assert decision.route == "CLARIFICATION"
    assert decision.intent == "show_kundali"
    assert decision.missing_slots == ["birth_details"]


def test_classify_route_clarification_for_matchmaking_without_partner_details() -> None:
    decision = classify_route(
        message="Match my kundali with partner",
        birth_details=None,
        matchmaking_details=None,
        pre_guardrail=pre_scope_guardrail("Match my kundali with partner"),
    )
    assert decision.route == "CLARIFICATION"
    assert decision.intent == "matchmaking"
    assert decision.missing_slots == ["matchmaking_details"]


def test_classify_route_tool_flow_for_matchmaking_with_details() -> None:
    decision = classify_route(
        message="Please check our matchmaking",
        birth_details=None,
        matchmaking_details={"primary": {}, "partner": {}},
        pre_guardrail=pre_scope_guardrail("Please check our matchmaking"),
    )
    assert decision.route == "TOOL_FLOW"
    assert decision.intent == "matchmaking"
    assert decision.should_call_tool is True


def test_classify_route_tool_flow_for_booking() -> None:
    decision = classify_route(
        message="Book Satyanarayan pooja at home",
        birth_details=None,
        matchmaking_details=None,
        pre_guardrail=pre_scope_guardrail("Book Satyanarayan pooja at home"),
    )
    assert decision.route == "TOOL_FLOW"
    assert decision.intent == "book_pooja"
    assert decision.should_call_tool is True


def test_needs_rag_for_tool_supporting_actions() -> None:
    route_decision = ChatRouteDecision(
        route="TOOL_FLOW",
        intent="recommend_product",
        confidence=0.95,
        risk_level="low",
        reason="product_request",
        should_call_tool=True,
        normalized_args={"search_query": "rudraksha career"},
    )
    plan = PlannerResult(
        action="recommend_product",
        confidence=0.95,
        arguments={"search_query": "rudraksha career"},
        missing_information=[],
        should_call_tool=True,
        reasoning="product guidance requested",
    )

    assert ChatService._needs_rag(route_decision, plan) is True


def test_policy_match_can_enable_soft_product_recommendation() -> None:
    assert ChatService._policy_allows_product_recommendation(
        [
            {
                "metadata": {
                    "domain": "remedy_guidance",
                    "allowed_actions": ["can_recommend", "explain_only"],
                }
            }
        ]
    ) is True


def test_infer_soft_product_query_maps_career_issue_to_catalog_friendly_query() -> None:
    assert ChatService._infer_soft_product_query(
        message="I have career delay and too many obstacles, any remedy?",
        kundali_summary=None,
    ) == "rudraksha career"


def test_should_offer_soft_product_for_general_guidance_when_policy_allows() -> None:
    plan = PlannerResult(
        action="respond_only",
        confidence=0.86,
        arguments={},
        missing_information=[],
        should_call_tool=False,
        reasoning="general astrology guidance",
    )

    assert ChatService._should_offer_soft_product(
        message="I have a career issue and feel stuck, any remedy?",
        plan=plan,
        retrieval_policy_matches=[
            {
                "metadata": {
                    "domain": "product_policy",
                    "allowed_actions": ["recommend_product"],
                }
            }
        ],
        kundali_summary=None,
    ) is True


def test_should_not_offer_soft_product_when_feature_flag_disabled(monkeypatch) -> None:
    plan = PlannerResult(
        action="respond_only",
        confidence=0.86,
        arguments={},
        missing_information=[],
        should_call_tool=False,
        reasoning="general astrology guidance",
    )

    monkeypatch.setattr("src.core.chat_service.settings.SOFT_PRODUCT_RECOMMENDATIONS_ENABLED", False)

    assert ChatService._should_offer_soft_product(
        message="I have a career issue and feel stuck.",
        plan=plan,
        retrieval_policy_matches=[
            {
                "metadata": {
                    "domain": "product_policy",
                    "allowed_actions": ["recommend_product"],
                }
            }
        ],
        kundali_summary=None,
    ) is False


def test_response_metadata_includes_compact_retrieval_trace() -> None:
    route_decision = ChatRouteDecision(
        route="TOOL_FLOW",
        intent="recommend_product",
        confidence=0.95,
        risk_level="low",
        reason="product_request",
        should_call_tool=True,
    )
    plan = PlannerResult(
        action="recommend_product",
        confidence=0.95,
        arguments={"search_query": "rudraksha career"},
        missing_information=[],
        should_call_tool=True,
        reasoning="product guidance requested",
    )

    response = ChatService._response_metadata(
        reply="A five mukhi rudraksha is usually suggested for steadiness.",
        route_decision=route_decision,
        plan=plan,
        message="Suggest a rudraksha for career support.",
        tool_outputs=[
            {
                "tool": "recommend_product",
                "search_query": "rudraksha career",
                "items": [{"name": "5 Mukhi Rudraksha"}],
            }
        ],
        latency_ms=182,
        model="test-model",
        retrieval_metadata={
            "provider": "embedding_store",
            "retrieval_strategy": "db_embedding_hybrid_v3",
            "embedding_provider": "precomputed",
            "embedding_model": "local-hash-v1",
            "vector_backend": "json_scan",
            "keyword_backend": "keyword_scan",
            "reranker_provider": "heuristic",
            "reranker_model": "heuristic-v1",
            "chart_context_used": False,
            "document_count": 14,
        },
        retrieval_matches=[
            {
                "source": "Product policy guide",
                "type": "policy",
                "path": "data/policy/product.md",
                "score": 0.91421,
                "metadata": {
                    "domain": "product_policy",
                    "risk": "low",
                    "allowed_actions": ["recommend_product"],
                    "chunk_id": "policy-1",
                },
            },
            {
                "source": "Rudraksha notes",
                "type": "knowledge",
                "path": "data/knowledge/rudraksha.md",
                "score": 0.87391,
                "metadata": {
                    "domain": "remedy_reference",
                    "risk": "low",
                    "allowed_actions": ["explain_only"],
                    "chunk_id": "knowledge-4",
                },
            },
        ],
        retrieval_knowledge_matches=[
            {
                "source": "Rudraksha notes",
                "type": "knowledge",
                "path": "data/knowledge/rudraksha.md",
                "score": 0.87391,
                "metadata": {
                    "domain": "remedy_reference",
                    "risk": "low",
                    "allowed_actions": ["explain_only"],
                    "chunk_id": "knowledge-4",
                },
            }
        ],
        retrieval_policy_matches=[
            {
                "source": "Product policy guide",
                "type": "policy",
                "path": "data/policy/product.md",
                "score": 0.91421,
                "metadata": {
                    "domain": "product_policy",
                    "risk": "low",
                    "allowed_actions": ["recommend_product"],
                    "chunk_id": "policy-1",
                },
            }
        ],
        recommendation_context={
            "soft_product": {"eligible": False, "reason": "plan_ineligible", "query": None}
        },
    )

    retrieval_trace = response["metadata"]["retrieval_trace"]
    product_trace = response["metadata"]["product_recommendation_trace"]

    assert retrieval_trace["match_count"] == 2
    assert retrieval_trace["knowledge_match_count"] == 1
    assert retrieval_trace["policy_match_count"] == 1
    assert retrieval_trace["provider"] == "embedding_store"
    assert retrieval_trace["strategy"] == "db_embedding_hybrid_v3"
    assert retrieval_trace["embedding_provider"] == "precomputed"
    assert retrieval_trace["embedding_model"] == "local-hash-v1"
    assert retrieval_trace["vector_backend"] == "json_scan"
    assert retrieval_trace["keyword_backend"] == "keyword_scan"
    assert retrieval_trace["reranker_provider"] == "heuristic"
    assert retrieval_trace["reranker_model"] == "heuristic-v1"
    assert retrieval_trace["chart_context_used"] is False
    assert retrieval_trace["document_count"] == 14
    assert retrieval_trace["knowledge"][0] == {
        "source": "Rudraksha notes",
        "path": "data/knowledge/rudraksha.md",
        "domain": "remedy_reference",
        "score": 0.8739,
        "risk": "low",
        "allowed_actions": ["explain_only"],
        "chunk_id": "knowledge-4",
    }
    assert retrieval_trace["policy"][0] == {
        "source": "Product policy guide",
        "path": "data/policy/product.md",
        "domain": "product_policy",
        "score": 0.9142,
        "risk": "low",
        "allowed_actions": ["recommend_product"],
        "chunk_id": "policy-1",
    }
    assert product_trace["mode"] == "explicit"
    assert product_trace["presented"] is True
    assert product_trace["result_count"] == 1
    assert product_trace["search_query"] == "rudraksha career"
    assert product_trace["item_names"] == ["5 Mukhi Rudraksha"]


def test_llm_trace_metadata_includes_retrieval_trace() -> None:
    context = {
        "plan": PlannerResult(
            action="book_pooja",
            confidence=0.91,
            arguments={"search_query": "satyanarayan puja"},
            missing_information=[],
            should_call_tool=True,
            reasoning="booking flow",
        ),
        "route_decision": ChatRouteDecision(
            route="TOOL_FLOW",
            intent="book_pooja",
            confidence=0.91,
            risk_level="low",
            reason="booking_request",
            should_call_tool=True,
        ),
        "tool_execution_allowed": True,
        "tool_outputs": [{"tool": "book_pooja"}],
        "message": "Book a Satyanarayan puja for home.",
        "retrieval_metadata": {
            "provider": "embedding_store",
            "retrieval_strategy": "db_embedding_hybrid_v3",
            "embedding_provider": "precomputed",
            "embedding_model": "local-hash-v1",
            "vector_backend": "json_scan",
            "keyword_backend": "keyword_scan",
            "reranker_provider": "heuristic",
            "reranker_model": "heuristic-v1",
            "chart_context_used": True,
            "document_count": 8,
        },
        "retrieval_knowledge_matches": [
            {
                "source": "Satyanarayan basics",
                "path": "data/knowledge/puja.md",
                "score": 0.81,
                "metadata": {"domain": "puja_reference"},
            }
        ],
        "retrieval_policy_matches": [
            {
                "source": "Booking policy",
                "path": "data/policy/booking.md",
                "score": 0.92,
                "metadata": {"domain": "booking_guidance", "allowed_actions": ["book_pooja"]},
            }
        ],
        "recommendation_context": {
            "soft_product": {"eligible": False, "reason": "plan_ineligible", "query": None}
        },
    }

    trace_metadata = ChatService._llm_trace_metadata(context)

    assert trace_metadata["intent"] == "book_pooja"
    assert trace_metadata["route"] == "TOOL_FLOW"
    assert trace_metadata["tool_execution_allowed"] is True
    assert trace_metadata["tool_count"] == 1
    assert trace_metadata["retrieval_trace"]["provider"] == "embedding_store"
    assert trace_metadata["retrieval_trace"]["strategy"] == "db_embedding_hybrid_v3"
    assert trace_metadata["retrieval_trace"]["embedding_provider"] == "precomputed"
    assert trace_metadata["retrieval_trace"]["vector_backend"] == "json_scan"
    assert trace_metadata["retrieval_trace"]["keyword_backend"] == "keyword_scan"
    assert trace_metadata["retrieval_trace"]["reranker_provider"] == "heuristic"
    assert trace_metadata["retrieval_trace"]["reranker_model"] == "heuristic-v1"
    assert trace_metadata["retrieval_trace"]["chart_context_used"] is True
    assert trace_metadata["retrieval_trace"]["knowledge_match_count"] == 1
    assert trace_metadata["retrieval_trace"]["policy_match_count"] == 1
    assert trace_metadata["product_recommendation_trace"]["mode"] == "none"
    assert trace_metadata["product_recommendation_trace"]["soft_reason"] == "plan_ineligible"


def test_postprocess_reply_keeps_general_guidance_before_birth_details_prompt() -> None:
    plan = PlannerResult(
        action="respond_only",
        confidence=0.86,
        arguments={},
        missing_information=[],
        should_call_tool=False,
        reasoning="fast personal astrology guidance",
    )

    reply = ChatService._postprocess_reply(
        reply=(
            "Career pressure usually shows a phase of delay, unclear direction, or low confidence rather than a permanent block. "
            "In astrology, the 10th house, Saturn, and current timing are usually checked for this. "
            "If you want an exact chart-based answer, share your birth details."
        ),
        plan=plan,
        message="What does astrology say about my career and financial prospects?",
    )

    # _postprocess_reply compacts respond_only to 2 sentences; the core
    # guidance is preserved even when the third sentence is trimmed.
    assert "10th house" in reply
    assert "career pressure" in reply.lower()


def test_route_name_for_plan_maps_tool_actions_to_tool_flow() -> None:
    plan = PlannerResult(
        action="recommend_product",
        confidence=0.92,
        arguments={"search_query": "rudraksha career"},
        missing_information=[],
        should_call_tool=True,
        reasoning="planner picked product support",
    )

    assert ChatService._route_name_for_plan(plan) == "TOOL_FLOW"


def test_complete_reply_context_allows_deferred_planner_to_enable_tool_flow() -> None:
    service = ChatService.__new__(ChatService)
    service.settings = type(
        "SettingsStub",
        (),
        {
            "FAST_RAG_TOP_K": 2,
            "RAG_TOP_K": 5,
            "TOOL_TIMEOUT_SECONDS": 1,
        },
    )()

    class _MemoryStub:
        def long_term_context(self, session_id: str, user_id=None) -> str:  # noqa: ANN001, ANN202
            del session_id, user_id
            return "career concern"

    class _RagStub:
        def retrieve_context_bundle(self, *args, **kwargs):  # noqa: ANN202
            del args, kwargs
            return {"chunks": [], "knowledge_chunks": [], "policy_chunks": [], "retrieval_metadata": {}}

    class _PlannerStub:
        async def plan(self, **kwargs):  # noqa: ANN202
            del kwargs
            return PlannerResult(
                action="recommend_product",
                confidence=0.94,
                arguments={"search_query": "rudraksha career"},
                missing_information=[],
                should_call_tool=True,
                reasoning="planner resolved product guidance",
            )

    class _CoreStub:
        async def search_products(self, query: str):  # noqa: ANN202
            assert query == "rudraksha career"
            return [{"id": "prod-1", "name": "5 Mukhi Rudraksha"}]

    service.memory_service = _MemoryStub()
    service.rag_service = _RagStub()
    service.planner = _PlannerStub()
    service.core_service_client = _CoreStub()

    context = {
        "scope_guardrail": {"allowed": True, "reason": "astrology_qa"},
        "session_id": "session-1",
        "message": "I have career obstacles. What should I do?",
        "plan": PlannerResult(
            action="respond_only",
            confidence=0.86,
            arguments={},
            missing_information=[],
            should_call_tool=False,
            reasoning="astrology_qa",
        ),
        "recent_messages": [],
        "internal_user_id": None,
        "tool_execution_allowed": False,
        "route_decision": ChatRouteDecision(
            route="FAST_CHAT",
            intent="respond_only",
            confidence=0.86,
            risk_level="low",
            reason="astrology_qa",
            should_call_tool=False,
            needs_planner=True,
        ),
        "normalized_message": "I have career obstacles. What should I do?",
        "deferred_planner": True,
        "effective_birth_details": None,
        "route": type("Route", (), {"provider": "groq", "model": "test-model", "reasoning_profile": "fast-answer"})(),
        "emotion": type("Emotion", (), {"label": "calm", "emotion": "calm"})(),
        "tool_outputs": [],
        "retrieval_matches": [],
        "kundali_chart": None,
        "kundali_summary": None,
        "matchmaking_result": None,
        "metadata_json": None,
        "birth_details_followup": False,
        "partial_birth_details": None,
        "needs_birth_details": False,
        "matchmaking_details": None,
        "current_user": None,
    }

    enriched = asyncio.run(service._complete_reply_context(context))

    assert enriched["route_decision"].route == "TOOL_FLOW"
    assert enriched["tool_execution_allowed"] is True
    assert enriched["route"].reasoning_profile == "tool-aware"
    assert enriched["tool_outputs"][0]["tool"] == "recommend_product"


def test_complete_reply_context_passes_chart_context_into_rag_when_birth_details_exist() -> None:
    service = ChatService.__new__(ChatService)
    service.settings = type(
        "SettingsStub",
        (),
        {
            "FAST_RAG_TOP_K": 2,
            "RAG_TOP_K": 5,
            "TOOL_TIMEOUT_SECONDS": 1,
        },
    )()

    class _MemoryStub:
        def long_term_context(self, session_id: str, user_id=None) -> str:  # noqa: ANN001, ANN202
            del session_id, user_id
            return "career concern"

    captured: dict[str, object] = {}

    class _RagStub:
        def retrieve_context_bundle(self, *args, **kwargs):  # noqa: ANN202
            captured["kwargs"] = kwargs
            return {"chunks": [], "knowledge_chunks": [], "policy_chunks": [], "retrieval_metadata": {}}

    service.memory_service = _MemoryStub()
    service.rag_service = _RagStub()
    service.planner = None
    service.core_service_client = type("CoreStub", (), {})()

    async def _compute_chart_context(_birth_details):  # noqa: ANN001, ANN202
        return {
            "chart": {"ascendant_sign_name": "Aries"},
            "rag_context": {
                "ascendant_sign": "Aries",
                "moon_sign": "Cancer",
                "current_mahadasha": "Saturn",
                "current_antardasha": "Jupiter",
                "placements": [{"planet": "saturn", "house": 10, "sign": "capricorn"}],
                "astro_entities": {
                    "planets": ["saturn"],
                    "houses": [10],
                    "signs": ["capricorn"],
                    "nakshatras": [],
                    "dashas": ["saturn", "jupiter"],
                },
            },
        }

    service._compute_rag_chart_context = _compute_chart_context  # type: ignore[method-assign]

    context = {
        "scope_guardrail": {"allowed": True, "reason": "astrology_qa"},
        "session_id": "session-1",
        "message": "Will I get a job change soon?",
        "plan": PlannerResult(
            action="respond_only",
            confidence=0.88,
            arguments={},
            missing_information=[],
            should_call_tool=False,
            reasoning="career guidance",
        ),
        "recent_messages": [],
        "internal_user_id": None,
        "tool_execution_allowed": False,
        "route_decision": ChatRouteDecision(
            route="FAST_CHAT",
            intent="respond_only",
            confidence=0.88,
            risk_level="low",
            reason="astrology_qa",
            should_call_tool=False,
            needs_planner=False,
        ),
        "normalized_message": "Will I get a job change soon?",
        "deferred_planner": False,
        "effective_birth_details": {"birth_datetime": "1990-01-01T10:00:00"},
        "route": type("Route", (), {"provider": "groq", "model": "test-model", "reasoning_profile": "fast-answer"})(),
        "emotion": type("Emotion", (), {"label": "calm", "emotion": "calm"})(),
        "tool_outputs": [],
        "retrieval_matches": [],
        "kundali_chart": None,
        "kundali_summary": None,
        "matchmaking_result": None,
        "metadata_json": None,
        "birth_details_followup": False,
        "partial_birth_details": None,
        "needs_birth_details": False,
        "matchmaking_details": None,
        "current_user": None,
    }

    asyncio.run(service._complete_reply_context(context))

    chart_context = captured["kwargs"]["chart_context"]
    assert isinstance(chart_context, dict)
    assert chart_context["current_mahadasha"] == "Saturn"
    assert chart_context["placements"][0]["house"] == 10


def test_complete_reply_context_uses_compact_session_context_and_trims_recent_history() -> None:
    service = ChatService.__new__(ChatService)
    service.settings = type(
        "SettingsStub",
        (),
        {
            "FAST_RAG_TOP_K": 2,
            "RAG_TOP_K": 5,
            "TOOL_TIMEOUT_SECONDS": 1,
        },
    )()

    class _MemoryStub:
        def long_term_context(self, session_id: str, user_id=None) -> str:  # noqa: ANN001, ANN202
            del session_id, user_id
            return "career concern"

    class _RagStub:
        def retrieve_context_bundle(self, *args, **kwargs):  # noqa: ANN202
            del args, kwargs
            return {"chunks": [], "knowledge_chunks": [], "policy_chunks": [], "retrieval_metadata": {}}

    service.memory_service = _MemoryStub()
    service.rag_service = _RagStub()
    service.planner = None
    service.core_service_client = type("CoreStub", (), {})()

    recent_messages = [
        {"role": "user", "content": f"old-{idx}"}
        for idx in range(6)
    ]
    context = {
        "scope_guardrail": {"allowed": True, "reason": "astrology_qa"},
        "session_id": "session-1",
        "message": "What next for my career?",
        "plan": PlannerResult(
            action="respond_only",
            confidence=0.88,
            arguments={},
            missing_information=[],
            should_call_tool=False,
            reasoning="career guidance",
        ),
        "recent_messages": recent_messages,
        "internal_user_id": None,
        "tool_execution_allowed": False,
        "route_decision": ChatRouteDecision(
            route="FAST_CHAT",
            intent="respond_only",
            confidence=0.88,
            risk_level="low",
            reason="astrology_qa",
            should_call_tool=False,
            needs_planner=False,
        ),
        "normalized_message": "What next for my career?",
        "deferred_planner": False,
        "effective_birth_details": {"birth_datetime": "1990-01-01T10:00:00"},
        "route": type("Route", (), {"provider": "groq", "model": "test-model", "reasoning_profile": "fast-answer"})(),
        "emotion": type("Emotion", (), {"label": "calm", "emotion": "calm"})(),
        "tool_outputs": [],
        "retrieval_matches": [],
        "kundali_chart": None,
        "kundali_summary": None,
        "matchmaking_result": None,
        "metadata_json": None,
        "birth_details_followup": False,
        "partial_birth_details": None,
        "needs_birth_details": False,
        "matchmaking_details": None,
        "current_user": None,
        "session_state": {
            "active_intent": "show_kundali",
            "birth_details": {"birth_datetime": "1990-01-01T10:00:00"},
            "pending_slots": [],
            "last_tool": "show_kundali",
            "last_tool_summary": "Leo ascendant with Saturn influence.",
            "last_user_goal": "career guidance",
        },
    }

    enriched = asyncio.run(service._complete_reply_context(context))

    compact_system_messages = [
        item["content"]
        for item in enriched["messages"]
        if item["role"] == "system" and "Compact session state:" in item["content"]
    ]
    assert len(compact_system_messages) == 1
    assert "Last tool: show_kundali | Leo ascendant with Saturn influence." in compact_system_messages[0]
    assert enriched["compact_recent_messages"] == recent_messages[-4:]
    trimmed_history = [
        item["content"]
        for item in enriched["messages"]
        if item["role"] != "system"
    ]
    assert trimmed_history[:-1] == [msg["content"] for msg in recent_messages[-4:]]
    assert trimmed_history[-1] == "What next for my career?"


def test_product_recommendation_trace_captures_soft_product_exposure() -> None:
    plan = PlannerResult(
        action="respond_only",
        confidence=0.88,
        arguments={},
        missing_information=[],
        should_call_tool=False,
        reasoning="general guidance",
    )

    trace = ChatService._build_product_recommendation_trace(
        message="I have career delay and pressure.",
        plan=plan,
        tool_outputs=[
            {
                "tool": "recommend_product",
                "soft_recommendation": True,
                "search_query": "rudraksha career",
                "policy_note": "Mention products only if they appear in these catalog results, and keep them optional support.",
                "items": [{"name": "7 Mukhi Rudraksha"}, {"name": "Saturn Protection Bracelet"}],
            }
        ],
        recommendation_context={
            "soft_product": {
                "eligible": True,
                "reason": "soft_product_added",
                "query": "rudraksha career",
            }
        },
    )

    assert trace["mode"] == "soft"
    assert trace["presented"] is True
    assert trace["result_count"] == 2
    assert trace["soft_eligible"] is True
    assert trace["soft_reason"] == "soft_product_added"
    assert trace["soft_query"] == "rudraksha career"
    assert trace["search_query"] == "rudraksha career"


def test_tool_specific_guardrail_blocks_black_magic_product_query() -> None:
    decision = tool_specific_guardrail(
        "recommend_product",
        "Suggest product to remove black magic",
        {"search_query": "remove black magic"},
    )
    assert decision.allowed is False
    assert decision.reason == "fear_based_monetization_block"


def test_tool_specific_guardrail_allows_safe_product_query() -> None:
    decision = tool_specific_guardrail(
        "recommend_product",
        "Suggest rudraksha for focus",
        {"search_query": "rudraksha   for   focus"},
    )
    assert decision.allowed is True
    assert decision.normalized_args["search_query"] == "rudraksha for focus"


def test_final_response_guardrail_softens_guaranteed_language() -> None:
    softened = final_response_guardrail(
        "You are cursed and this pooja will fix everything. You must buy this."
    )
    assert "you are cursed" not in softened.lower()
    assert "fix everything" not in softened.lower()
    assert "must buy this" not in softened.lower()


def test_finalize_reply_text_compacts_show_kundali_replies() -> None:
    plan = PlannerResult(
        action="show_kundali",
        confidence=0.99,
        arguments={},
        missing_information=[],
        should_call_tool=True,
        reasoning="chart available",
    )

    reply = ChatService._finalize_reply_text(
        reply=(
            "Your chart shows steady career growth through disciplined effort. "
            "Saturn points to slow but stable progress when you stay consistent. "
            "Mercury supports planning and communication in work. "
            "This also suggests you should avoid comparing your timeline with others."
        ),
        plan=plan,
        message="What does my chart say about career?",
    )

    assert reply.count(".") <= 3


def test_finalize_reply_text_allows_more_depth_when_user_asks_for_detail() -> None:
    plan = PlannerResult(
        action="respond_only",
        confidence=0.99,
        arguments={},
        missing_information=[],
        should_call_tool=False,
        reasoning="general astrology answer",
    )

    reply = ChatService._finalize_reply_text(
        reply=(
            "Saturn is important because it represents discipline and delayed maturity. "
            "It often shows where effort feels slow at first. "
            "That same area can become strong with patience. "
            "In astrology this is why Saturn is linked with responsibility and long-term structure. "
            "So Saturn is not only about struggle, it is also about durable growth."
        ),
        plan=plan,
        message="Explain Saturn in detail",
    )

    assert reply.count(".") <= 5


def test_finalize_reply_text_keeps_only_one_question() -> None:
    plan = PlannerResult(
        action="ask_clarification",
        confidence=0.99,
        arguments={},
        missing_information=[],
        should_call_tool=False,
        reasoning="needs clarification",
    )

    reply = ChatService._finalize_reply_text(
        reply="I understand. Is this more about career timing? Or is it more about confidence? What feels strongest right now?",
        plan=plan,
        message="I am confused about career",
    )

    assert reply.count("?") == 1


def test_tool_backed_reply_uses_less_robotic_kundali_phrasing() -> None:
    plan = PlannerResult(
        action="show_kundali",
        confidence=0.99,
        arguments={},
        missing_information=[],
        should_call_tool=True,
        reasoning="chart available",
    )

    reply = ChatService._build_tool_backed_reply(
        message="Show my kundali",
        plan=plan,
        tool_outputs=[{"tool": "show_kundali", "summary": "Ascendant: Aries."}],
    )

    assert reply is not None
    assert "prepared" not in reply.lower()
    assert "summary is below" in reply.lower()


def test_format_tool_context_lists_items_and_reference_instruction() -> None:
    context = ChatService._format_tool_context(
        [
            {
                "tool": "recommend_product",
                "summary": "Relevant products from the Digveda catalog: 5 Mukhi Rudraksha.",
                "items": [{"name": "5 Mukhi Rudraksha"}],
            }
        ]
    )

    assert "Items shown to user:" in context
    assert "- 5 Mukhi Rudraksha" in context
    assert "MUST reference these items" in context


def test_format_tool_context_includes_empty_result_guardrail() -> None:
    context = ChatService._format_tool_context(
        [
            {
                "tool": "recommend_product",
                "summary": "No matching product catalog results were found for 'rudraksha career growth'.",
                "policy_note": "No matching catalog items were found for this request.",
                "items": [],
            }
        ]
    )

    assert "Tool policy: No matching catalog items were found for this request." in context
    assert "No items were returned by this tool." in context
    assert "Do NOT imply" in context


def test_verify_card_text_consistency_warns_on_mismatch(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        ChatService._verify_card_text_consistency(
            reply="These remedies may help bring steadiness.",
            tool_outputs=[
                {
                    "tool": "recommend_product",
                    "items": [{"name": "5 Mukhi Rudraksha"}],
                }
            ],
            conversation_id="session-1",
        )

    assert "card_text_mismatch" in caplog.text


def test_verify_card_text_consistency_allows_named_reference(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        ChatService._verify_card_text_consistency(
            reply="The 5 Mukhi Rudraksha is the most relevant option here.",
            tool_outputs=[
                {
                    "tool": "recommend_product",
                    "items": [{"name": "5 Mukhi Rudraksha"}],
                }
            ],
            conversation_id="session-1",
        )

    assert "card_text_mismatch" not in caplog.text


def test_build_cards_returns_product_and_service_cards() -> None:
    cards = build_cards(
        [
            {
                "tool": "recommend_product",
                "items": [{"id": "prod-1", "name": "Rudraksha", "image_url": "https://img"}],
            },
            {
                "tool": "book_pooja",
                "home_puja_services": [{"id": "svc-1", "name": "Satyanarayan Puja"}],
                "temple_services": [],
                "pandits": [],
            },
        ]
    )
    assert cards[0]["type"] == "product"
    assert cards[0]["id"] == "prod-1"
    assert cards[1]["type"] == "service"
    assert cards[1]["id"] == "svc-1"


def test_stream_reply_events_emits_status_then_tool_event_before_message() -> None:
    plan = PlannerResult(
        action="book_pooja",
        confidence=0.93,
        arguments={"search_query": "satyanarayan puja home"},
        missing_information=[],
        should_call_tool=True,
        reasoning="booking guidance requested",
    )

    service = ChatService.__new__(ChatService)

    class _GroqStub:
        is_configured = False
        last_usage = None

    class _MemoryStub:
        class repository:  # noqa: N801
            @staticmethod
            def upsert_fact(*args, **kwargs):  # noqa: ANN202
                del args, kwargs

        def recent_messages(self, session_id: str, limit: int) -> list[dict[str, str]]:
            del session_id, limit
            return []

    class _EmotionStub:
        label = "calm"
        emotion = "calm"

    service.groq_client = _GroqStub()
    service.memory_service = _MemoryStub()
    service._persist_chat_turns = lambda *args, **kwargs: None  # type: ignore[method-assign]

    async def _background(*args, **kwargs):  # noqa: ANN202
        del args, kwargs

    async def _prepare_base(**kwargs):  # noqa: ANN202
        return {
            "plan": plan,
            "route": type("Route", (), {"provider": "groq", "model": "test-model", "reasoning_profile": "tool-aware"})(),
            "emotion": _EmotionStub(),
            "messages": [],
            "tool_outputs": [],
            "retrieval_matches": [],
            "kundali_chart": None,
            "kundali_summary": None,
            "matchmaking_result": None,
            "metadata_json": None,
            "message": kwargs["message"],
            "session_id": kwargs["session_id"],
            "scope_guardrail": {"allowed": True, "reason": "astrology_service_action"},
            "tool_guardrail": {"allowed": True, "reason": "passed", "search_query": "satyanarayan puja home"},
            "tool_execution_allowed": True,
            "birth_details_followup": False,
            "birth_details_capture_pending": False,
            "effective_birth_details": None,
            "matchmaking_details": None,
            "recent_messages": [],
            "internal_user_id": None,
            "route_decision": ChatRouteDecision(
                route="TOOL_FLOW",
                intent="book_pooja",
                confidence=0.93,
                risk_level="low",
                reason="explicit_booking_request",
                should_call_tool=True,
                normalized_args={"search_query": "satyanarayan puja home"},
            ),
            "normalized_message": kwargs["message"],
        }

    async def _complete(context, **kwargs):  # noqa: ANN202
        del kwargs
        enriched = dict(context)
        enriched.update(
            {
                "messages": [{"role": "system", "content": "tool aware"}],
                "tool_outputs": [
                    {
                        "tool": "book_pooja",
                        "event_name": "suggestion_booking",
                        "summary": "Home puja services: Satyanarayan Puja.",
                        "home_puja_services": [{"id": "svc-1", "name": "Satyanarayan Puja"}],
                        "temple_services": [],
                        "pandits": [],
                        "source": "core-service",
                    }
                ],
            }
        )
        return enriched

    service._background_memory_extraction = _background  # type: ignore[method-assign]
    service._prepare_base_reply_context = _prepare_base  # type: ignore[method-assign]
    service._complete_reply_context = _complete  # type: ignore[method-assign]

    async def _collect() -> list[tuple[str, dict[str, object]]]:
        events: list[tuple[str, dict[str, object]]] = []
        async for item in service.stream_reply_events(
            session_id="session-1",
            message="Book a Satyanarayan puja at home",
        ):
            events.append(item)
        return events

    events = asyncio.run(_collect())
    event_names = [name for name, _payload in events]

    assert event_names[0] == "status"
    assert event_names[1] == "meta"
    assert "suggestion_booking" in event_names
    assert event_names.index("suggestion_booking") < event_names.index("message")
    assert events[-1][0] == "done"
    assert events[-1][1]["metadata"]["cards"][0]["type"] == "service"
    message_text = "".join(payload["delta"] for name, payload in events if name == "message")
    assert events[-1][1]["reply"] == message_text


def test_stream_reply_events_persists_partial_reply_when_client_disconnects() -> None:
    plan = PlannerResult(
        action="respond_only",
        confidence=0.93,
        arguments={},
        missing_information=[],
        should_call_tool=False,
        reasoning="general reply",
    )

    service = ChatService.__new__(ChatService)

    class _GroqStub:
        is_configured = True
        last_usage = None

        async def stream_generate(self, *args, **kwargs):  # noqa: ANN202
            del args, kwargs
            yield "Saturn shows slow but steady progress. "
            yield "Stay consistent with one clear effort."

    class _EmotionStub:
        label = "calm"
        emotion = "calm"

    persisted: dict[str, object] = {}

    service.groq_client = _GroqStub()
    service._persist_chat_turns = lambda context, reply, response_metadata=None, partial=False: persisted.update(  # type: ignore[method-assign]
        {"reply": reply, "partial": partial, "metadata": response_metadata, "session_id": context["session_id"]}
    )
    service._persist_lightweight_memory = lambda *args, **kwargs: None  # type: ignore[method-assign]

    async def _background(*args, **kwargs):  # noqa: ANN202
        del args, kwargs

    async def _prepare_base(**kwargs):  # noqa: ANN202
        return {
            "plan": plan,
            "route": type("Route", (), {"provider": "groq", "model": "test-model", "reasoning_profile": "tool-aware"})(),
            "emotion": _EmotionStub(),
            "messages": [],
            "tool_outputs": [],
            "retrieval_matches": [],
            "kundali_chart": None,
            "kundali_summary": None,
            "matchmaking_result": None,
            "metadata_json": None,
            "message": kwargs["message"],
            "session_id": kwargs["session_id"],
            "scope_guardrail": {"allowed": True, "reason": "astrology_service_action"},
            "tool_guardrail": {"allowed": False, "reason": "planner_declined_tool"},
            "tool_execution_allowed": False,
            "birth_details_followup": False,
            "birth_details_capture_pending": False,
            "effective_birth_details": None,
            "matchmaking_details": None,
            "recent_messages": [],
            "internal_user_id": None,
            "route_decision": ChatRouteDecision(
                route="TOOL_FLOW",
                intent="respond_only",
                confidence=0.93,
                risk_level="low",
                reason="general_reply",
                should_call_tool=False,
            ),
            "normalized_message": kwargs["message"],
        }

    async def _complete(context, **kwargs):  # noqa: ANN202
        del kwargs
        enriched = dict(context)
        enriched.update(
            {
                "messages": [{"role": "system", "content": "tool aware"}],
                "tool_outputs": [],
            }
        )
        return enriched

    disconnect_calls = 0

    async def _disconnect_checker() -> bool:
        nonlocal disconnect_calls
        disconnect_calls += 1
        return disconnect_calls >= 3

    service._background_memory_extraction = _background  # type: ignore[method-assign]
    service._prepare_base_reply_context = _prepare_base  # type: ignore[method-assign]
    service._complete_reply_context = _complete  # type: ignore[method-assign]

    async def _collect() -> list[tuple[str, dict[str, object]]]:
        events: list[tuple[str, dict[str, object]]] = []
        async for item in service.stream_reply_events(
            session_id="session-1",
            message="Tell me about Saturn",
            disconnect_checker=_disconnect_checker,
        ):
            events.append(item)
        return events

    events = asyncio.run(_collect())

    assert all(name != "done" for name, _payload in events)
    assert persisted["partial"] is True
    assert "Saturn shows slow but steady progress" in str(persisted["reply"])
