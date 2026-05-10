from src.core.chat_service import ChatService
from src.core.planner import PlannerResult


def test_chat_service_is_defined() -> None:
    assert ChatService


def test_infer_response_language_prefers_english_for_plain_english_message() -> None:
    message = "I want one practical next step for my career this week."
    assert ChatService._infer_response_language(message) == "english"


def test_infer_response_language_prefers_hinglish_for_hinglish_message() -> None:
    message = "Mujhe career ke liye ek practical step batayein, main bahut confused hoon."
    assert ChatService._infer_response_language(message) == "hinglish"


def test_response_style_context_blocks_unsolicited_products_in_general_chat() -> None:
    plan = PlannerResult(
        action="respond_only",
        confidence=0.91,
        arguments={},
        missing_information=[],
        should_call_tool=False,
        reasoning="general guidance",
    )
    style = ChatService._build_response_style_context(
        message="My career feels stuck and I need clarity.",
        plan=plan,
        tool_outputs=[],
    )
    assert "plain English only" in style
    assert "Do not introduce remedies" in style
    assert "2 to 4 short sentences" in style
    assert "feel human and engaging" in style
    assert "not a mini article" in style


def test_response_style_context_honors_product_decline_and_single_step_request() -> None:
    plan = PlannerResult(
        action="ask_clarification",
        confidence=0.88,
        arguments={},
        missing_information=[],
        should_call_tool=False,
        reasoning="needs one next step",
    )
    style = ChatService._build_response_style_context(
        message="I do not want product suggestions right now. Give me one practical next step for this week only.",
        plan=plan,
        tool_outputs=[],
    )
    assert "the user does not want product suggestions right now" in style.lower()
    assert "exactly one practical next step" in style


def test_response_style_context_limits_clarification_replies() -> None:
    plan = PlannerResult(
        action="ask_clarification",
        confidence=0.9,
        arguments={},
        missing_information=["goal"],
        should_call_tool=False,
        reasoning="need one clear question",
    )
    style = ChatService._build_response_style_context(
        message="Things feel stuck and I do not know what to ask.",
        plan=plan,
        tool_outputs=[],
    )
    assert "ask exactly one clear question" in style
    assert "acknowledge the concern in a human way" in style
    assert "2 or 3 short sentences total" in style
    assert "Avoid filler phrases" in style


def test_response_style_context_for_consultant_reply_stays_advisory() -> None:
    plan = PlannerResult(
        action="suggest_consultant",
        confidence=0.94,
        arguments={"search_query": "career guidance"},
        missing_information=[],
        should_call_tool=True,
        reasoning="user wants expert guidance",
    )
    style = ChatService._build_response_style_context(
        message="Maybe I should speak to someone directly about my career confusion.",
        plan=plan,
        tool_outputs=[],
    )
    assert "Sound advisory and personal, not transactional" in style


def test_postprocess_reply_compacts_verbose_career_clarification() -> None:
    plan = PlannerResult(
        action="ask_clarification",
        confidence=0.92,
        arguments={},
        missing_information=[],
        should_call_tool=False,
        reasoning="needs one focused question",
    )
    reply = (
        "It can be really challenging when you're feeling unclear about your career path. "
        "Let's break it down a bit. When you think about your career, what's the main thing "
        "that's bothering you - is it finding the right direction, building confidence in "
        "your current role, or maybe timing your next steps?"
    )
    compacted = ChatService._postprocess_reply(
        reply=reply,
        plan=plan,
        message="I am confused about my career and I do not know what to focus on",
    )
    assert compacted == (
        "I understand. Career confusion usually comes from direction, confidence, or timing. "
        "Which of these feels strongest right now?"
    )


def test_postprocess_reply_compacts_alternate_career_clarification_wording() -> None:
    plan = PlannerResult(
        action="ask_clarification",
        confidence=0.92,
        arguments={},
        missing_information=[],
        should_call_tool=False,
        reasoning="needs one focused question",
    )
    reply = (
        "Career confusion can be really frustrating. It's like standing at a crossroads, unsure "
        "which path to take. To get a better sense of what might be going on, can you tell me "
        "what's been troubling you the most about your career - is it finding the right direction, "
        "lacking confidence, or feeling stuck in your current situation?"
    )
    compacted = ChatService._postprocess_reply(
        reply=reply,
        plan=plan,
        message="I am confused about my career and I do not know what to focus on",
    )
    assert compacted == (
        "I understand. Career confusion usually comes from direction, confidence, or timing. "
        "Which of these feels strongest right now?"
    )


def test_postprocess_reply_compacts_love_life_clarification() -> None:
    plan = PlannerResult(
        action="ask_clarification",
        confidence=0.9,
        arguments={},
        missing_information=[],
        should_call_tool=False,
        reasoning="needs focused relationship question",
    )
    reply = (
        "It can be really tough to navigate uncertainty in our personal relationships. "
        "When it comes to love life, astrology often looks at the 5th and 7th houses in a person's birth chart. "
        "But to get a better understanding of what's going on, can you tell me a bit more about what's been troubling you "
        "- is it a current relationship, finding a new partner, or something else?"
    )
    compacted = ChatService._postprocess_reply(
        reply=reply,
        plan=plan,
        message="I feel confused about my love life lately.",
    )
    assert compacted == (
        "I understand. In love matters, the real issue is usually clarity, trust, or timing. "
        "Which of these feels most unsettled right now?"
    )


def test_postprocess_reply_compacts_relationship_distance_clarification() -> None:
    plan = PlannerResult(
        action="ask_clarification",
        confidence=0.9,
        arguments={},
        missing_information=[],
        should_call_tool=False,
        reasoning="needs focused relationship question",
    )
    reply = (
        "It sounds like there's a sense of disconnection and miscommunication in your relationship. "
        "In astrology, Venus and the Moon are two planets that can influence our emotional connections and relationships. "
        "To better understand what might be going on, do you feel like the distance and misunderstandings are coming from you, "
        "your partner, or a bit of both?"
    )
    compacted = ChatService._postprocess_reply(
        reply=reply,
        plan=plan,
        message="There are repeated misunderstandings and distance in my relationship.",
    )
    assert compacted == (
        "That sounds more like emotional distance than one big fight. "
        "Do you feel this is coming more from you, your partner, or both?"
    )


def test_postprocess_reply_compacts_generic_consultant_handoff() -> None:
    plan = PlannerResult(
        action="suggest_consultant",
        confidence=0.92,
        arguments={"search_query": "relationship astrologer"},
        missing_information=[],
        should_call_tool=True,
        reasoning="user wants expert help",
    )
    reply = (
        "I can suggest a consultation with a pandit who specializes in relationship guidance and astrology. "
        "They can help you understand the cosmic influences at play in your relationship and provide guidance on how to navigate "
        "the challenges you're facing. Would you like me to arrange a consultation with a pandit who can offer you personalized advice and support?"
    )
    compacted = ChatService._postprocess_reply(
        reply=reply,
        plan=plan,
        message="Yes, recommend someone for relationship guidance.",
    )
    assert compacted == (
        "Yes, speaking to a relationship astrologer would help here. "
        "I can show you available pandits for relationship guidance."
    )
