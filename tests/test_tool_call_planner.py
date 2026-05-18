"""Tests for ToolCallPlanner — Groq function calling based intent classification."""
from __future__ import annotations

import asyncio
import json

from src.core.llm import GroqClient
from src.core.planner import ToolCallPlanner, _TOOL_CALL_CONFIDENCE, ConversationPlanner


class _StubGroqClient:
    """Minimal stub that mimics GroqClient for planner tests."""

    is_configured = True

    def __init__(self, tool_calls: list[dict] | None = None, content: str | None = None) -> None:
        self._message: dict = {}
        if tool_calls is not None:
            self._message["tool_calls"] = tool_calls
        if content is not None:
            self._message["content"] = content
        self._should_fail = False

    async def generate_with_tools(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        if self._should_fail:
            raise RuntimeError("Groq is down")
        return self._message


def _stub_groq(tool_calls: list[dict] | None = None, content: str | None = None) -> _StubGroqClient:
    return _StubGroqClient(tool_calls=tool_calls, content=content)


def _make_tool_call(name: str, arguments: dict) -> dict:
    return {
        "id": "call_test",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments),
        },
    }


def _run_plan(planner: ToolCallPlanner, message: str, **kwargs) -> object:  # type: ignore[no-untyped-def]
    defaults = {
        "has_birth_details": False,
        "has_matchmaking_details": False,
        "is_authenticated": False,
    }
    defaults.update(kwargs)
    return asyncio.run(planner.plan(message=message, **defaults))


def test_no_tool_call_returns_respond_only() -> None:
    planner = ToolCallPlanner(_stub_groq(content="Just a greeting, no action needed."))
    result = _run_plan(planner, "Hello, how are you?")
    assert result.action == "respond_only"
    assert result.should_call_tool is False
    assert result.confidence == _TOOL_CALL_CONFIDENCE["respond_only"]


def test_recommend_product_tool_call() -> None:
    planner = ToolCallPlanner(_stub_groq(tool_calls=[
        _make_tool_call("recommend_product", {
            "search_query": "rudraksha career",
            "reasoning": "User wants a product for career growth.",
        })
    ]))
    result = _run_plan(planner, "suggest me a rudraksha for career")
    assert result.action == "recommend_product"
    assert result.should_call_tool is True
    assert "rudraksha" in result.arguments.get("search_query", "")
    assert result.confidence == _TOOL_CALL_CONFIDENCE["recommend_product"]


def test_show_kundali_blocked_without_birth_details() -> None:
    planner = ToolCallPlanner(_stub_groq(tool_calls=[
        _make_tool_call("show_kundali", {"reasoning": "User wants to see kundali."})
    ]))
    result = _run_plan(planner, "Show my kundali", has_birth_details=False)
    assert result.action == "ask_clarification"
    assert result.should_call_tool is False
    assert "birth_details" in result.missing_information


def test_show_kundali_allowed_with_birth_details() -> None:
    planner = ToolCallPlanner(_stub_groq(tool_calls=[
        _make_tool_call("show_kundali", {"reasoning": "User wants kundali."})
    ]))
    result = _run_plan(planner, "Show my kundali", has_birth_details=True)
    assert result.action == "show_kundali"
    assert result.should_call_tool is True


def test_matchmaking_blocked_without_details() -> None:
    planner = ToolCallPlanner(_stub_groq(tool_calls=[
        _make_tool_call("matchmaking", {"reasoning": "Guna milan request."})
    ]))
    result = _run_plan(planner, "Check our compatibility", has_matchmaking_details=False)
    assert result.action == "ask_clarification"
    assert "matchmaking_details" in result.missing_information


def test_unknown_tool_name_falls_back() -> None:
    planner = ToolCallPlanner(_stub_groq(tool_calls=[
        _make_tool_call("create_booking", {"reasoning": "Malicious."})
    ]))
    result = _run_plan(planner, "create a booking for me")
    assert result.action == "respond_only"
    assert result.should_call_tool is False


def test_ask_clarification_tool_call() -> None:
    planner = ToolCallPlanner(_stub_groq(tool_calls=[
        _make_tool_call("ask_clarification", {
            "missing_information": ["birth_details"],
            "reasoning": "Need birth details to proceed.",
        })
    ]))
    result = _run_plan(planner, "Read my chart")
    assert result.action == "ask_clarification"
    assert result.should_call_tool is False
    assert "birth_details" in result.missing_information


def test_suggest_consultant_extracts_query() -> None:
    planner = ToolCallPlanner(_stub_groq(tool_calls=[
        _make_tool_call("suggest_consultant", {
            "search_query": "career astrologer",
            "reasoning": "User wants career guidance from an astrologer.",
        })
    ]))
    result = _run_plan(planner, "I want to talk to a career astrologer")
    assert result.action == "suggest_consultant"
    assert result.should_call_tool is True


def test_book_pooja_extracts_query() -> None:
    planner = ToolCallPlanner(_stub_groq(tool_calls=[
        _make_tool_call("book_pooja", {
            "search_query": "satyanarayan puja",
            "reasoning": "User wants to book a puja.",
        })
    ]))
    result = _run_plan(planner, "I want to book satyanarayan puja")
    assert result.action == "book_pooja"
    assert result.should_call_tool is True


def test_groq_failure_returns_fallback() -> None:
    client = _StubGroqClient()
    client._should_fail = True
    planner = ToolCallPlanner(client)  # type: ignore[arg-type]
    result = _run_plan(planner, "recommend rudraksha")
    assert result.action == "respond_only"
    assert result.should_call_tool is False
