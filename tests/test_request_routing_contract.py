"""Integration contracts for QUIPS ownership and isolated unit routing."""

from types import SimpleNamespace
from unittest.mock import patch

import Jarwis as Ag
from preprocessor_brain import PreprocessorBrain
from quips import analyze_request


def _run(query, preprocessor):
    analysis = analyze_request(query, Ag.detect_intent)
    return Ag.process_input(
        query, {}, {"active": [], "completed": []},
        analysis.requests[0].legacy_intent or "unknown",
        preprocessor_brain=preprocessor,
        quips_result=analysis,
        request_unit=analysis.requests[0],
    )


def test_compound_self_info_is_lossless_and_never_calls_brain():
    query = "Tell me who you are, what you're capable of, and how you work internally."
    with patch.object(Ag, "ask_brain", side_effect=AssertionError("LLM must not be called")):
        response = _run(query, PreprocessorBrain())

    assert "I am Jarwis" in response
    assert "I can currently" in response
    assert "deterministic offline processing layer" in response


def test_one_failed_unit_does_not_erase_other_unit_responses():
    class IsolatingPreprocessor:
        def process_self_info_intent(self, subtype, original_text):
            if subtype == "ARCHITECTURE":
                raise RuntimeError("intentional test failure")
            return SimpleNamespace(success=True, result=f"{subtype} result")

    response = _run(
        "Tell me about yourself, what can you do, and how do you work internally.",
        IsolatingPreprocessor(),
    )

    assert "IDENTITY result" in response
    assert "CAPABILITIES result" in response
    assert "couldn't complete this part" in response


def test_conversation_manager_keeps_a_compound_turn_as_one_turn_pair():
    manager = Ag.ConversationManager()
    query = "What can you do and what can't you do?"
    manager.add_turn("user", query, {"request_count": 2})
    manager.add_turn("assistant", "capabilities\n\nlimitations")

    state = manager.get_state()
    assert state["turn_count"] == 2
    assert state["recent_turns"][0]["content"] == query
    assert state["recent_turns"][0]["metadata"]["request_count"] == 2


def test_conversation_reference_remains_one_contextual_follow_up():
    manager = Ag.ConversationManager()
    manager.set_topic("local brain")
    manager.add_turn("user", "What brains do you have?")
    manager.add_turn("assistant", "Local brain and cloud brain are available.")

    result = manager.resolve_reference("the local one")
    assert result["resolved"] is True
    assert result["reference_type"] == "recent_turn_fallback"
