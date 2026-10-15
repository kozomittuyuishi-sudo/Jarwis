import logging
from types import SimpleNamespace

import brains.brain as brain
from brains.cloud_brain.adaptive_brain import AdaptiveCloudBrain
from brains.cloud_brain.response_validator import (
    ResponseValidator,
    ValidationAxis,
)


def _response(content, finish_reason="stop"):
    return SimpleNamespace(
        id="safe-test-id",
        _response=SimpleNamespace(status_code=200),
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(
                    content=content,
                    refusal=None,
                    tool_calls=None,
                    reasoning_content="private provider reasoning",
                ),
            ),
        ],
        usage=SimpleNamespace(
            prompt_tokens=20,
            completion_tokens=3,
            total_tokens=23,
        ),
    )


def _client_for(responses, calls):
    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return next(responses)

    return SimpleNamespace(
        chat=SimpleNamespace(completions=Completions()),
    )


def _validated_response(content, finish_reason="stop"):
    return {
        "choices": [{
            "finish_reason": finish_reason,
            "message": {"content": content},
        }],
    }


def test_normal_provider_answer_is_extracted_and_validated():
    validator = ResponseValidator()
    response = _validated_response("A complete answer.")

    assert brain._extract_response_text("A complete answer.") == "A complete answer."
    assert validator.validate(response).passed


def test_empty_and_null_provider_content_fail_validation():
    validator = ResponseValidator()

    empty = validator.validate(_validated_response(""))
    null = validator.validate({
        "choices": [{
            "finish_reason": "stop",
            "message": {"content": None},
        }],
    })

    assert not empty.passed
    assert not null.passed
    assert ValidationAxis.MESSAGE_CONTENT in {
        axis.axis for axis in empty.failures
    }
    assert ValidationAxis.MESSAGE_CONTENT in {
        axis.axis for axis in null.failures
    }


def test_length_terminated_output_fails_even_when_partial_text_exists():
    result = ResponseValidator().validate(
        _validated_response("This sentence is unfinished", "length")
    )

    assert not result.passed
    finish_axis = result.axis_result(ValidationAxis.FINISH_REASON)
    assert finish_axis is not None and not finish_axis.passed


def test_text_blocks_extract_only_explicit_user_facing_text():
    content = [
        {"type": "reasoning", "text": "internal plan"},
        {"type": "text", "text": "Final "},
        {"type": "output_text", "text": "answer."},
        {"type": "tool_call", "text": "do not surface"},
    ]
    assert brain._extract_response_text(content) == "Final answer."
    assert brain._extract_response_text([
        {"type": "reasoning", "text": "internal plan"},
        {"type": "unknown", "text": "not a user answer"},
    ]) is None


def test_unexpected_and_malformed_content_blocks_are_rejected():
    validator = ResponseValidator()

    unexpected = validator.validate(
        _validated_response([{"type": "reasoning", "text": "private"}])
    )
    malformed = validator.validate(
        _validated_response([{"type": "text", "text": None}])
    )

    assert not unexpected.passed
    assert not malformed.passed
    assert not unexpected.axis_result(ValidationAxis.MESSAGE_CONTENT).passed
    assert not malformed.axis_result(ValidationAxis.MESSAGE_CONTENT).passed


def test_internal_planning_text_is_rejected():
    validator = ResponseValidator()
    for leaked_text in (
        "appropriately: acknowledge the previous answer, then respond",
        "Let me craft a response that addresses the request.",
        "<think>private reasoning</think>",
    ):
        result = validator.validate(_validated_response(leaked_text))
        assert not result.passed
        assert not result.axis_result(ValidationAxis.MESSAGE_CONTENT).passed


def test_truncated_answer_is_retried_and_never_returned_as_success(monkeypatch):
    calls = []
    responses = iter([
        _response("Partial answer that ends", "length"),
        _response("Complete answer."),
    ])
    adaptive = AdaptiveCloudBrain()
    monkeypatch.setattr(brain, "get_adaptive_brain", lambda: adaptive)
    monkeypatch.setattr(
        brain, "get_client", lambda: _client_for(responses, calls),
    )
    monkeypatch.setattr(
        brain,
        "get_persona_controller",
        lambda: SimpleNamespace(
            mode="JARWIS",
            instructions="",
            addressing_guidance="Use forms naturally.",
        ),
    )

    assert brain.ask_cloud_brain("Answer the test.") == "Complete answer."
    assert len(calls) == 2
    assert calls[1]["max_tokens"] > calls[0]["max_tokens"]


def test_internal_planning_leak_exhausts_bounded_retries_and_returns_empty(
    monkeypatch, caplog
):
    calls = []
    responses = iter([
        _response("appropriately: acknowledge the prompt, then answer"),
        _response("Let me craft a response that sounds useful."),
    ])
    monkeypatch.setattr(brain, "get_adaptive_brain", AdaptiveCloudBrain)
    monkeypatch.setattr(
        brain, "get_client", lambda: _client_for(responses, calls),
    )
    monkeypatch.setattr(
        brain,
        "get_persona_controller",
        lambda: SimpleNamespace(
            mode="ULTRON",
            instructions="",
            addressing_guidance="Use forms naturally.",
        ),
    )

    with caplog.at_level(logging.WARNING):
        result = brain.ask_cloud_brain("Answer the test.")

    assert result == ""
    assert len(calls) <= 3
    assert "INTERNAL_PLANNING" in caplog.text or "INVALID_RESPONSE" in caplog.text
    assert "appropriately: acknowledge" not in caplog.text
    assert "Let me craft a response" not in caplog.text


def test_retry_exhaustion_returns_no_truncated_success(monkeypatch):
    calls = []
    responses = iter([
        _response(None, "length"),
        _response(None, "length"),
        _response(None, "length"),
    ])
    monkeypatch.setattr(brain, "get_adaptive_brain", AdaptiveCloudBrain)
    monkeypatch.setattr(
        brain, "get_client", lambda: _client_for(responses, calls),
    )
    monkeypatch.setattr(
        brain,
        "get_persona_controller",
        lambda: SimpleNamespace(
            mode="JARWIS",
            instructions="",
            addressing_guidance="Use forms naturally.",
        ),
    )

    assert brain.ask_cloud_brain("Answer the test.") == ""
    assert len(calls) == 3


def test_truncated_response_at_output_ceiling_is_not_retried(monkeypatch):
    calls = []
    adaptive = AdaptiveCloudBrain()
    monkeypatch.setattr(brain, "get_adaptive_brain", lambda: adaptive)
    monkeypatch.setattr(
        brain,
        "get_client",
        lambda: _client_for(iter([_response("Partial answer", "length")]), calls),
    )
    monkeypatch.setattr(
        brain,
        "get_persona_controller",
        lambda: SimpleNamespace(
            mode="JARWIS",
            instructions="",
            addressing_guidance="Use forms naturally.",
        ),
    )

    result = brain.ask_cloud_brain("Write a detailed report on the test.")

    assert result == ""
    assert len(calls) == 1
    assert calls[0]["max_tokens"] == 2048


def test_response_metadata_reports_safe_structure_not_reasoning_text():
    metadata = brain._response_metadata(
        _response(None, "length"),
        "openrouter",
        "test-model",
        1,
        256,
    )

    assert metadata["finish_reason"] == "length"
    assert metadata["content_is_null"] is True
    assert "reasoning_content" in metadata["message_fields"]
    assert metadata["reasoning_field_chars"]["reasoning_content"] > 0
    assert "private provider reasoning" not in repr(metadata)


def test_openai_wrapper_preserves_http_status_with_parsed_response():
    parsed = _response("Answer.")

    class RawResponse:
        status_code = 206

        def parse(self):
            return parsed

    class CompletionEndpoints:
        with_raw_response = SimpleNamespace(
            create=lambda **_kwargs: RawResponse(),
        )

        @staticmethod
        def create(**_kwargs):
            raise AssertionError("Raw response path should be used when available.")

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=CompletionEndpoints()),
    )
    wrapped = brain.ClientWrapper(client).chat.completions.create(model="test")

    assert wrapped.status_code == 206
    assert wrapped.choices[0].message.content == "Answer."
