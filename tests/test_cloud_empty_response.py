import logging
from dataclasses import dataclass
from types import SimpleNamespace

import brains.brain as brain
from brains.brain import BrainStatus


def _response(content, *, status=200, finish_reason="stop"):
    return SimpleNamespace(
        _response=SimpleNamespace(status_code=status),
        id="response-safe-id",
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content, refusal=None, tool_calls=[]),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=9, completion_tokens=0, total_tokens=9),
    )


def _adaptive():
    @dataclass
    class Prep:
        ready: bool
        max_tokens: int
        estimation: object
        validation_report: object

    class Adaptive:
        def prepare(self, **_kwargs):
            return Prep(
                ready=True,
                max_tokens=64,
                estimation=SimpleNamespace(provider_ceiling=512),
                validation_report=SimpleNamespace(issues=[]),
            )

        def validate_response(self, *, raw_response, **_kwargs):
            choices = raw_response.get("choices", [])
            content = (
                choices[0].get("message", {}).get("content")
                if choices else None
            )
            if isinstance(content, str) and content.strip():
                return SimpleNamespace(
                    status=BrainStatus.PASSED,
                    retry_outcome=None,
                )
            return SimpleNamespace(
                status=BrainStatus.RETRYING,
                retry_outcome=SimpleNamespace(
                    should_retry=True,
                    adjusted_max_tokens=32,
                ),
            )

    return Adaptive()


def test_empty_cloud_response_gets_one_bounded_same_budget_retry(
    monkeypatch, caplog
):
    responses = iter([_response(None), _response("Recovered answer")])
    calls = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return next(responses)

    monkeypatch.setattr(brain, "get_adaptive_brain", _adaptive)
    monkeypatch.setattr(
        brain,
        "get_client",
        lambda: SimpleNamespace(
            chat=SimpleNamespace(completions=Completions())
        ),
    )
    monkeypatch.setattr(brain, "get_persona_controller", lambda: SimpleNamespace(
        mode="JARWIS",
        instructions="",
        addressing_guidance="Use forms of address naturally and sparingly.",
    ))

    with caplog.at_level(logging.WARNING, logger="ag.brain.diagnostics"):
        result = brain.ask_cloud_brain("A test request")

    assert result == "Recovered answer"
    assert len(calls) == 2
    assert [call["max_tokens"] for call in calls] == [64, 64]
    assert calls[0]["model"] == "deepseek/deepseek-v4-flash-0731"
    assert [message["role"] for message in calls[0]["messages"]] == [
        "system", "user"
    ]
    assert "code=EMPTY_RESPONSE" in caplog.text
    assert "status': 200" in caplog.text
    assert "response-safe-id" in caplog.text


def test_repeated_empty_cloud_response_returns_no_success_text(
    monkeypatch, caplog
):
    calls = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return _response("")

    monkeypatch.setattr(brain, "get_adaptive_brain", _adaptive)
    monkeypatch.setattr(
        brain,
        "get_client",
        lambda: SimpleNamespace(
            chat=SimpleNamespace(completions=Completions())
        ),
    )
    monkeypatch.setattr(brain, "get_persona_controller", lambda: SimpleNamespace(
        mode="ULTRON",
        instructions="",
        addressing_guidance="Use forms of address naturally and sparingly.",
    ))

    result = brain.ask_cloud_brain("A test request")

    assert result == ""
    assert len(calls) == 2
    assert "Active response persona: ULTRON" in calls[0]["messages"][0]["content"]
    assert "EMPTY_RESPONSE" in caplog.text


def test_truncated_empty_response_retries_with_bounded_larger_budget(monkeypatch):
    responses = iter([
        _response(None, finish_reason="length"),
        _response("Recovered after truncation"),
    ])
    calls = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return next(responses)

    monkeypatch.setattr(brain, "get_adaptive_brain", _adaptive)
    monkeypatch.setattr(
        brain,
        "get_client",
        lambda: SimpleNamespace(
            chat=SimpleNamespace(completions=Completions())
        ),
    )
    monkeypatch.setattr(brain, "get_persona_controller", lambda: SimpleNamespace(
        mode="JARWIS",
        instructions="",
        addressing_guidance="Use forms of address naturally and sparingly.",
    ))

    assert brain.ask_cloud_brain("A test request") == "Recovered after truncation"
    assert [call["max_tokens"] for call in calls] == [64, 128]


def test_cloud_empty_output_uses_failure_fallback_and_local_self_info_stays_local(
    monkeypatch, tmp_path
):
    from runtime.persona_controller import PersonaController
    import Jarwis as Ag

    monkeypatch.setattr(brain, "load_brain_mode", lambda: "cloud")
    monkeypatch.setattr(brain, "ask_cloud_brain", lambda *_args, **_kwargs: "")

    fallback = brain.ask_brain("A normal request")
    assert fallback.startswith(brain._FALLBACK_PREFIX)

    controller = PersonaController(tmp_path / "persona.json")
    monkeypatch.setattr(Ag, "get_persona_controller", lambda: controller)
    monkeypatch.setattr(
        Ag,
        "ask_brain",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("local self-information must not use cloud generation")
        ),
    )
    identity = Ag.process_input(
        "Who are you?",
        {},
        {"active": [], "completed": []},
        Ag.detect_intent("Who are you?"),
    )
    assert "Jarwis" in identity


def test_response_text_extracts_text_blocks_without_exposing_other_metadata():
    assert brain._extract_response_text([
        {"type": "text", "text": "First "},
        SimpleNamespace(type="text", text="second"),
        {"type": "image", "url": "not text"},
    ]) == "First second"
    assert brain._extract_response_text(None) is None


def test_provider_error_diagnostics_redact_credentials():
    secret = "sk-or-v1-sensitive-example-secret"
    detail = brain._safe_error_detail(
        Exception(f"Authorization: Bearer {secret}; api_key={secret}")
    )

    assert secret not in detail
    assert "[REDACTED]" in detail
