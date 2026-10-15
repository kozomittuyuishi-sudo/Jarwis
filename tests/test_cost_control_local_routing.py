import logging
from types import SimpleNamespace

import brains.brain as brain
import Jarwis as Ag
from directory_control import DirectoryControl
from runtime.persona_controller import PersonaController
from self_info.task_assessor import VERDICT_FEASIBLE, assess_task
from self_info import SelfInfoEngine


def test_deterministic_cli_turns_do_not_call_a_brain(tmp_path, monkeypatch, capsys):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "local.txt").write_text("local", encoding="utf-8")
    control = DirectoryControl()
    control.set_directory(str(workspace))
    controller = PersonaController(tmp_path / "persona.json")
    cloud_calls = []
    user_inputs = iter([
        "hey",
        "oh hello jarwis",
        "hey J",
        "J?",
        "Jarwis?",
        "Who are you?",
        "Switch to Ultron mode",
        "What mode are you in?",
        "What can you do?",
        "what files are here?",
    ])

    def read_input(_prompt):
        try:
            return next(user_inputs)
        except StopIteration:
            raise EOFError

    monkeypatch.chdir(tmp_path)
    (tmp_path / "configuration").mkdir()
    monkeypatch.setattr(Ag, "input", read_input, raising=False)
    monkeypatch.setattr(Ag, "startup", lambda: None)
    monkeypatch.setattr(Ag, "get_persona_controller", lambda: controller)
    monkeypatch.setattr(Ag, "DirectoryControl", lambda: control)
    monkeypatch.setattr(Ag, "load_brain_mode", lambda: "cloud")
    monkeypatch.setattr(brain, "load_brain_mode", lambda: "cloud")
    monkeypatch.setattr(
        brain,
        "ask_cloud_brain",
        lambda *args, **kwargs: cloud_calls.append(args) or "unexpected cloud answer",
    )

    Ag.main()
    output = capsys.readouterr().out

    assert cloud_calls == []
    assert "Jarwis mode is active" in output
    assert "I am Jarwis" in output
    assert "Ultron mode enabled" in output
    assert "I am in Ultron mode" in output
    assert "I can currently" in output
    assert "local.txt" in output
    assert controller.mode == "ULTRON"


def test_greeting_and_direct_name_variations_are_local_intents():
    for phrase in (
        "hey",
        "oh hello jarwis",
        "hey J",
        "J?",
        "Jarwis?",
        "hey Ultron",
    ):
        assert Ag.detect_intent(phrase) == "greeting"


def test_implementation_assessment_uses_existing_workspace_capability():
    query = "could you implement that within you?"
    assessment = assess_task(query)
    response = SelfInfoEngine().answer(query).context_block

    assert assessment.verdict == VERDICT_FEASIBLE
    assert "Workspace and filesystem control" in assessment.matching_capabilities
    assert any("not unrestricted self-modification" in note for note in assessment.notes)
    assert "FEASIBLE" in response
    assert "Workspace and filesystem control" in response


def test_complex_brainstorming_uses_reasoning_budget():
    from brains.cloud_brain.token_estimator import RequestCategory, TokenEstimator

    task = (
        "Brainstorm practical approaches, compare their tradeoffs, and "
        "recommend one."
    )
    estimate = TokenEstimator().estimate(task, task_text=task)

    assert estimate.category == RequestCategory.REASONING
    assert estimate.max_tokens == 1024


def test_http_402_stops_after_one_openrouter_attempt(monkeypatch, caplog):
    calls = []
    secret = "sk-or-v1-sensitive-test-secret"

    class CreditError(Exception):
        status_code = 402

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            raise CreditError(f"HTTP 402 insufficient credits api_key={secret}")

    monkeypatch.setattr(brain, "get_adaptive_brain", brain.AdaptiveCloudBrain)
    monkeypatch.setattr(
        brain,
        "get_persona_controller",
        lambda: SimpleNamespace(
            mode="JARWIS",
            instructions="",
            addressing_guidance="Use forms naturally.",
        ),
    )
    monkeypatch.setattr(
        brain,
        "get_client",
        lambda: SimpleNamespace(
            chat=SimpleNamespace(completions=Completions()),
        ),
    )

    with caplog.at_level(logging.ERROR, logger="ag.brain.diagnostics"):
        result = brain.ask_cloud_brain("A short test request.")

    assert len(calls) == 1
    assert calls[0]["model"] == "deepseek/deepseek-v4-flash-0731"
    assert 0 < calls[0]["max_tokens"] <= 2048
    assert "HTTP 402" in result
    assert "No retry or provider switch" in result
    assert "code=CREDIT_LIMIT" in caplog.text
    assert "status=402" in caplog.text
    assert "max_tokens=" in caplog.text
    assert secret not in caplog.text
