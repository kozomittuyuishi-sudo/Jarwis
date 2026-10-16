from unittest.mock import patch
import builtins

import Jarwis as Ag
import pytest
from cognition.executive_layer import ExecutiveLayer
from memory.working_memory import WorkingMemory
from preprocessor_brain import PreprocessorBrain
from quips import analyze_request


@pytest.fixture(autouse=True)
def keep_integration_tests_offline_and_side_effect_free(monkeypatch):
    monkeypatch.setattr(Ag, "log_event", lambda *_args, **_kwargs: None)


def test_empty_quips_result_does_not_index_the_missing_first_request():
    analysis = analyze_request("")
    assert Ag._first_quips_request(analysis) is None
    assert Ag._first_quips_request(type("Malformed", (), {"requests": None})()) is None


def test_cli_empty_request_returns_diagnostic_without_index_error(tmp_path, monkeypatch, capsys):
    control = Ag.DirectoryControl()
    control.set_directory(str(tmp_path))
    inputs = iter([""])

    def read_input(_prompt=""):
        try:
            return next(inputs)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr(Ag, "DirectoryControl", lambda: control)
    monkeypatch.setattr(Ag, "load_memory", lambda: {})
    monkeypatch.setattr(Ag, "load_tasks", lambda: {"active": [], "completed": []})
    monkeypatch.setattr(Ag, "load_project_context", lambda: {})
    monkeypatch.setattr(Ag, "load_brain_mode", lambda: "local")
    monkeypatch.setattr(Ag, "startup", lambda: None)
    monkeypatch.setattr(builtins, "input", read_input)
    monkeypatch.setattr(
        Ag, "_pipeline_response",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("provider route")),
    )
    monkeypatch.setattr(
        Ag, "ask_brain",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("provider call")),
    )

    Ag.main()

    assert "couldn't safely process this request (parsing)" in capsys.readouterr().out


def test_compound_filesystem_plan_executes_with_verified_results(tmp_path):
    query = (
        "J, create a folder named Jarwis_ANVIL_Test inside the active workspace. "
        "Inside it, create a file named report.txt, write "
        "'ANVIL pipeline verification' into the file, read it back, and tell me "
        "the exact path and content. Track each step and report any failure honestly."
    )
    control = Ag.DirectoryControl()
    control.set_directory(str(tmp_path))
    memory = WorkingMemory()
    preprocessor = PreprocessorBrain(directory_control=control)
    analysis = analyze_request(query, Ag.detect_intent)
    executive = ExecutiveLayer()
    plan = executive.analyze_action_plan(analysis)

    assert plan.valid
    assert [unit.intent for unit in analysis.requests] == [
        "CREATE_FOLDER", "CREATE_FILE", "WRITE_FILE", "READ_FILE",
        "EXPLAIN_RESULT",
    ]
    assert [step.dependencies for step in plan.steps] == [
        [], ["step-1"], ["step-2"], ["step-3"], ["step-4"],
    ]

    with patch.object(
        Ag, "_pipeline_response",
        side_effect=AssertionError("deterministic workflow must not invoke a provider"),
    ) as provider, patch.object(
        Ag, "ask_brain",
        side_effect=AssertionError("deterministic workflow must not invoke a provider"),
    ) as legacy_provider:
        response = Ag.process_input(
            query,
            {},
            {"active": [], "completed": []},
            analysis.requests[0].legacy_intent or Ag.detect_intent(query),
            directory_control=control,
            preprocessor_brain=preprocessor,
            quips_result=analysis,
            request_unit=analysis.requests[0],
            working_memory=memory,
            action_plan=plan,
        )

    folder = tmp_path / "Jarwis_ANVIL_Test"
    file_path = folder / "report.txt"
    assert folder.is_dir()
    assert file_path.is_file()
    assert file_path.read_text(encoding="utf-8") == "ANVIL pipeline verification"
    assert str(file_path) in response
    assert "ANVIL pipeline verification" in response
    assert provider.call_count == 0
    assert legacy_provider.call_count == 0

    events = memory.get_action_events()
    relevant = [
        event for event in events
        if event.get("original_request") == query
    ]
    assert [event["action"] for event in relevant] == ["CREATE", "CREATE", "WRITE", "READ"]
    assert all(event["status"] == "SUCCESS" for event in relevant)
    assert all(event["verification_status"] == "VERIFIED" for event in relevant)
    assert relevant[1]["resolved_path"] == str(file_path)
    assert relevant[2]["actual_result"]["contents"] == "ANVIL pipeline verification"
    assert relevant[3]["actual_result"]["contents"] == "ANVIL pipeline verification"
    assert relevant[1]["depends_on_action_ids"] == [relevant[0]["action_id"]]
    assert relevant[2]["depends_on_action_ids"] == [relevant[1]["action_id"]]
    assert relevant[3]["depends_on_action_ids"] == [relevant[2]["action_id"]]
    assert all(step.execution_status == "COMPLETED" for step in plan.steps)
    assert str(file_path) in plan.steps[-1].actual_result
    assert "ANVIL pipeline verification" in plan.steps[-1].actual_result
    assert memory.get_fact("anvil_last_plan")["status"] == "COMPLETED"
    follow_up = memory.resolve_context_reference(
        "the file I just created",
        expected_type="FILE",
    )
    assert follow_up["resolved_path"] == str(file_path)


def test_failed_prerequisite_blocks_dependent_but_independent_step_runs(tmp_path):
    query = (
        "Create folder Broken, create dependent.txt inside it, "
        "and create independent.txt."
    )
    control = Ag.DirectoryControl()
    control.set_directory(str(tmp_path))
    memory = WorkingMemory()
    analysis = analyze_request(query, Ag.detect_intent)
    plan = ExecutiveLayer().analyze_action_plan(analysis)
    original_create_directory = control.create_directory

    def fail_folder_creation(path):
        raise Ag.DirectoryControlError("simulated folder creation failure")

    control.create_directory = fail_folder_creation
    try:
        response = Ag.process_input(
            query,
            {},
            {"active": [], "completed": []},
            analysis.requests[0].legacy_intent or "unknown",
            directory_control=control,
            preprocessor_brain=PreprocessorBrain(directory_control=control),
            quips_result=analysis,
            request_unit=analysis.requests[0],
            working_memory=memory,
            action_plan=plan,
        )
    finally:
        control.create_directory = original_create_directory

    assert not (tmp_path / "Broken").exists()
    assert (tmp_path / "independent.txt").is_file()
    assert plan.steps[0].execution_status == "FAILED"
    assert plan.steps[1].execution_status == "BLOCKED"
    assert plan.steps[2].execution_status == "COMPLETED"
    assert "did not succeed" in response


def test_failed_verification_cannot_be_reported_as_success():
    query = "create report.txt and tell me the exact path and content"
    analysis = analyze_request(query, Ag.detect_intent)
    final_unit = analysis.requests[-1]
    memory = WorkingMemory()
    event = memory.begin_action_event(
        action="READ",
        target="report.txt",
        target_type="FILE",
        desired_outcome="FILE_CONTENT",
        original_request=query,
        request_unit=analysis.requests[0].to_dict(),
    )
    memory.update_action_event(
        event["action_id"],
        resolved_path="D:\\workspace\\report.txt",
        status="FAILED",
        verification_status="FAILED",
        actual_result={"contents": "unverified"},
    )

    response = Ag.process_input(
        query,
        {},
        {"active": [], "completed": []},
        final_unit.legacy_intent or "unknown",
        request_unit=final_unit,
        working_memory=memory,
    )

    assert "can't report success" in response
    assert "Exact path:" not in response
