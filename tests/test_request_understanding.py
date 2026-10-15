from unittest.mock import patch

import Jarwis as Ag
from conversation.cognitive.evaluator import CognitiveEvaluator
from conversation.core.thought import Thought
from directory_control import DirectoryControl
from directory_control.directory_control import VerificationError
from memory.working_memory import WorkingMemory
from quips import DesiredOutcome, RequestCategory, analyze_request
from target_resolution_models import TargetType


def _unit(text):
    return analyze_request(text, Ag.detect_intent).requests[0]


def _process(text, control, memory):
    result = analyze_request(text, Ag.detect_intent)
    unit = result.requests[0]
    return Ag.process_input(
        text,
        {},
        {"active": [], "completed": []},
        unit.legacy_intent or Ag.detect_intent(text),
        directory_control=control,
        quips_result=result,
        request_unit=unit,
        working_memory=memory,
    )


def test_filesystem_requests_have_distinct_desired_outcomes():
    cases = {
        "open brains": (DesiredOutcome.OPEN_DIRECTORY, "brains"),
        "read Brain.py": (DesiredOutcome.FILE_CONTENT, "Brain.py"),
        "show me what's inside brains": (DesiredOutcome.DIRECTORY_CONTENTS, "brains"),
        "tell me about Brain.py": (DesiredOutcome.EXPLAIN_FILE, "Brain.py"),
        "create test.txt": (DesiredOutcome.CREATE_FILE, "test.txt"),
    }
    for text, (outcome, target) in cases.items():
        understanding = _unit(text).understanding
        assert understanding.domain == "FILESYSTEM"
        assert understanding.desired_outcome == outcome
        assert understanding.target == target


def test_contextual_file_requests_resolve_prior_successful_action(tmp_path):
    control = DirectoryControl()
    control.set_directory(str(tmp_path))
    memory = WorkingMemory()

    assert "created: test.txt" in _process("create test.txt", control, memory)
    created = memory.get_action_events()[-1]
    assert created["action"] == "CREATE"
    assert created["target_type"] == "FILE"
    assert created["resolved_path"] == str(tmp_path / "test.txt")
    (tmp_path / "test.txt").write_text("remembered content", encoding="utf-8")
    assert "remembered content" in _process("read it", control, memory)
    assert "remembered content" in _process("open it", control, memory)


def test_directory_context_and_type_specific_references(tmp_path):
    control = DirectoryControl()
    control.set_directory(str(tmp_path))
    memory = WorkingMemory()
    (tmp_path / "brains").mkdir()
    (tmp_path / "brains" / "Brain.py").write_text("brain source", encoding="utf-8")

    assert "Brain.py" in _process("open brains", control, memory)
    assert "Brain.py" in _process("what's inside it?", control, memory)

    assert "created: projects" in _process("create a folder called projects", control, memory)
    assert "projects" in _process("open it", control, memory)


def test_created_earlier_reference_resolves_through_action_memory(tmp_path):
    control = DirectoryControl()
    control.set_directory(str(tmp_path))
    memory = WorkingMemory()
    assert "created: test.txt" in _process("create test.txt", control, memory)
    (tmp_path / "test.txt").write_text("earlier file contents", encoding="utf-8")

    assert "earlier file contents" in _process(
        "read the file I created earlier", control, memory,
    )


def test_missing_target_is_not_sent_to_filesystem_executor(tmp_path):
    control = DirectoryControl()
    control.set_directory(str(tmp_path))
    response = _process("create a folder", control, WorkingMemory())

    assert "need a target name" in response
    assert list(tmp_path.iterdir()) == []


def test_unresolved_context_reference_requests_clarification(tmp_path):
    control = DirectoryControl()
    control.set_directory(str(tmp_path))

    response = _process("read it", control, WorkingMemory())

    assert "need a target name or a valid previous filesystem item" in response
    assert list(tmp_path.iterdir()) == []


def test_failed_postcondition_is_reported_and_not_remembered(tmp_path, monkeypatch):
    control = DirectoryControl()
    control.set_directory(str(tmp_path))
    memory = WorkingMemory()

    def fail_verification(_path, *, must_exist=True):
        raise VerificationError("simulated postcondition failure")

    monkeypatch.setattr(control, "verify_operation", fail_verification)
    response = _process("create test.txt", control, memory)

    assert "simulated postcondition failure" in response
    event = memory.get_action_events()[-1]
    assert event["status"] == "FAILED"
    assert event["failure_stage"] == "verification"
    assert event["verification_status"] == "NOT_VERIFIED"


def test_rename_resolves_both_targets_verifies_state_and_updates_memory(tmp_path):
    control = DirectoryControl()
    control.set_directory(str(tmp_path))
    source = tmp_path / "Brain.py"
    source.write_text("brain", encoding="utf-8")
    memory = WorkingMemory()

    response = _process("rename Brain.py to Brain_old.py", control, memory)

    destination = tmp_path / "Brain_old.py"
    assert "Renamed to: Brain_old.py" in response
    assert destination.read_text(encoding="utf-8") == "brain"
    assert not source.exists()
    event = memory.get_action_events()[-1]
    assert event["action"] == "RENAME"
    assert event["resolved_path"] == str(destination)


def test_compound_create_then_explain_keeps_unit_order_and_result_context(tmp_path):
    control = DirectoryControl()
    control.set_directory(str(tmp_path))
    memory = WorkingMemory()
    with patch.object(Ag, "_pipeline_response", return_value="AG: unexpected LLM response") as brain:
        analysis = analyze_request("create test.txt and explain what you created", Ag.detect_intent)
        result = Ag.process_input(
            analysis.original_text, {}, {"active": [], "completed": []}, "unknown",
            directory_control=control, quips_result=analysis,
            request_unit=analysis.requests[0], working_memory=memory,
        )

    assert analysis.is_compound
    assert "created the file" in result
    assert "empty" in result
    brain.assert_not_called()
    record = memory.get_action_events()[-1]
    assert record["original_request"] == "create test.txt and explain what you created"
    assert record["request_unit_index"] == 0
    assert record["is_compound"] is True
    assert record["verification_status"] == "VERIFIED"
    assert record["actual_result"]["size_bytes"] == 0


def test_create_open_write_read_and_change_summary_use_verified_action_results(tmp_path):
    control = DirectoryControl()
    control.set_directory(str(tmp_path))
    memory = WorkingMemory()

    assert "File created: experiment.txt" in _process("Create a file called experiment.txt.", control, memory)
    assert "Contents of experiment.txt" in _process("Open it.", control, memory)
    assert "verified it by reading it back" in _process("Write 'Hello AG' into it.", control, memory)
    assert "Hello AG" in _process("Read it.", control, memory)
    summary = _process("What did you change?", control, memory)

    file_path = tmp_path / "experiment.txt"
    assert file_path.read_text(encoding="utf-8") == "Hello AG"
    assert "wrote 'Hello AG'" in summary
    write_event = next(event for event in reversed(memory.get_action_events()) if event["action"] == "WRITE")
    assert write_event["status"] == "SUCCESS"
    assert write_event["actual_result"]["contents"] == "Hello AG"
    assert write_event["verification_status"] == "VERIFIED"
    assert write_event["verification_evidence"]["method"] == "independent read-back"
    assert write_event["resolved_path"] == str(file_path)
    events = memory.get_action_events()
    write_index = events.index(write_event)
    assert write_event["depends_on_action_ids"] == [events[write_index - 1]["action_id"]]


def test_compound_read_then_explain_receives_actual_file_contents(tmp_path):
    control = DirectoryControl()
    control.set_directory(str(tmp_path))
    (tmp_path / "compound_test.txt").write_text("print('verified')", encoding="utf-8")
    memory = WorkingMemory()
    analysis = analyze_request(
        "Read compound_test.txt and tell me what it does.", Ag.detect_intent,
    )
    assert analysis.is_compound
    with patch.object(Ag, "_pipeline_response", return_value="AG: explained from contents") as brain:
        result = Ag.process_input(
            analysis.original_text, {}, {"active": [], "completed": []}, "unknown",
            directory_control=control, quips_result=analysis,
            request_unit=analysis.requests[0], working_memory=memory,
        )

    assert result.endswith("AG: explained from contents")
    assert "print('verified')" in brain.call_args.args[0]


def test_contextual_write_fails_if_the_referenced_file_disappears(tmp_path):
    control = DirectoryControl()
    control.set_directory(str(tmp_path))
    memory = WorkingMemory()
    assert "created: test.txt" in _process("create test.txt", control, memory)
    (tmp_path / "test.txt").unlink()

    response = _process("write 'Hello AG' into it", control, memory)

    assert "no longer exists" in response
    assert not (tmp_path / "test.txt").exists()
    event = memory.get_action_events()[-1]
    assert event["status"] == "FAILED"
    assert event["failure_stage"] == "target_resolution"


def test_write_execution_failure_is_recorded(tmp_path, monkeypatch):
    control = DirectoryControl()
    control.set_directory(str(tmp_path))
    memory = WorkingMemory()
    assert "created: test.txt" in _process("create test.txt", control, memory)

    def fail_write(_path, _content):
        raise Ag.DirectoryControlError("simulated write failure")

    monkeypatch.setattr(control, "write_file", fail_write)
    response = _process("write 'Hello AG' into it", control, memory)

    assert "simulated write failure" in response
    event = memory.get_action_events()[-1]
    assert event["status"] == "FAILED"
    assert event["failure_stage"] == "execution"


def test_write_verification_disagreement_is_not_reported_as_success(tmp_path, monkeypatch):
    control = DirectoryControl()
    control.set_directory(str(tmp_path))
    memory = WorkingMemory()
    assert "created: test.txt" in _process("create test.txt", control, memory)
    monkeypatch.setattr(control, "read_file", lambda _path: "different")

    response = _process("write 'Hello AG' into it", control, memory)

    assert "Verification failed" in response
    event = memory.get_action_events()[-1]
    assert event["status"] == "FAILED"
    assert event["verification_status"] == "FAILED"
    assert event["verification_evidence"]["matches"] is False


def test_create_failure_is_recorded_and_dependent_unit_is_not_run(tmp_path, monkeypatch):
    control = DirectoryControl()
    control.set_directory(str(tmp_path))
    memory = WorkingMemory()

    def fail_create(_path, _content=""):
        raise Ag.DirectoryControlError("simulated creation failure")

    monkeypatch.setattr(control, "create_file", fail_create)
    analysis = analyze_request(
        "create test.txt and explain what you created", Ag.detect_intent,
    )
    response = Ag.process_input(
        analysis.original_text, {}, {"active": [], "completed": []}, "unknown",
        directory_control=control, quips_result=analysis,
        request_unit=analysis.requests[0], working_memory=memory,
    )

    assert "simulated creation failure" in response
    assert "did not succeed" in response
    assert len(memory.get_action_events()) == 1
    assert memory.get_action_events()[0]["status"] == "FAILED"


def test_generic_file_reference_is_ambiguous_when_multiple_prior_files_match(tmp_path):
    control = DirectoryControl()
    control.set_directory(str(tmp_path))
    memory = WorkingMemory()
    assert "created: first.txt" in _process("create first.txt", control, memory)
    assert "created: second.txt" in _process("create second.txt", control, memory)

    response = _process("read the file", control, memory)

    assert "Which file" in response
    assert str(tmp_path / "first.txt") in response
    assert str(tmp_path / "second.txt") in response
    assert list(tmp_path.iterdir()) == [tmp_path / "first.txt", tmp_path / "second.txt"]


def test_explicit_older_filename_and_topic_isolation(tmp_path):
    control = DirectoryControl()
    control.set_directory(str(tmp_path))
    memory = WorkingMemory()
    assert "created: older.txt" in _process("create older.txt", control, memory)
    assert "created: newer.txt" in _process("create newer.txt", control, memory)
    (tmp_path / "older.txt").write_text("older result", encoding="utf-8")
    (tmp_path / "newer.txt").write_text("newer result", encoding="utf-8")

    assert "older result" in _process("read older.txt", control, memory)
    assert "older result" in _process("read the older file", control, memory)

    memory.update_working_memory("what is a black hole?", "An astronomical object.", "unknown")
    memory.update_working_memory("tell me about galaxies", "A topic change.", "unknown")
    isolated = _process("open it", control, memory)
    assert "valid previous filesystem item" in isolated


def test_copy_and_move_keep_using_directory_control_with_action_evidence(tmp_path):
    control = DirectoryControl()
    control.set_directory(str(tmp_path))
    source = tmp_path / "source.txt"
    source.write_text("source contents", encoding="utf-8")
    memory = WorkingMemory()

    assert "Copied to: copied.txt" in _process("copy source.txt to copied.txt", control, memory)
    assert source.read_text(encoding="utf-8") == "source contents"
    assert (tmp_path / "copied.txt").read_text(encoding="utf-8") == "source contents"
    copied = memory.get_action_events()[-1]
    assert copied["action"] == "COPY"
    assert copied["status"] == "SUCCESS"
    assert copied["verification_status"] == "VERIFIED"

    assert "Moved to: moved.txt" in _process("move copied.txt to moved.txt", control, memory)
    assert not (tmp_path / "copied.txt").exists()
    assert (tmp_path / "moved.txt").read_text(encoding="utf-8") == "source contents"
    moved = memory.get_action_events()[-1]
    assert moved["action"] == "MOVE"
    assert moved["status"] == "SUCCESS"
    assert moved["verification_evidence"]["source_exists"] is False


def test_semantic_reasoner_is_fallback_only_and_returns_validated_structure():
    calls = []

    def interpret(raw_input, _context):
        calls.append(raw_input)
        return {
            "action": "EXPLAIN",
            "target": "unclassified thing",
            "target_type": "UNKNOWN",
            "domain": "GENERAL",
            "desired_outcome": DesiredOutcome.GENERAL_RESPONSE,
            "is_complete": True,
        }

    evaluator = CognitiveEvaluator(semantic_reasoner=interpret)
    deterministic = _unit("read Brain.py")
    deterministic_thought = evaluator.evaluate_request(Thought("read Brain.py"), deterministic)
    assert deterministic_thought.request_understanding.desired_outcome == DesiredOutcome.FILE_CONTENT
    assert calls == []

    unclear = _unit("Florbulate the unclassified thing.")
    thought = evaluator.evaluate_request(Thought(unclear.text), unclear)
    assert calls == [unclear.text]
    assert thought.request_understanding.target == "unclassified thing"
    assert not thought.semantic_reasoning_required


def test_semantic_reasoner_cannot_return_execution_instructions():
    evaluator = CognitiveEvaluator(
        semantic_reasoner=lambda _text, _context: {
            "action": "CREATE",
            "target": "unsafe.txt",
            "target_type": "FILE",
            "domain": "FILESYSTEM",
            "desired_outcome": DesiredOutcome.CREATE_FILE,
            "execute": True,
        }
    )
    unclear = _unit("Florbulate the unclassified thing.")

    try:
        evaluator.evaluate_request(Thought(unclear.text), unclear)
    except ValueError as exc:
        assert "unsupported interpretation fields" in str(exc)
    else:
        raise AssertionError("Execution fields must not be accepted from semantic interpretation.")


def test_semantic_reference_without_matching_memory_is_incomplete():
    evaluator = CognitiveEvaluator(
        semantic_reasoner=lambda _text, _context: {
            "action": "OPEN",
            "target": "the thing I created earlier",
            "target_type": TargetType.UNKNOWN,
            "domain": "FILESYSTEM",
            "desired_outcome": DesiredOutcome.OPEN_TARGET,
            "references": ("the thing I created earlier",),
            "context_requirements": ("previously-mentioned filesystem entity",),
            "is_complete": True,
        }
    )
    unit = _unit("take me to the thing I created earlier")
    thought = evaluator.evaluate_request(
        Thought(unit.text), unit, WorkingMemory(),
    )

    assert not thought.is_complete
    assert "matching previous filesystem entity" in thought.request_understanding.missing_information


def test_semantic_fallback_interprets_before_existing_filesystem_execution(tmp_path):
    control = DirectoryControl()
    control.set_directory(str(tmp_path))
    (tmp_path / "test.txt").write_text("resolved content", encoding="utf-8")
    calls = []

    def interpret(raw_input, _context):
        calls.append(raw_input)
        return {
            "action": "READ",
            "target": "test.txt",
            "target_type": "FILE",
            "domain": "FILESYSTEM",
            "desired_outcome": DesiredOutcome.FILE_CONTENT,
            "is_complete": True,
        }

    text = "take me to the thing"
    analysis = analyze_request(text, Ag.detect_intent)
    unit = analysis.requests[0]
    assert unit.understanding.semantic_reasoning_required

    result = Ag.process_input(
        text,
        {},
        {"active": [], "completed": []},
        "unknown",
        directory_control=control,
        quips_result=analysis,
        request_unit=unit,
        working_memory=WorkingMemory(),
        semantic_reasoner=interpret,
    )

    assert calls == [text]
    assert "resolved content" in result
    assert (tmp_path / "test.txt").exists()


def test_configured_semantic_fallback_parses_structured_output():
    response = (
        '{"action":"READ","target":"test.txt","target_type":"FILE",'
        '"domain":"FILESYSTEM","desired_outcome":"FILE_CONTENT",'
        '"references":[],"context_requirements":[],"missing_information":[],'
        '"is_complete":true}'
    )
    with patch.object(Ag, "ask_brain", return_value=response) as brain:
        parsed = Ag._semantic_interpretation_fallback("take me to the thing", {})

    assert parsed["action"] == "READ"
    assert parsed["target"] == "test.txt"
    assert "Do not execute anything" in brain.call_args.args[0]


def test_domains_remain_isolated_from_active_workspace():
    assert _unit("what files are here?").understanding.domain == "FILESYSTEM"
    assert _unit("what are your capabilities?").understanding.domain == "SELF_INFO"
    conversation = _unit("what did I just ask you?")
    assert conversation.category == RequestCategory.CONVERSATION
    assert conversation.understanding.domain == "CONVERSATION"
