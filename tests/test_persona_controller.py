from types import SimpleNamespace

import Jarwis as Ag
import brains.brain as brain
import self_info.ab_self_model as self_model
from directory_control import DirectoryControl
from memory.working_memory import WorkingMemory
from quips import analyze_request
from runtime.persona_controller import PersonaController
from self_info import SelfInfoEngine


def test_persona_defaults_to_jarwis_and_persists_mode(tmp_path):
    config = tmp_path / "persona.json"
    controller = PersonaController(config)

    assert controller.mode == "JARWIS"
    assert controller.set_mode("ultron")
    assert PersonaController(config).mode == "ULTRON"
    assert not controller.set_mode("unknown")
    assert controller.mode == "ULTRON"


def test_identity_is_jarwis_independent_of_active_persona():
    identity = SelfInfoEngine().answer("who are you").context_block

    assert "Jarwis" in identity
    assert "Short name: J" in identity
    assert "Name:     AG" not in identity
    assert "verified from project_context.json" not in identity
    assert "Active persona:" in identity


def test_creator_is_reported_only_when_explicitly_configured(tmp_path, monkeypatch):
    context = tmp_path / "project_context.json"
    monkeypatch.setattr(self_model, "_PROJECT_CONTEXT_PATH", context)
    context.write_text('{"current_version": "v1"}', encoding="utf-8")
    assert "creator" not in self_model.get_identity()

    context.write_text(
        '{"current_version": "v1", "creator_name": "Configured Creator"}',
        encoding="utf-8",
    )
    assert self_model.get_identity()["creator"] == "Configured Creator"


def test_named_greetings_are_deterministic_and_report_active_persona(
    tmp_path, monkeypatch
):
    controller = PersonaController(tmp_path / "persona.json")
    monkeypatch.setattr(Ag, "get_persona_controller", lambda: controller)
    monkeypatch.setattr(
        Ag,
        "ask_brain",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("greeting must not dispatch to a brain")
        ),
    )

    jarwis_greeting = Ag.process_input(
        "Hey Jarwis", {}, {"active": [], "completed": []},
        Ag.detect_intent("Hey Jarwis"),
    )
    assert Ag.detect_intent("Hey Jarwis") == "greeting"
    assert "Jarwis mode is active" in jarwis_greeting

    assert controller.set_mode("ULTRON")
    ultron_greeting = Ag.process_input(
        "Hey Ultron", {}, {"active": [], "completed": []},
        Ag.detect_intent("Hey Ultron"),
    )
    assert Ag.detect_intent("Hey Ultron") == "greeting"
    assert "Ultron mode is active" in ultron_greeting


def test_short_name_and_identity_are_local_and_stable_in_both_personas(
    tmp_path, monkeypatch
):
    controller = PersonaController(tmp_path / "persona.json")
    monkeypatch.setattr(Ag, "get_persona_controller", lambda: controller)
    monkeypatch.setattr(
        Ag,
        "ask_brain",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("identity questions must not dispatch to a brain")
        ),
    )

    for mode in ("JARWIS", "ULTRON"):
        assert controller.set_mode(mode)
        identity = Ag.process_input(
            "Who are you?", {}, {"active": [], "completed": []},
            Ag.detect_intent("Who are you?"),
        )
        named_identity = Ag.process_input(
            "What's your name?", {}, {"active": [], "completed": []},
            Ag.detect_intent("What's your name?"),
        )
        short_name = Ag.process_input(
            "What's your short name?", {}, {"active": [], "completed": []},
            Ag.detect_intent("What's your short name?"),
        )
        assert "I am Jarwis" in identity
        assert "I am Jarwis" in named_identity
        assert "My short name is J." in short_name
        assert "Jarwis J" not in identity + named_identity + short_name

    assert Ag.detect_intent("Hey J") == "greeting"
    for mode in ("JARWIS", "ULTRON"):
        assert controller.set_mode(mode)
        greeting = Ag.process_input(
            "Hey J", {}, {"active": [], "completed": []},
            Ag.detect_intent("Hey J"),
        )
        assert f"{mode.title()} mode is active" in greeting


def test_greetings_and_identity_stay_local_when_cloud_generation_fails(
    tmp_path, monkeypatch
):
    controller = PersonaController(tmp_path / "persona.json")
    monkeypatch.setattr(Ag, "get_persona_controller", lambda: controller)
    monkeypatch.setattr(brain, "load_brain_mode", lambda: "cloud")
    cloud_calls = []

    def fail_cloud(*_args, **_kwargs):
        cloud_calls.append(True)
        raise RuntimeError("simulated provider failure")

    monkeypatch.setattr(brain, "ask_cloud_brain", fail_cloud)

    greeting = Ag.process_input(
        "hey J", {}, {"active": [], "completed": []}, Ag.detect_intent("hey J"),
    )
    identity = Ag.process_input(
        "Who are you?", {}, {"active": [], "completed": []},
        Ag.detect_intent("Who are you?"),
    )

    assert "Jarwis mode is active" in greeting
    assert "I am Jarwis" in identity
    assert cloud_calls == []


def test_contextual_addressing_is_consistent_and_applied_sparingly(tmp_path):
    controller = PersonaController(tmp_path / "persona.json")

    assert controller.prepare_turn("Hey J", "greeting") == "B"
    assert controller.apply_address("Jarwis: Hello.", controller.current_address) == (
        "Jarwis: B, Hello."
    )
    assert controller.prepare_turn("That's helpful.", "unknown") is None
    assert "'B'" in controller.addressing_guidance

    assert controller.prepare_turn("Create a test file.", "create_file") == "Sir"
    assert controller.apply_address(
        "Jarwis: File created and verified.", controller.current_address,
    ) == "Jarwis: Sir, File created and verified."
    assert controller.prepare_turn("Write content to the file.", "write_file") is None
    assert controller.prepare_turn("Read the file.", "read_file") is None
    assert controller.prepare_turn("Inspect the file.", "read_file") == "Sir"


def test_recent_session_turns_are_included_in_brain_generation_prompt(monkeypatch):
    manager = Ag.ConversationManager()
    manager.add_turn("user", "brainstorm me?")
    manager.add_turn(
        "assistant",
        "The answer stopped mid-sentence at the first useful observation.",
    )
    manager.add_turn("user", "That response was truncated mid-sentence.")
    received = []
    monkeypatch.setattr(
        Ag, "ask_brain",
        lambda prompt: received.append(prompt) or "I can continue from that point.",
    )

    response = Ag._pipeline_response(
        "brainstorm me?",
        "unknown",
        manager,
        None,
        None,
        None,
        None,
    )

    assert response == "Jarwis: I can continue from that point."
    assert "Recent conversation context from this current session" in received[0]
    assert "The answer stopped mid-sentence" in received[0]
    assert "That response was truncated mid-sentence." in received[0]
    assert received[0].endswith("Current user request:\nbrainstorm me?")


def test_dispatcher_planning_leak_is_replaced_before_terminal_rendering():
    class Dispatcher:
        def dispatch(self, _payload):
            return SimpleNamespace(
                content="appropriately: acknowledge the prompt, then respond",
            )

    response = Ag._pipeline_response(
        "A test request",
        "unknown",
        None,
        None,
        None,
        Dispatcher(),
        Ag.ResponseProcessor(),
    )

    assert response == f"Jarwis: {Ag.BRAIN_FALLBACK}"


def test_startup_presents_jarwis_and_active_persona(tmp_path, monkeypatch, capsys):
    controller = PersonaController(tmp_path / "persona.json")
    assert controller.set_mode("ULTRON")
    monkeypatch.setattr(Ag, "get_persona_controller", lambda: controller)

    Ag.startup()

    output = capsys.readouterr().out
    assert "Jarwis:" in output
    assert "Ultron mode active" in output


def test_persona_commands_are_deterministic_and_preserve_session_state(
    tmp_path, monkeypatch
):
    controller = PersonaController(tmp_path / "persona.json")
    monkeypatch.setattr(Ag, "get_persona_controller", lambda: controller)
    memory = {"general": {"fact": "kept"}}
    tasks = {"active": ["task"], "completed": []}
    working_memory = Ag.WorkingMemory()
    working_memory.add_to_discussion("earlier question", "earlier answer")

    to_ultron = Ag.process_input(
        "Switch to Ultron mode.", memory, tasks, "unknown",
        working_memory=working_memory,
    )
    assert controller.mode == "ULTRON"
    assert "Ultron mode enabled" in to_ultron
    assert "Ultron mode" in Ag.process_input(
        "What mode are you in?", memory, tasks, "unknown",
    )

    to_jarwis = Ag.process_input(
        "Please switch to Jarwis mode", memory, tasks, "unknown",
        working_memory=working_memory,
    )
    assert controller.mode == "JARWIS"
    assert "Jarwis mode enabled" in to_jarwis
    assert memory == {"general": {"fact": "kept"}}
    assert tasks == {"active": ["task"], "completed": []}
    assert working_memory.get_discussion_entries()


def test_invalid_persona_command_is_safe_and_does_not_capture_brain_mode(
    tmp_path, monkeypatch
):
    controller = PersonaController(tmp_path / "persona.json")
    monkeypatch.setattr(Ag, "get_persona_controller", lambda: controller)

    invalid = Ag.process_input(
        "Switch to villain mode", {}, {"active": [], "completed": []}, "unknown",
    )
    assert "not a supported persona" in invalid
    assert controller.mode == "JARWIS"

    # Existing brain-selection commands remain in their existing route.
    assert Ag._persona_control_response("set brain mode to local") is None


def test_filesystem_execution_and_verification_are_shared_by_both_personas(
    tmp_path, monkeypatch
):
    controller = PersonaController(tmp_path / "persona.json")
    monkeypatch.setattr(Ag, "get_persona_controller", lambda: controller)
    directory = DirectoryControl()
    directory.set_directory(str(tmp_path))
    working_memory = WorkingMemory()

    for mode, filename in (
        ("JARWIS", "jarwis-result.txt"),
        ("ULTRON", "ultron-result.txt"),
    ):
        assert controller.set_mode(mode)
        query = f"Create a file called {filename}."
        analysis = analyze_request(query, Ag.detect_intent)
        unit = analysis.requests[0]
        response = Ag.process_input(
            query,
            {},
            {"active": [], "completed": []},
            unit.legacy_intent or Ag.detect_intent(query),
            directory_control=directory,
            quips_result=analysis,
            request_unit=unit,
            working_memory=working_memory,
        )
        assert "File created:" in response
        assert (tmp_path / filename).is_file()
        result = working_memory.get_action_events()[-1]
        assert result["status"] == "SUCCESS"
        assert result["verification_status"] == "VERIFIED"


def test_folder_then_file_workflow_uses_verified_folder_result_and_recent_reference(
    tmp_path
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    control = DirectoryControl()
    control.set_directory(str(workspace))
    working_memory = WorkingMemory()
    query = (
        "Create a folder called compound_dir and create a file "
        "called child.txt inside it."
    )
    analysis = analyze_request(query, Ag.detect_intent)

    assert analysis.is_compound
    assert len(analysis.requests) == 2
    assert analysis.requests[1].depends_on == (0,)

    response = Ag.process_input(
        query,
        {},
        {"active": [], "completed": []},
        analysis.requests[0].legacy_intent,
        directory_control=control,
        quips_result=analysis,
        working_memory=working_memory,
    )

    child = workspace / "compound_dir" / "child.txt"
    assert child.is_file()
    assert "File created" in response
    events = working_memory.get_action_events()
    assert [event["status"] for event in events[-2:]] == ["SUCCESS", "SUCCESS"]
    assert events[-1]["resolved_path"] == str(child)
    assert events[-1]["depends_on_action_ids"] == [events[-2]["action_id"]]
    assert events[-1]["related_entities"][0]["resolved_location"] == str(
        workspace / "compound_dir"
    )

    reference = Ag.process_input(
        "What file did I just create?",
        {},
        {"active": [], "completed": []},
        Ag.detect_intent("What file did I just create?"),
        working_memory=working_memory,
    )
    assert "child.txt" in reference
    assert str(child) in reference


def test_append_file_is_deterministic_and_verified(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "notes.txt").write_text("before", encoding="utf-8")
    control = DirectoryControl()
    control.set_directory(str(workspace))
    working_memory = WorkingMemory()
    query = 'append " after" to notes.txt'
    analysis = analyze_request(query, Ag.detect_intent)
    unit = analysis.requests[0]

    assert unit.intent == "APPEND_FILE"
    response = Ag.process_input(
        query,
        {},
        {"active": [], "completed": []},
        unit.legacy_intent or Ag.detect_intent(query),
        directory_control=control,
        quips_result=analysis,
        request_unit=unit,
        working_memory=working_memory,
    )

    assert "verified it by reading it back" in response
    assert (workspace / "notes.txt").read_text(encoding="utf-8") == "before after"
    event = working_memory.get_action_events()[-1]
    assert event["action"] == "APPEND"
    assert event["status"] == "SUCCESS"
    assert event["verification_status"] == "VERIFIED"


def test_filename_search_uses_existing_directory_preprocessor(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "notes.txt").write_text("a searchable note", encoding="utf-8")
    (workspace / "other.txt").write_text("unrelated", encoding="utf-8")
    control = DirectoryControl()
    control.set_directory(str(workspace))
    query = "Find file named notes"
    analysis = analyze_request(query, Ag.detect_intent)
    unit = analysis.requests[0]

    assert unit.intent == "SEARCH_FILES"
    response = Ag.process_input(
        query,
        {},
        {"active": [], "completed": []},
        unit.legacy_intent or Ag.detect_intent(query),
        directory_control=control,
        preprocessor_brain=Ag.PreprocessorBrain(directory_control=control),
        quips_result=analysis,
        request_unit=unit,
    )

    assert "notes.txt" in response
    assert "other.txt" not in response


def test_persona_instructions_reach_local_and_cloud_generators(
    tmp_path, monkeypatch
):
    controller = PersonaController(tmp_path / "persona.json")
    monkeypatch.setattr(brain, "get_persona_controller", lambda: controller)

    local_calls = []
    monkeypatch.setattr(
        brain, "ask_local_brain",
        lambda prompt: local_calls.append(prompt) or "4",
    )
    assert brain.ask_local("What is 2 + 2?") == "4"
    assert "Active response persona: JARWIS" in local_calls[-1]
    assert "calm, articulate, composed" in local_calls[-1].lower()
    assert "do not claim an operation succeeded" in local_calls[-1].lower()
    assert "address the user naturally and sparingly" in local_calls[-1].lower()

    assert controller.set_mode("ULTRON")
    controller.prepare_turn("Create a file.", "create_file")
    assert brain.ask_local("What is 2 + 2?") == "4"
    assert "Active response persona: ULTRON" in local_calls[-1]
    assert "calculating, assertive, strategically analytical" in local_calls[-1].lower()
    assert "Challenge weak assumptions" in local_calls[-1]
    assert "contextually preferred form of address" in local_calls[-1]
    assert brain.ask_local(
        "Return a JSON object.", include_persona=False,
    ) == "4"
    assert "Active response persona" not in local_calls[-1]

    cloud_calls = []

    class Completions:
        def create(self, **kwargs):
            cloud_calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content="4"),
                )],
            )

    monkeypatch.setattr(
        brain,
        "get_client",
        lambda: SimpleNamespace(
            chat=SimpleNamespace(completions=Completions()),
        ),
    )
    monkeypatch.setattr(
        brain,
        "get_adaptive_brain",
        lambda: SimpleNamespace(
            prepare=lambda **_kwargs: SimpleNamespace(
                ready=True,
                max_tokens=32,
                validation_report=SimpleNamespace(issues=[]),
                estimation=SimpleNamespace(provider_ceiling=256),
            ),
            validate_response=lambda **_kwargs: SimpleNamespace(
                status=brain.BrainStatus.PASSED,
                retry_outcome=None,
            ),
        ),
    )
    assert brain.ask_cloud_brain("What is 2 + 2?") == "4"
    system_message = cloud_calls[-1]["messages"][0]["content"]
    assert "Active response persona: ULTRON" in system_message
    assert "Challenge weak assumptions" in system_message
