from types import SimpleNamespace
from unittest.mock import patch

import Jarwis as Ag
from preprocessor_brain import PreprocessorBrain
from quips import RequestCategory, RoutingDecision, analyze_request


def _unit(query):
    return analyze_request(query, Ag.detect_intent).requests[0]


def _process(query, *, preprocessor=None, directory_control=None, directory_session=None):
    analysis = analyze_request(query, Ag.detect_intent)
    unit = analysis.requests[0]
    return Ag.process_input(
        query,
        {},
        {"active": [], "completed": []},
        unit.legacy_intent or Ag.detect_intent(query),
        directory_control=directory_control,
        directory_session=directory_session,
        preprocessor_brain=preprocessor,
        quips_result=analysis,
        request_unit=unit,
    )


def test_workspace_list_phrases_are_classified_as_filesystem_requests():
    for query in ("what files are here?", "list this directory"):
        unit = _unit(query)
        assert unit.category == RequestCategory.WORKSPACE_QUERY
        assert unit.intent == "WORKSPACE_LIST"
        assert unit.routing_decision == RoutingDecision.LEGACY_INTENT_ROUTER


def test_filesystem_request_uses_active_workspace(tmp_path):
    (tmp_path / "test.txt").write_text("workspace file contents", encoding="utf-8")
    directory_control = Ag.DirectoryControl()
    preprocessor = PreprocessorBrain(directory_control=directory_control)

    set_result = _process(
        f"set workspace to {tmp_path}",
        preprocessor=preprocessor,
        directory_control=directory_control,
    )
    assert str(tmp_path) in set_result
    assert directory_control.get_active_directory() == str(tmp_path)

    listing = _process(
        "what files are here?",
        preprocessor=preprocessor,
        directory_control=directory_control,
    )
    assert "test.txt" in listing

    contents = _process(
        "read test.txt",
        preprocessor=preprocessor,
        directory_control=directory_control,
    )
    assert "workspace file contents" in contents


def test_general_and_conversation_requests_bypass_active_workspace():
    class NoWorkspaceRouting:
        def process(self, _query):
            raise AssertionError("non-filesystem request reached PreprocessorBrain")

    directory_session = SimpleNamespace(is_active=True)
    for query, category in (
        ("how does QUIPS work?", RequestCategory.GENERAL),
        ("what did I just ask you?", RequestCategory.CONVERSATION),
        ("what did I just ask you to read?", RequestCategory.CONVERSATION),
        ("what did I just ask you to read from the folder?", RequestCategory.CONVERSATION),
    ):
        unit = _unit(query)
        assert unit.category == category
        if category == RequestCategory.CONVERSATION:
            assert unit.target_reference is None
        with patch.object(Ag, "_pipeline_response", return_value="AG: normal response"):
            result = _process(
                query,
                preprocessor=NoWorkspaceRouting(),
                directory_session=directory_session,
            )
        assert result == "AG: normal response"


def test_self_info_keeps_its_existing_route_with_workspace_set():
    class SelfInfoOnly:
        def process(self, _query):
            raise AssertionError("self-info should use its deterministic route")

        def process_self_info_intent(self, _intent, _query):
            return SimpleNamespace(success=True, result="self-info response")

    unit = _unit("what are your capabilities?")
    assert unit.category == RequestCategory.SELF_INFO
    assert unit.routing_decision == RoutingDecision.DETERMINISTIC_SELF_INFO
    assert _process(
        "what are your capabilities?",
        preprocessor=SelfInfoOnly(),
        directory_session=SimpleNamespace(is_active=True),
    ) == "Jarwis: self-info response"


def test_general_request_does_not_require_a_workspace():
    class NoWorkspaceRouting:
        def process(self, _query):
            raise AssertionError("general request reached workspace routing")

    with patch.object(Ag, "_pipeline_response", return_value="AG: normal response"):
        result = _process("how does QUIPS work?", preprocessor=NoWorkspaceRouting())

    assert result == "AG: normal response"
