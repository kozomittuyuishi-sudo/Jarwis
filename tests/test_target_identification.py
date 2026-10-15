from pathlib import Path

import Jarwis as Ag
from quips import RequestCategory, analyze_request, resolve_target_reference
from target_resolution_models import (
    ExistenceState,
    ResolutionMode,
    ResolutionStatus,
    TargetType,
    TargetReference,
)
from target_resolver import TargetResolver
from directory_control import DirectoryControl


def _unit(query):
    return analyze_request(query, Ag.detect_intent).requests[0]


def _resolve(unit, workspace):
    return resolve_target_reference(
        unit,
        resolver=TargetResolver(workspace_source=workspace),
    )


def _process(query, control):
    analysis = analyze_request(query, Ag.detect_intent)
    unit = analysis.requests[0]
    return Ag.process_input(
        query,
        {},
        {"active": [], "completed": []},
        unit.legacy_intent or Ag.detect_intent(query),
        directory_control=control,
        quips_result=analysis,
        request_unit=unit,
    )


def test_existing_file_is_identified_and_read(tmp_path):
    file_path = tmp_path / "Brain.py"
    file_path.write_text("brain contents", encoding="utf-8")
    control = DirectoryControl()
    control.set_directory(str(tmp_path))
    unit = _unit("read Brain.py")

    assert unit.target_reference.original_target == "Brain.py"
    assert unit.target_reference.target_type == TargetType.FILE
    resolved = _resolve(unit, control)
    assert resolved.status == ResolutionStatus.RESOLVED
    assert resolved.resolved_path == str(file_path)
    assert resolved.target_type == TargetType.FILE
    assert resolved.existence_state == ExistenceState.EXISTS_FILE
    assert "brain contents" in _process("read Brain.py", control)


def test_existing_directory_is_identified_and_opened(tmp_path):
    brain_dir = tmp_path / "brains"
    brain_dir.mkdir()
    (brain_dir / "local.py").write_text("local", encoding="utf-8")
    control = DirectoryControl()
    control.set_directory(str(tmp_path))
    unit = _unit("open brains")

    assert unit.target_reference.original_target == "brains"
    assert unit.target_reference.target_type == TargetType.UNKNOWN
    resolved = _resolve(unit, control)
    assert resolved.status == ResolutionStatus.RESOLVED
    assert resolved.resolved_path == str(brain_dir)
    assert resolved.target_type == TargetType.DIRECTORY
    assert resolved.existence_state == ExistenceState.EXISTS_DIRECTORY
    response = _process("open brains", control)
    assert "[F] local.py" in response


def test_filesystem_type_overrides_extension_hint(tmp_path):
    directory_named_like_file = tmp_path / "folder.txt"
    directory_named_like_file.mkdir()
    (directory_named_like_file / "inside.txt").write_text("inside", encoding="utf-8")
    (tmp_path / "plain_name").write_text("no extension", encoding="utf-8")
    control = DirectoryControl()
    control.set_directory(str(tmp_path))

    directory_unit = _unit("open folder.txt")
    assert directory_unit.target_reference.target_type == TargetType.FILE
    directory_result = _resolve(directory_unit, control)
    assert directory_result.target_type == TargetType.DIRECTORY
    assert directory_result.existence_state == ExistenceState.EXISTS_DIRECTORY
    assert "[F] inside.txt" in _process("open folder.txt", control)

    file_result = _resolve(_unit("open plain_name"), control)
    assert file_result.target_type == TargetType.FILE
    assert file_result.existence_state == ExistenceState.EXISTS_FILE
    assert "no extension" in _process("open plain_name", control)


def test_workspace_relative_target_type_comes_from_filesystem():
    workspace = Path(__file__).resolve().parents[1]
    control = DirectoryControl()
    control.set_directory(str(workspace))

    file_result = _resolve(_unit("read Jarwis.py"), control)
    assert file_result.resolved_path == str(workspace / "Jarwis.py")
    assert file_result.target_type == TargetType.FILE
    assert file_result.existence_state == ExistenceState.EXISTS_FILE

    directory_result = _resolve(_unit("open brains"), control)
    assert directory_result.resolved_path == str(workspace / "brains")
    assert directory_result.target_type == TargetType.DIRECTORY
    assert directory_result.existence_state == ExistenceState.EXISTS_DIRECTORY


def test_create_target_type_uses_explicit_request_semantics():
    file_unit = _unit("create a file called test.txt")
    folder_unit = _unit("create a folder called AB_Test")

    assert file_unit.target_reference.target_type == TargetType.FILE
    assert folder_unit.target_reference.target_type == TargetType.DIRECTORY


def test_conversation_requests_never_create_filesystem_targets():
    for query in (
        "what did I just ask you to read?",
        "what did I ask you to read?",
        "what were we talking about?",
        "explain what you just told me",
    ):
        unit = _unit(query)
        assert unit.category == RequestCategory.CONVERSATION
        assert unit.target_reference is None


def test_nonexistent_bare_open_target_keeps_unknown_type(tmp_path):
    unit = _unit("open missing_target")
    resolved = _resolve(unit, str(tmp_path))

    assert unit.target_reference.target_type == TargetType.UNKNOWN
    assert resolved.status == ResolutionStatus.NOT_FOUND
    assert resolved.target_type == TargetType.UNKNOWN
    assert not Path(resolved.resolved_path).exists()


def test_compound_create_and_explanation_keeps_target_types_separate():
    result = analyze_request("create test.txt and explain what you created", Ag.detect_intent)

    assert result.is_compound
    assert result.requests[0].category == RequestCategory.FILE_OPERATION
    assert result.requests[0].target_reference.target_type == TargetType.FILE
    assert result.requests[1].category == RequestCategory.EXPLANATION
    assert result.requests[1].target_reference is None


def test_resolver_preserves_ambiguous_search_candidates(tmp_path):
    (tmp_path / "foo").mkdir()
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "foo").write_text("nested match", encoding="utf-8")
    ref = TargetReference(
        original_target="foo",
        resolution_mode=ResolutionMode.SMART_SEARCH,
    )

    result = TargetResolver(workspace_source=str(tmp_path)).resolve(ref)

    assert result.status == ResolutionStatus.AMBIGUOUS
    assert len(result.candidates) == 2
    assert result.resolved_path is None


def test_exact_directory_and_same_stem_file_are_not_chosen_silently(tmp_path):
    (tmp_path / "foo").mkdir()
    (tmp_path / "foo.txt").write_text("file candidate", encoding="utf-8")
    control = DirectoryControl()
    control.set_directory(str(tmp_path))
    unit = _unit("open foo")

    result = _resolve(unit, control)

    assert result.status == ResolutionStatus.AMBIGUOUS
    assert set(result.candidates) == {
        str(tmp_path / "foo"),
        str(tmp_path / "foo.txt"),
    }
    assert result.target_type == TargetType.UNKNOWN
    assert result.resolved_path is None
    assert "ambiguous" in _process("open foo", control).lower()
