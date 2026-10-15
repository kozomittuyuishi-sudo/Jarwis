from pathlib import Path

from quips import analyze_request, resolve_target_reference
from target_resolution_models import ResolutionMode


def _legacy_intent(text: str) -> str:
    return "legacy:" + text.lower()


def test_explicit_absolute_target_is_passed_to_target_reference():
    unit = analyze_request("read D:/AG/test.txt", _legacy_intent).requests[0]
    ref = unit.target_reference

    assert ref is not None
    assert ref.original_target == "D:/AG/test.txt"
    assert ref.resolution_mode == ResolutionMode.ABSOLUTE_PATH
    assert ref.scope is None


def test_workspace_relative_target_is_preserved():
    workspace = str(Path(__file__).resolve().parents[1])
    unit = analyze_request("read test.txt", _legacy_intent).requests[0]
    ref = unit.target_reference

    assert ref is not None
    assert ref.original_target == "test.txt"
    assert ref.resolution_mode == ResolutionMode.WORKSPACE_RELATIVE

    result = resolve_target_reference(unit, workspace=workspace)
    assert result is not None
    assert result.status in {"RESOLVED", "NOT_FOUND"}


def test_named_scope_target_is_wrapped_in_target_reference():
    workspace = str(Path(__file__).resolve().parents[1])
    unit = analyze_request("read test.txt in AG", _legacy_intent).requests[0]
    ref = unit.target_reference

    assert ref is not None
    assert ref.original_target == "test.txt"
    assert ref.scope == "AG"
    assert ref.resolution_mode == ResolutionMode.NAMED_SCOPE

    result = resolve_target_reference(unit, workspace=workspace, scope_registry={"AG": workspace})
    assert result is not None
    assert result.status == "RESOLVED"


def test_compound_request_only_routes_filesystem_unit():
    result = analyze_request("create test.txt and explain what you created", _legacy_intent)

    assert result.is_compound
    filesystem_unit = result.requests[0]
    explanation_unit = result.requests[1]
    assert filesystem_unit.target_reference is not None
    assert filesystem_unit.target_reference.original_target == "test.txt"
    assert explanation_unit.target_reference is None
