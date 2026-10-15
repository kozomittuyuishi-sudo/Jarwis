import pytest

import Jarwis as Ag
from directory_control import DirectoryControl
from preprocessor_brain import PreprocessorBrain


def _send(query, control, *, preprocessor=None):
    return Ag.process_input(
        query,
        {},
        {"active": [], "completed": []},
        Ag.detect_intent(query),
        directory_control=control,
        preprocessor_brain=preprocessor,
    )


def test_workspace_set_clear_state_transition(tmp_path):
    workspace = tmp_path / "AG"
    workspace.mkdir()
    (workspace / "workspace_only.txt").write_text("old workspace", encoding="utf-8")
    control = DirectoryControl()
    preprocessor = PreprocessorBrain(directory_control=control)

    assert control.get_active_directory() is None
    _send(f"set workspace to {workspace}", control, preprocessor=preprocessor)
    assert control.get_active_directory() == str(workspace)

    assert "Active workspace cleared:" in _send(
        "clear workspace", control, preprocessor=preprocessor
    )
    assert control.get_active_directory() is None
    assert "No active workspace" in _send("what is the current workspace?", control)
    assert "No active workspace" in _send("what files are here?", control)
    after_clear = _send(
        "what files are here?", control, preprocessor=preprocessor
    )
    assert "workspace_only.txt" not in after_clear
    assert control.get_active_directory() is None


def test_valid_switch_then_clear(tmp_path):
    first = tmp_path / "AG"
    second = tmp_path / "Project"
    first.mkdir()
    second.mkdir()
    control = DirectoryControl()

    _send(f"set workspace to {first}", control)
    assert control.get_active_directory() == str(first)

    assert "Switched active workspace to:" in _send(
        f"switch workspace to {second}", control
    )
    assert control.get_active_directory() == str(second)

    _send("leave workspace", control)
    assert control.get_active_directory() is None


@pytest.mark.parametrize("command", [
    "leave workspace",
    "leave the workspace",
    "clear workspace",
    "clear the workspace",
    "unset workspace",
    "exit workspace",
    "stop using this workspace",
])
def test_natural_clear_commands_clear_stored_state(command, tmp_path):
    workspace = tmp_path / "AG"
    workspace.mkdir()
    control = DirectoryControl()
    control.set_directory(str(workspace))

    assert Ag.detect_intent(command) == "dc_clear_directory"
    _send(command, control)
    assert control.get_active_directory() is None


@pytest.mark.parametrize("operation", ["set", "switch"])
def test_invalid_workspace_path_preserves_existing_workspace(operation, tmp_path):
    workspace = tmp_path / "AG"
    workspace.mkdir()
    control = DirectoryControl()
    control.set_directory(str(workspace))
    invalid_path = tmp_path / "does-not-exist"
    command = f"{operation} workspace to {invalid_path}"

    _send(command, control)

    assert control.get_active_directory() == str(workspace)


def test_exit_word_alone_does_not_clear_workspace(tmp_path):
    workspace = tmp_path / "AG"
    workspace.mkdir()
    control = DirectoryControl()
    control.set_directory(str(workspace))

    assert Ag.detect_intent("exit the process") != "dc_clear_directory"
    assert control.get_active_directory() == str(workspace)
