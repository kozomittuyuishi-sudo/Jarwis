"""
tests/test_inspect.py
======================
Focused tests for the INSPECT filesystem capability.

Test plan
---------
 1.  Unit — IntentDetector correctly classifies INSPECT phrases
 2.  Unit — CommandParser correctly extracts targets
 3.  Unit — DirectoryControl.inspect_target() — file metadata
 4.  Unit — DirectoryControl.inspect_target() — directory metadata
 5.  Unit — DirectoryControl.inspect_target() — absolute paths
 6.  Unit — DirectoryControl.inspect_target() — relative / nested paths
 7.  Unit — DirectoryControl.inspect_target() — not-found error
 8.  PreprocessorBrain end-to-end — "inspect Jarwis.py" style (relative file)
 9.  PreprocessorBrain end-to-end — "inspect backup" style (relative dir)
10.  PreprocessorBrain end-to-end — "inspect ." (active workspace)
11.  PreprocessorBrain end-to-end — "inspect the current workspace"
12.  PreprocessorBrain end-to-end — absolute file path
13.  PreprocessorBrain end-to-end — absolute dir path
14.  PreprocessorBrain end-to-end — nested relative path
15.  PreprocessorBrain end-to-end — not-found error
16.  Read-only guarantee — inspect never creates, modifies, or deletes
17.  Read-only guarantee — inspect never changes workspace or nav state
18.  Structured result contract — required keys always present
19.  INSPECT vs READ distinction
20.  INSPECT vs NAVIGATE distinction
21.  Regression — workspace setting still works after INSPECT
22.  Regression — navigation still works after INSPECT
23.  Regression — READ still works after INSPECT
24.  Regression — self-info still works after INSPECT
25.  Regression — existing directory workflows unaffected
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from directory_control import (
    DirectoryControl,
    FileNotFoundInWorkspace,
    NoActiveWorkspaceError,
)
from preprocessor_brain import (
    ACTIVE_WORKSPACE,
    CommandParser,
    DirErrorCode,
    DirectoryOperations,
    DirectoryOperationExecutor,
    Domain,
    Intent,
    IntentDetector,
    ParsedCommand,
    PreprocessorBrain,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture()
def ws_tree():
    """
    Workspace layout::

        <root>/
            sample.txt          (content: "hello inspect")
            data.py             (content: "# module")
            subdir/
                nested.txt      (content: "nested")
            backup/
                archived.log    (content: "log")
    """
    with tempfile.TemporaryDirectory() as root_str:
        root = Path(root_str)
        (root / "sample.txt").write_text("hello inspect", encoding="utf-8")
        (root / "data.py").write_text("# module\n", encoding="utf-8")
        subdir = root / "subdir"
        subdir.mkdir()
        (subdir / "nested.txt").write_text("nested", encoding="utf-8")
        backup = root / "backup"
        backup.mkdir()
        (backup / "archived.log").write_text("log", encoding="utf-8")
        yield root


@pytest.fixture()
def dc(ws_tree):
    """A DirectoryControl with the workspace set to ws_tree."""
    ctrl = DirectoryControl()
    ctrl.set_directory(str(ws_tree))
    return ctrl


@pytest.fixture()
def pb(ws_tree):
    """A PreprocessorBrain backed by a real DirectoryControl in ws_tree."""
    ctrl = DirectoryControl()
    ctrl.set_directory(str(ws_tree))
    return PreprocessorBrain(directory_control=ctrl)


@pytest.fixture()
def pb_standalone(ws_tree):
    """A PreprocessorBrain using the standalone DirectoryOperations fallback."""
    ops = DirectoryOperations()
    ops.set_directory(str(ws_tree))
    brain = PreprocessorBrain()
    brain.directory_ops = ops
    brain.executor = DirectoryOperationExecutor(ops)
    return brain


# ===========================================================================
# 1. IntentDetector — phrase classification
# ===========================================================================

class TestInspectIntentDetection:

    detector = IntentDetector()

    def _detect(self, text: str) -> str:
        return self.detector.detect(text).intent

    # Canonical "inspect <target>"
    def test_inspect_file(self):
        assert self._detect("inspect sample.txt") == Intent.INSPECT

    def test_inspect_py_file(self):
        assert self._detect("inspect Jarwis.py") == Intent.INSPECT

    def test_inspect_directory(self):
        assert self._detect("inspect backup") == Intent.INSPECT

    def test_inspect_dot(self):
        assert self._detect("inspect .") == Intent.INSPECT

    def test_inspect_current_workspace(self):
        assert self._detect("inspect the current workspace") == Intent.INSPECT

    def test_inspect_this_file(self):
        assert self._detect("inspect this file") == Intent.INSPECT

    def test_inspect_this_directory(self):
        assert self._detect("inspect this directory") == Intent.INSPECT

    def test_inspect_absolute_file(self):
        assert self._detect("inspect D:/AG/Jarwis.py") == Intent.INSPECT

    def test_inspect_absolute_dir(self):
        assert self._detect("inspect D:/AG") == Intent.INSPECT

    # "show information about …" variant
    def test_show_information_about_file(self):
        assert self._detect("show information about Jarwis.py") == Intent.INSPECT

    def test_show_info_about_file(self):
        assert self._detect("show info about Jarwis.py") == Intent.INSPECT

    # "tell me about …" variant
    def test_tell_me_about_file(self):
        assert self._detect("tell me about Jarwis.py") == Intent.INSPECT

    def test_tell_me_about_directory(self):
        assert self._detect("tell me about the backup directory") == Intent.INSPECT

    # INSPECT must NOT shadow READ or NAVIGATE
    def test_read_is_not_inspect(self):
        assert self._detect("read Jarwis.py") == Intent.READ_FILE

    def test_navigate_is_not_inspect(self):
        assert self._detect("go to backup") == Intent.NAVIGATE

    def test_list_is_not_inspect(self):
        assert self._detect("list the current workspace") == Intent.LIST_DIRECTORY


# ===========================================================================
# 2. CommandParser — target extraction
# ===========================================================================

class TestInspectCommandParsing:

    parser = CommandParser()

    def _parse(self, text: str) -> ParsedCommand:
        return self.parser.parse(Intent.INSPECT, text)

    def test_bare_file(self):
        cmd = self._parse("inspect sample.txt")
        assert cmd.intent == Intent.INSPECT
        assert cmd.target == "sample.txt"
        assert cmd.complete

    def test_bare_py_file(self):
        cmd = self._parse("inspect Jarwis.py")
        assert cmd.target == "Jarwis.py"

    def test_bare_directory(self):
        cmd = self._parse("inspect backup")
        assert cmd.target == "backup"
        assert cmd.complete

    def test_dot_target_is_active_workspace_sentinel(self):
        cmd = self._parse("inspect .")
        assert cmd.target == ACTIVE_WORKSPACE

    def test_here_is_active_workspace_sentinel(self):
        cmd = self._parse("inspect here")
        assert cmd.target == ACTIVE_WORKSPACE

    def test_current_workspace_phrase(self):
        cmd = self._parse("inspect the current workspace")
        assert cmd.target == ACTIVE_WORKSPACE

    def test_this_directory_phrase(self):
        cmd = self._parse("inspect this directory")
        assert cmd.target == ACTIVE_WORKSPACE

    def test_absolute_windows_path_forward_slash(self):
        cmd = self._parse("inspect D:/AG/Jarwis.py")
        assert "D:" in cmd.target or "d:" in cmd.target.lower()
        assert "Jarwis.py" in cmd.target

    def test_absolute_windows_path_backslash(self):
        cmd = self._parse("inspect D:\\AG\\Jarwis.py")
        assert "Jarwis.py" in cmd.target

    def test_relative_nested_path(self):
        cmd = self._parse("inspect backup/archived.log")
        assert "backup" in cmd.target
        assert "archived.log" in cmd.target

    def test_show_info_about_extracts_target(self):
        cmd = self._parse("show information about sample.txt")
        assert cmd.target == "sample.txt"

    def test_tell_me_about_extracts_target(self):
        cmd = self._parse("tell me about data.py")
        assert cmd.target == "data.py"


# ===========================================================================
# 3. DirectoryControl.inspect_target() — file metadata
# ===========================================================================

class TestInspectTargetFile:

    def test_file_metadata_keys(self, dc, ws_tree):
        result = dc.inspect_target("sample.txt")
        required_keys = {
            "operation", "target_type", "name", "path",
            "extension", "size", "modified", "readable", "writable",
        }
        assert required_keys.issubset(result.keys()), (
            f"Missing keys: {required_keys - result.keys()}"
        )

    def test_file_operation_field(self, dc, ws_tree):
        result = dc.inspect_target("sample.txt")
        assert result["operation"] == "INSPECT"

    def test_file_target_type(self, dc, ws_tree):
        result = dc.inspect_target("sample.txt")
        assert result["target_type"] == "file"

    def test_file_name(self, dc, ws_tree):
        result = dc.inspect_target("sample.txt")
        assert result["name"] == "sample.txt"

    def test_file_extension(self, dc, ws_tree):
        result = dc.inspect_target("sample.txt")
        assert result["extension"] == ".txt"

    def test_py_file_extension(self, dc, ws_tree):
        result = dc.inspect_target("data.py")
        assert result["extension"] == ".py"

    def test_file_path_is_absolute(self, dc, ws_tree):
        result = dc.inspect_target("sample.txt")
        assert Path(result["path"]).is_absolute()

    def test_file_path_correct(self, dc, ws_tree):
        result = dc.inspect_target("sample.txt")
        assert Path(result["path"]).resolve() == (ws_tree / "sample.txt").resolve()

    def test_file_size_correct(self, dc, ws_tree):
        result = dc.inspect_target("sample.txt")
        expected = (ws_tree / "sample.txt").stat().st_size
        assert result["size"] == expected

    def test_file_size_is_nonnegative(self, dc, ws_tree):
        result = dc.inspect_target("sample.txt")
        assert result["size"] >= 0

    def test_file_modified_is_iso_string(self, dc, ws_tree):
        result = dc.inspect_target("sample.txt")
        mtime = result["modified"]
        assert isinstance(mtime, str)
        # ISO-8601 basic check: contains 'T' separator
        assert "T" in mtime

    def test_file_readable_is_bool(self, dc, ws_tree):
        result = dc.inspect_target("sample.txt")
        assert isinstance(result["readable"], bool)

    def test_file_writable_is_bool(self, dc, ws_tree):
        result = dc.inspect_target("sample.txt")
        assert isinstance(result["writable"], bool)

    def test_file_is_readable(self, dc, ws_tree):
        result = dc.inspect_target("sample.txt")
        assert result["readable"] is True

    def test_file_is_writable(self, dc, ws_tree):
        result = dc.inspect_target("sample.txt")
        assert result["writable"] is True


# ===========================================================================
# 4. DirectoryControl.inspect_target() — directory metadata
# ===========================================================================

class TestInspectTargetDirectory:

    def test_directory_metadata_keys(self, dc, ws_tree):
        result = dc.inspect_target("backup")
        required_keys = {
            "operation", "target_type", "name", "path",
            "file_count", "directory_count", "modified",
        }
        assert required_keys.issubset(result.keys()), (
            f"Missing keys: {required_keys - result.keys()}"
        )

    def test_directory_operation_field(self, dc, ws_tree):
        result = dc.inspect_target("backup")
        assert result["operation"] == "INSPECT"

    def test_directory_target_type(self, dc, ws_tree):
        result = dc.inspect_target("backup")
        assert result["target_type"] == "directory"

    def test_directory_name(self, dc, ws_tree):
        result = dc.inspect_target("backup")
        assert result["name"] == "backup"

    def test_directory_path_is_absolute(self, dc, ws_tree):
        result = dc.inspect_target("backup")
        assert Path(result["path"]).is_absolute()

    def test_directory_path_correct(self, dc, ws_tree):
        result = dc.inspect_target("backup")
        assert Path(result["path"]).resolve() == (ws_tree / "backup").resolve()

    def test_directory_file_count(self, dc, ws_tree):
        result = dc.inspect_target("backup")
        # backup/ contains archived.log — 1 file
        assert result["file_count"] == 1

    def test_directory_dir_count(self, dc, ws_tree):
        result = dc.inspect_target("backup")
        # backup/ has no subdirectories
        assert result["directory_count"] == 0

    def test_workspace_root_dir_count(self, dc, ws_tree):
        result = dc.inspect_target(str(ws_tree))
        # ws_tree has: sample.txt, data.py (files) and subdir, backup (dirs)
        assert result["file_count"] == 2
        assert result["directory_count"] == 2

    def test_directory_modified_is_iso_string(self, dc, ws_tree):
        result = dc.inspect_target("backup")
        assert isinstance(result["modified"], str)
        assert "T" in result["modified"]

    def test_directory_no_extension_field(self, dc, ws_tree):
        result = dc.inspect_target("backup")
        # Directories must NOT have extension, size, readable, writable
        assert "extension" not in result
        assert "size" not in result
        assert "readable" not in result
        assert "writable" not in result


# ===========================================================================
# 5. DirectoryControl.inspect_target() — absolute paths
# ===========================================================================

class TestInspectTargetAbsolute:

    def test_absolute_file_path(self, dc, ws_tree):
        abs_path = str((ws_tree / "sample.txt").resolve())
        result = dc.inspect_target(abs_path)
        assert result["target_type"] == "file"
        assert result["name"] == "sample.txt"

    def test_absolute_dir_path(self, dc, ws_tree):
        abs_path = str(ws_tree.resolve())
        result = dc.inspect_target(abs_path)
        assert result["target_type"] == "directory"

    def test_absolute_path_not_in_workspace_is_accessible(self, dc, ws_tree):
        """
        Absolute paths outside the workspace are allowed for INSPECT —
        consistent with navigate_directory() which also accepts arbitrary
        absolute paths.  Inspection never modifies anything.
        """
        import sys
        # Use the Python executable's parent as a known absolute dir
        py_parent = Path(sys.executable).parent
        result = dc.inspect_target(str(py_parent))
        assert result["target_type"] == "directory"


# ===========================================================================
# 6. DirectoryControl.inspect_target() — relative/nested paths
# ===========================================================================

class TestInspectTargetRelative:

    def test_relative_file_in_subdir(self, dc, ws_tree):
        result = dc.inspect_target("subdir/nested.txt")
        assert result["target_type"] == "file"
        assert result["name"] == "nested.txt"

    def test_relative_subdir(self, dc, ws_tree):
        result = dc.inspect_target("subdir")
        assert result["target_type"] == "directory"
        assert result["name"] == "subdir"

    def test_relative_nested_dir(self, dc, ws_tree):
        result = dc.inspect_target("backup")
        assert result["target_type"] == "directory"

    def test_relative_nested_file(self, dc, ws_tree):
        result = dc.inspect_target("backup/archived.log")
        assert result["target_type"] == "file"
        assert result["name"] == "archived.log"
        assert result["extension"] == ".log"


# ===========================================================================
# 7. DirectoryControl.inspect_target() — error: not found
# ===========================================================================

class TestInspectTargetNotFound:

    def test_nonexistent_relative(self, dc):
        with pytest.raises(FileNotFoundInWorkspace):
            dc.inspect_target("does_not_exist.py")

    def test_nonexistent_absolute(self, dc):
        with pytest.raises((FileNotFoundInWorkspace, Exception)):
            dc.inspect_target("D:/this_path_definitely_does_not_exist_xyzzy")

    def test_no_active_workspace_relative(self):
        ctrl = DirectoryControl()
        with pytest.raises((NoActiveWorkspaceError, Exception)):
            ctrl.inspect_target("anything.txt")


# ===========================================================================
# 8–14. PreprocessorBrain end-to-end INSPECT
# ===========================================================================

class TestPreprocessorBrainInspect:

    # ---- 8. Relative file (real DirectoryControl) ----------------------

    def test_inspect_relative_file(self, pb, ws_tree):
        result = pb.process("inspect sample.txt")
        assert result.success, f"Failed: {result.error} / {result.metadata}"
        assert result.domain == Domain.DIRECTORY
        assert result.intent == Intent.INSPECT
        meta = result.result
        assert meta["target_type"] == "file"
        assert meta["name"] == "sample.txt"

    def test_inspect_file_correct_path(self, pb, ws_tree):
        result = pb.process("inspect sample.txt")
        assert result.success
        assert Path(result.result["path"]).resolve() == (ws_tree / "sample.txt").resolve()

    def test_inspect_py_file(self, pb, ws_tree):
        result = pb.process("inspect data.py")
        assert result.success
        assert result.result["target_type"] == "file"
        assert result.result["extension"] == ".py"

    # ---- 9. Relative directory -----------------------------------------

    def test_inspect_relative_directory(self, pb, ws_tree):
        result = pb.process("inspect backup")
        assert result.success, f"Failed: {result.error} / {result.metadata}"
        assert result.result["target_type"] == "directory"
        assert result.result["name"] == "backup"

    def test_inspect_directory_has_counts(self, pb, ws_tree):
        result = pb.process("inspect backup")
        assert result.success
        meta = result.result
        assert "file_count" in meta
        assert "directory_count" in meta
        assert meta["file_count"] == 1  # archived.log

    # ---- 10. "inspect ." → active workspace ----------------------------

    def test_inspect_dot(self, pb, ws_tree):
        result = pb.process("inspect .")
        assert result.success, f"Failed: {result.error} / {result.metadata}"
        assert result.result["target_type"] == "directory"
        assert Path(result.result["path"]).resolve() == ws_tree.resolve()

    # ---- 11. "inspect the current workspace" ---------------------------

    def test_inspect_current_workspace(self, pb, ws_tree):
        result = pb.process("inspect the current workspace")
        assert result.success, f"Failed: {result.error} / {result.metadata}"
        assert result.result["target_type"] == "directory"
        assert Path(result.result["path"]).resolve() == ws_tree.resolve()

    def test_inspect_this_directory(self, pb, ws_tree):
        result = pb.process("inspect this directory")
        assert result.success
        assert result.result["target_type"] == "directory"

    # ---- 12. Absolute file path ----------------------------------------

    def test_inspect_absolute_file(self, pb, ws_tree):
        abs_path = str((ws_tree / "sample.txt").resolve())
        result = pb.process(f"inspect {abs_path}")
        assert result.success, f"Failed: {result.error} / {result.metadata}"
        assert result.result["target_type"] == "file"
        assert result.result["name"] == "sample.txt"

    # ---- 13. Absolute directory path -----------------------------------

    def test_inspect_absolute_directory(self, pb, ws_tree):
        abs_path = str(ws_tree.resolve())
        result = pb.process(f"inspect {abs_path}")
        assert result.success, f"Failed: {result.error} / {result.metadata}"
        assert result.result["target_type"] == "directory"

    # ---- 14. Nested relative path --------------------------------------

    def test_inspect_nested_relative_file(self, pb, ws_tree):
        result = pb.process("inspect subdir/nested.txt")
        assert result.success, f"Failed: {result.error} / {result.metadata}"
        assert result.result["target_type"] == "file"
        assert result.result["name"] == "nested.txt"

    def test_inspect_nested_relative_dir(self, pb, ws_tree):
        result = pb.process("inspect subdir")
        assert result.success
        assert result.result["target_type"] == "directory"

    # ---- Phrasing variants ---------------------------------------------

    def test_show_information_about(self, pb, ws_tree):
        result = pb.process("show information about sample.txt")
        assert result.success
        assert result.result["target_type"] == "file"

    def test_show_info_about(self, pb, ws_tree):
        result = pb.process("show info about sample.txt")
        assert result.success
        assert result.result["target_type"] == "file"


# ===========================================================================
# 15. PreprocessorBrain — not-found error
# ===========================================================================

class TestInspectNotFound:

    def test_nonexistent_absolute_returns_error(self, pb, ws_tree):
        result = pb.process("inspect D:/this_path_definitely_does_not_exist_xyzzy_99999")
        assert not result.success
        assert result.domain == Domain.DIRECTORY

    def test_nonexistent_relative_returns_error(self, pb, ws_tree):
        result = pb.process("inspect does_not_exist.py")
        assert not result.success
        assert result.error in (DirErrorCode.NOT_FOUND, DirErrorCode.OPERATION_FAILED)

    def test_error_result_has_no_result_payload(self, pb, ws_tree):
        result = pb.process("inspect totally_nonexistent_xyzzy.txt")
        assert not result.success
        assert result.result is None


# ===========================================================================
# 16. Read-only guarantee — filesystem not modified
# ===========================================================================

class TestInspectReadOnly:
    """INSPECT must never create, modify, or delete anything."""

    def test_inspect_does_not_create_files(self, pb, ws_tree):
        before = set(ws_tree.rglob("*"))
        pb.process("inspect sample.txt")
        after = set(ws_tree.rglob("*"))
        assert before == after, f"New paths appeared after inspect: {after - before}"

    def test_inspect_does_not_modify_file_content(self, pb, ws_tree):
        target = ws_tree / "sample.txt"
        original_content = target.read_text(encoding="utf-8")
        pb.process("inspect sample.txt")
        assert target.read_text(encoding="utf-8") == original_content

    def test_inspect_does_not_modify_file_mtime(self, pb, ws_tree):
        target = ws_tree / "sample.txt"
        original_mtime = target.stat().st_mtime
        pb.process("inspect sample.txt")
        assert target.stat().st_mtime == original_mtime

    def test_inspect_directory_does_not_create_files(self, pb, ws_tree):
        before = set(ws_tree.rglob("*"))
        pb.process("inspect backup")
        after = set(ws_tree.rglob("*"))
        assert before == after

    def test_inspect_dot_does_not_create_files(self, pb, ws_tree):
        before = set(ws_tree.rglob("*"))
        pb.process("inspect .")
        after = set(ws_tree.rglob("*"))
        assert before == after

    def test_inspect_nonexistent_does_not_create_files(self, pb, ws_tree):
        before = set(ws_tree.rglob("*"))
        pb.process("inspect completely_nonexistent_file.txt")
        after = set(ws_tree.rglob("*"))
        assert before == after


# ===========================================================================
# 17. Read-only guarantee — navigation state not modified
# ===========================================================================

class TestInspectDoesNotChangeState:

    def test_inspect_does_not_change_active_directory(self, pb, ws_tree):
        """INSPECT must not alter the current active workspace."""
        initial_dir = pb.process("what is my current directory?")
        assert initial_dir.success
        original_path = initial_dir.result

        pb.process("inspect backup")

        after_dir = pb.process("what is my current directory?")
        assert after_dir.success
        assert after_dir.result == original_path, (
            f"Active directory changed after inspect! "
            f"Was {original_path!r}, now {after_dir.result!r}"
        )

    def test_inspect_dot_does_not_change_active_directory(self, pb, ws_tree):
        initial = pb.process("what is my current directory?")
        pb.process("inspect .")
        after = pb.process("what is my current directory?")
        assert after.result == initial.result

    def test_multiple_inspects_do_not_change_active_directory(self, pb, ws_tree):
        initial = pb.process("what is my current directory?")
        pb.process("inspect sample.txt")
        pb.process("inspect backup")
        pb.process("inspect .")
        after = pb.process("what is my current directory?")
        assert after.result == initial.result


# ===========================================================================
# 18. Structured result contract
# ===========================================================================

class TestInspectStructuredResult:

    def test_file_result_has_operation_inspect(self, pb, ws_tree):
        result = pb.process("inspect sample.txt")
        assert result.success
        assert result.result["operation"] == "INSPECT"

    def test_dir_result_has_operation_inspect(self, pb, ws_tree):
        result = pb.process("inspect backup")
        assert result.success
        assert result.result["operation"] == "INSPECT"

    def test_result_is_dict(self, pb, ws_tree):
        result = pb.process("inspect sample.txt")
        assert isinstance(result.result, dict)

    def test_requires_llm_is_false(self, pb, ws_tree):
        result = pb.process("inspect sample.txt")
        assert result.requires_llm is False

    def test_domain_is_directory(self, pb, ws_tree):
        result = pb.process("inspect sample.txt")
        assert result.domain == Domain.DIRECTORY

    def test_intent_is_inspect(self, pb, ws_tree):
        result = pb.process("inspect sample.txt")
        assert result.intent == Intent.INSPECT

    def test_operation_field_is_inspect(self, pb, ws_tree):
        result = pb.process("inspect sample.txt")
        assert result.operation == "INSPECT"

    def test_target_field_is_absolute_path(self, pb, ws_tree):
        result = pb.process("inspect sample.txt")
        assert result.success
        # result.target should be the resolved absolute path
        assert Path(result.target).is_absolute()

    def test_file_result_all_required_keys(self, pb, ws_tree):
        result = pb.process("inspect sample.txt")
        assert result.success
        for key in ("operation", "target_type", "name", "path",
                    "extension", "size", "modified", "readable", "writable"):
            assert key in result.result, f"Missing key: {key}"

    def test_directory_result_all_required_keys(self, pb, ws_tree):
        result = pb.process("inspect backup")
        assert result.success
        for key in ("operation", "target_type", "name", "path",
                    "file_count", "directory_count", "modified"):
            assert key in result.result, f"Missing key: {key}"


# ===========================================================================
# 19. INSPECT vs READ distinction
# ===========================================================================

class TestInspectVsRead:

    def test_inspect_does_not_return_file_content(self, pb, ws_tree):
        result = pb.process("inspect sample.txt")
        assert result.success
        # result.result is a metadata dict, not a content string
        assert not isinstance(result.result, str), (
            "INSPECT returned raw file content — should return metadata dict"
        )
        # Specifically, the file content "hello inspect" must not be the result
        assert result.result != "hello inspect"
        assert result.result.get("target_type") == "file"

    def test_read_returns_content_not_metadata(self, pb, ws_tree):
        result = pb.process("read sample.txt")
        assert result.success
        assert isinstance(result.result, str)
        assert "hello inspect" in result.result

    def test_inspect_and_read_different_intents(self, pb, ws_tree):
        from preprocessor_brain import IntentDetector
        d = IntentDetector()
        assert d.detect("inspect sample.txt").intent == Intent.INSPECT
        assert d.detect("read sample.txt").intent == Intent.READ_FILE


# ===========================================================================
# 20. INSPECT vs NAVIGATE distinction
# ===========================================================================

class TestInspectVsNavigate:

    def test_inspect_does_not_navigate(self, pb, ws_tree):
        initial_dir_result = pb.process("what is my current directory?")
        original_dir = initial_dir_result.result

        pb.process("inspect backup")

        # Active directory must not have changed to backup/
        after = pb.process("what is my current directory?")
        assert Path(after.result).resolve() == Path(original_dir).resolve(), (
            "inspect backup changed the active directory — it must not"
        )

    def test_navigate_intent_not_inspect(self):
        d = IntentDetector()
        assert d.detect("go to backup").intent == Intent.NAVIGATE
        assert d.detect("navigate to backup").intent == Intent.NAVIGATE


# ===========================================================================
# 21–25. Regression tests
# ===========================================================================

class TestInspectRegression:

    def test_workspace_setting_still_works(self, ws_tree):
        """set workspace to <path> must still work after INSPECT is added."""
        brain = PreprocessorBrain()
        result = brain.process(f"set workspace to {ws_tree}")
        assert result.success
        assert result.intent == Intent.SWITCH_DIRECTORY

    def test_navigation_still_works(self, pb, ws_tree):
        """go to <dir> must still navigate correctly."""
        result = pb.process("go to backup")
        assert result.success
        assert result.intent == Intent.NAVIGATE
        assert Path(result.result).resolve() == (ws_tree / "backup").resolve()

    def test_read_still_works(self, pb, ws_tree):
        """read <file> must still return file content."""
        result = pb.process("read sample.txt")
        assert result.success
        assert result.intent == Intent.READ_FILE
        assert "hello inspect" in result.result

    def test_self_info_still_works(self, pb):
        """What are your capabilities? must still route to self-info."""
        result = pb.process("What are your capabilities?")
        assert result.success
        assert result.domain == Domain.SELF_INFO

    def test_list_directory_still_works(self, pb, ws_tree):
        """list the current workspace must still list directory contents."""
        result = pb.process("list the current workspace")
        assert result.success
        assert result.intent == Intent.LIST_DIRECTORY
        names = {e["name"] for e in result.result}
        assert "sample.txt" in names
        assert "backup" in names

    def test_create_file_still_works(self, pb, ws_tree):
        """create a file called X must still create files."""
        result = pb.process("create a file called regression_check.txt")
        assert result.success
        assert result.intent == Intent.CREATE_FILE
        assert (ws_tree / "regression_check.txt").exists()

    def test_current_directory_still_works(self, pb, ws_tree):
        """what is my current directory? must still return workspace path."""
        result = pb.process("what is my current directory?")
        assert result.success
        assert result.intent == Intent.CURRENT_DIRECTORY
        assert Path(result.result).resolve() == ws_tree.resolve()

    def test_inspect_does_not_affect_subsequent_read(self, pb, ws_tree):
        """A successful inspect must not corrupt state for a subsequent read."""
        inspect_result = pb.process("inspect sample.txt")
        assert inspect_result.success

        read_result = pb.process("read sample.txt")
        assert read_result.success
        assert "hello inspect" in read_result.result

    def test_inspect_does_not_affect_subsequent_navigate(self, pb, ws_tree):
        """A successful inspect must not corrupt state for a subsequent navigate."""
        pb.process("inspect backup")
        nav_result = pb.process("go to backup")
        assert nav_result.success

    def test_inspect_does_not_affect_subsequent_list(self, pb, ws_tree):
        """A successful inspect must not corrupt state for a subsequent list."""
        pb.process("inspect backup")
        list_result = pb.process("list the current workspace")
        assert list_result.success
        names = {e["name"] for e in list_result.result}
        assert "backup" in names


# ===========================================================================
# 26. Standalone DirectoryOperations fallback — same contracts
# ===========================================================================

class TestInspectStandaloneOps:
    """
    Verify that the standalone DirectoryOperations fallback (used when no
    real DirectoryControl is supplied) satisfies the same inspection
    contracts as the real DirectoryControl.
    """

    def test_standalone_file_inspect(self, pb_standalone, ws_tree):
        result = pb_standalone.process("inspect sample.txt")
        assert result.success
        assert result.result["target_type"] == "file"
        assert result.result["name"] == "sample.txt"

    def test_standalone_dir_inspect(self, pb_standalone, ws_tree):
        result = pb_standalone.process("inspect backup")
        assert result.success
        assert result.result["target_type"] == "directory"

    def test_standalone_dot_inspect(self, pb_standalone, ws_tree):
        result = pb_standalone.process("inspect .")
        assert result.success
        assert result.result["target_type"] == "directory"

    def test_standalone_not_found(self, pb_standalone, ws_tree):
        result = pb_standalone.process("inspect does_not_exist.txt")
        assert not result.success
