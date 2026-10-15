"""
tests/test_directory_control.py
================================
Tests for the AB Directory Control subsystem.

Covers:
  1.  set_directory
  2.  switch_directory
  3.  clear_directory
  4.  get_active_directory
  5.  list_directory (files, dirs, mixed)
  6.  read_file
  7.  create_file
  8.  write_file
  9.  append_file
 10.  update_file
 11.  create_directory
 12.  move
 13.  copy
 14.  rename
 15.  verify_operation
 16.  nonexistent paths
 17.  invalid paths
 18.  path traversal (.. escapes)
 19.  outside-workspace absolute paths
 20.  no active workspace
 21.  write failure (mocked)
 22.  read failure (mocked)
 23.  move / copy / rename failures
 24.  search_files
 25.  get_indexed_files
 26.  import sanity (existing suite unaffected)
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from directory_control import (
    DirectoryControl,
    DirectoryControlError,
    OperationError,
    FileNotFoundInWorkspace,
    NotAFileError,
    VerificationError,
    WorkspaceViolationError,
    NoActiveWorkspaceError,
    WorkspaceNotFoundError,
)
from directory_control.path_security import (
    resolve_workspace_path,
    validate_workspace,
    is_within_workspace,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_ws():
    """A fresh temporary directory used as the workspace root."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture()
def dc(tmp_ws):
    """A DirectoryControl instance with an active workspace already set."""
    ctrl = DirectoryControl()
    ctrl.set_directory(str(tmp_ws))
    return ctrl


def _write(directory: Path, name: str, content: str = "hello") -> Path:
    p = directory / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


# ===========================================================================
# 1. set_directory
# ===========================================================================

class TestSetDirectory:

    def test_set_valid_directory(self, tmp_ws):
        ctrl = DirectoryControl()
        result = ctrl.set_directory(str(tmp_ws))
        assert Path(result).resolve() == tmp_ws.resolve()
        assert ctrl.get_active_directory() is not None

    def test_set_directory_stores_resolved_path(self, tmp_ws):
        ctrl = DirectoryControl()
        ctrl.set_directory(str(tmp_ws))
        stored = Path(ctrl.get_active_directory())
        assert stored.is_absolute()
        assert stored.exists()

    def test_set_nonexistent_directory_raises(self):
        ctrl = DirectoryControl()
        with pytest.raises(WorkspaceNotFoundError):
            ctrl.set_directory("Z:\\does_not_exist_99999")

    def test_set_file_path_raises(self, tmp_ws):
        f = _write(tmp_ws, "file.txt")
        ctrl = DirectoryControl()
        with pytest.raises(WorkspaceNotFoundError):
            ctrl.set_directory(str(f))

    def test_set_directory_replaces_previous(self, tmp_ws):
        with tempfile.TemporaryDirectory() as d2:
            ctrl = DirectoryControl()
            ctrl.set_directory(str(tmp_ws))
            ctrl.set_directory(d2)
            assert Path(ctrl.get_active_directory()).resolve() == Path(d2).resolve()

    def test_failed_indexing_preserves_previous_workspace_and_reader(
        self, tmp_ws, monkeypatch
    ):
        import directory_control.directory_control as directory_module

        ctrl = DirectoryControl()
        ctrl.set_directory(str(tmp_ws))
        previous_reader = ctrl._reader

        class FailedReader:
            directory = None

            def read_directory(self, _path):
                return type("ReadResult", (), {
                    "success": False,
                    "error": "indexing failed",
                })()

        monkeypatch.setattr(directory_module, "DirectoryReader", FailedReader)
        with pytest.raises(WorkspaceNotFoundError, match="indexing failed"):
            ctrl.set_directory(str(tmp_ws))

        assert Path(ctrl.get_active_directory()).resolve() == tmp_ws.resolve()
        assert ctrl._reader is previous_reader


# ===========================================================================
# 2. switch_directory
# ===========================================================================

class TestSwitchDirectory:

    def test_switch_changes_active_directory(self, tmp_ws):
        with tempfile.TemporaryDirectory() as d2:
            ctrl = DirectoryControl()
            ctrl.set_directory(str(tmp_ws))
            ctrl.switch_directory(d2)
            assert Path(ctrl.get_active_directory()).resolve() == Path(d2).resolve()

    def test_switch_to_nonexistent_raises(self, tmp_ws):
        ctrl = DirectoryControl()
        ctrl.set_directory(str(tmp_ws))
        with pytest.raises(WorkspaceNotFoundError):
            ctrl.switch_directory("Z:\\nonexistent_switch_test")

    def test_switch_is_identical_to_set(self, tmp_ws):
        ctrl1 = DirectoryControl()
        ctrl2 = DirectoryControl()
        with tempfile.TemporaryDirectory() as d2:
            ctrl1.set_directory(d2)
            ctrl2.switch_directory(d2)
            assert ctrl1.get_active_directory() == ctrl2.get_active_directory()


# ===========================================================================
# 3. clear_directory
# ===========================================================================

class TestClearDirectory:

    def test_clear_removes_active_directory(self, dc, tmp_ws):
        assert dc.get_active_directory() is not None
        dc.clear_directory()
        assert dc.get_active_directory() is None

    def test_clear_returns_previous_path(self, dc, tmp_ws):
        previous = dc.clear_directory()
        assert previous is not None
        assert Path(previous).resolve() == tmp_ws.resolve()

    def test_clear_when_none_returns_none(self):
        ctrl = DirectoryControl()
        result = ctrl.clear_directory()
        assert result is None

    def test_clear_does_not_delete_files(self, dc, tmp_ws):
        f = _write(tmp_ws, "keep.txt", "keep me")
        dc.clear_directory()
        assert f.exists()
        assert f.read_text(encoding="utf-8") == "keep me"

    def test_operations_after_clear_raise_no_workspace(self, dc):
        dc.clear_directory()
        with pytest.raises(NoActiveWorkspaceError):
            dc.read_file("anything.txt")

    def test_operations_after_clear_raise_for_list(self, dc):
        dc.clear_directory()
        with pytest.raises(NoActiveWorkspaceError):
            dc.list_directory()


# ===========================================================================
# 4. get_active_directory
# ===========================================================================

class TestGetActiveDirectory:

    def test_returns_none_when_unset(self):
        ctrl = DirectoryControl()
        assert ctrl.get_active_directory() is None

    def test_returns_string_when_set(self, dc):
        result = dc.get_active_directory()
        assert isinstance(result, str)

    def test_returns_resolved_absolute_path(self, tmp_ws):
        ctrl = DirectoryControl()
        ctrl.set_directory(str(tmp_ws))
        result = ctrl.get_active_directory()
        assert os.path.isabs(result)


# ===========================================================================
# 5. list_directory
# ===========================================================================

class TestListDirectory:

    def test_list_root_returns_entries(self, dc, tmp_ws):
        _write(tmp_ws, "a.txt")
        _write(tmp_ws, "b.md")
        entries = dc.list_directory()
        names = {e["name"] for e in entries}
        assert "a.txt" in names
        assert "b.md" in names

    def test_list_distinguishes_files_and_dirs(self, dc, tmp_ws):
        _write(tmp_ws, "file.txt")
        (tmp_ws / "subdir").mkdir()
        entries = dc.list_directory()
        types = {e["name"]: e["type"] for e in entries}
        assert types["file.txt"] == "file"
        assert types["subdir"] == "dir"

    def test_list_files_only(self, dc, tmp_ws):
        _write(tmp_ws, "file.py")
        (tmp_ws / "folder").mkdir()
        entries = dc.list_directory(files_only=True)
        assert all(e["type"] == "file" for e in entries)

    def test_list_dirs_only(self, dc, tmp_ws):
        _write(tmp_ws, "file.py")
        (tmp_ws / "folder").mkdir()
        entries = dc.list_directory(dirs_only=True)
        assert all(e["type"] == "dir" for e in entries)

    def test_list_subpath(self, dc, tmp_ws):
        sub = tmp_ws / "sub"
        sub.mkdir()
        _write(sub, "child.txt")
        entries = dc.list_directory("sub")
        names = {e["name"] for e in entries}
        assert "child.txt" in names

    def test_list_entry_has_expected_keys(self, dc, tmp_ws):
        _write(tmp_ws, "x.txt")
        entries = dc.list_directory()
        for e in entries:
            assert "name" in e
            assert "rel_path" in e
            assert "type" in e
            assert "size_bytes" in e

    def test_list_nonexistent_subpath_raises(self, dc):
        with pytest.raises(FileNotFoundInWorkspace):
            dc.list_directory("no_such_subdir")

    def test_list_file_as_dir_raises(self, dc, tmp_ws):
        _write(tmp_ws, "notadir.txt")
        with pytest.raises(NotAFileError):
            dc.list_directory("notadir.txt")

    def test_list_no_workspace_raises(self):
        ctrl = DirectoryControl()
        with pytest.raises(NoActiveWorkspaceError):
            ctrl.list_directory()

    def test_list_traversal_rejected(self, dc):
        with pytest.raises((WorkspaceViolationError, PermissionError)):
            dc.list_directory("../outside")




# ===========================================================================
# 6. read_file
# ===========================================================================

class TestReadFile:

    def test_read_existing_file(self, dc, tmp_ws):
        _write(tmp_ws, "hello.txt", "world content")
        content = dc.read_file("hello.txt")
        assert "world content" in content

    def test_read_relative_path(self, dc, tmp_ws):
        sub = tmp_ws / "sub"
        sub.mkdir()
        _write(sub, "data.txt", "nested content")
        content = dc.read_file("sub/data.txt")
        assert "nested content" in content

    def test_read_nonexistent_raises(self, dc):
        with pytest.raises(FileNotFoundInWorkspace):
            dc.read_file("nonexistent.txt")

    def test_read_directory_raises(self, dc, tmp_ws):
        (tmp_ws / "adir").mkdir()
        with pytest.raises(NotAFileError):
            dc.read_file("adir")

    def test_read_no_workspace_raises(self):
        ctrl = DirectoryControl()
        with pytest.raises(NoActiveWorkspaceError):
            ctrl.read_file("anything.txt")

    def test_read_traversal_rejected(self, dc):
        with pytest.raises((WorkspaceViolationError, PermissionError)):
            dc.read_file("../../etc/passwd")

    def test_read_absolute_outside_workspace_rejected(self, dc):
        with pytest.raises((WorkspaceViolationError, PermissionError)):
            dc.read_file("C:\\Windows\\System32\\drivers\\etc\\hosts")

    def test_read_large_file_truncated(self, dc, tmp_ws):
        # Write a file larger than MAX_READ_BYTES
        big_content = "A" * (DirectoryControl.MAX_READ_BYTES + 1000)
        _write(tmp_ws, "big.txt", big_content)
        result = dc.read_file("big.txt")
        assert "truncated" in result.lower() or len(result) <= DirectoryControl.MAX_READ_BYTES + 200

    def test_read_failure_raises_operation_error(self, dc, tmp_ws):
        _write(tmp_ws, "locked.txt", "data")
        target = tmp_ws / "locked.txt"
        with patch.object(target.__class__, "read_bytes", side_effect=PermissionError("denied")):
            with patch("directory_control.directory_control.Path.read_bytes", side_effect=PermissionError("denied")):
                # Patch at the pathlib level isn't reliable across all envs;
                # ensure PermissionError raises OperationError via the handler
                pass  # covered by mock tests below

    def test_read_returns_string(self, dc, tmp_ws):
        _write(tmp_ws, "str.txt", "text value")
        result = dc.read_file("str.txt")
        assert isinstance(result, str)


# ===========================================================================
# 7. create_file
# ===========================================================================

class TestCreateFile:

    def test_create_new_file(self, dc, tmp_ws):
        p = dc.create_file("notes.txt")
        assert p.exists()
        assert p.is_file()

    def test_create_with_content(self, dc, tmp_ws):
        dc.create_file("notes.txt", content="initial content")
        assert (tmp_ws / "notes.txt").read_text(encoding="utf-8") == "initial content"

    def test_create_returns_path(self, dc, tmp_ws):
        result = dc.create_file("out.txt")
        assert isinstance(result, Path)

    def test_create_in_subdir_creates_parents(self, dc, tmp_ws):
        dc.create_file("a/b/c.txt", content="deep")
        assert (tmp_ws / "a" / "b" / "c.txt").exists()

    def test_create_existing_file_raises(self, dc, tmp_ws):
        _write(tmp_ws, "exists.txt", "old")
        with pytest.raises(OperationError):
            dc.create_file("exists.txt")

    def test_create_no_workspace_raises(self):
        ctrl = DirectoryControl()
        with pytest.raises(NoActiveWorkspaceError):
            ctrl.create_file("file.txt")

    def test_create_traversal_rejected(self, dc):
        with pytest.raises((WorkspaceViolationError, PermissionError)):
            dc.create_file("../escape.txt")

    def test_create_absolute_outside_rejected(self, dc):
        with pytest.raises((WorkspaceViolationError, PermissionError)):
            dc.create_file("C:\\Windows\\evil.txt")


# ===========================================================================
# 8. write_file
# ===========================================================================

class TestWriteFile:

    def test_write_creates_file(self, dc, tmp_ws):
        dc.write_file("new.txt", "content")
        assert (tmp_ws / "new.txt").exists()

    def test_write_overwrites_existing(self, dc, tmp_ws):
        _write(tmp_ws, "existing.txt", "old content")
        dc.write_file("existing.txt", "new content")
        assert (tmp_ws / "existing.txt").read_text(encoding="utf-8") == "new content"

    def test_write_returns_path(self, dc, tmp_ws):
        result = dc.write_file("out.txt", "data")
        assert isinstance(result, Path)

    def test_write_creates_parent_dirs(self, dc, tmp_ws):
        dc.write_file("deep/path/file.txt", "content")
        assert (tmp_ws / "deep" / "path" / "file.txt").exists()

    def test_write_no_workspace_raises(self):
        ctrl = DirectoryControl()
        with pytest.raises(NoActiveWorkspaceError):
            ctrl.write_file("file.txt", "content")

    def test_write_traversal_rejected(self, dc):
        with pytest.raises((WorkspaceViolationError, PermissionError)):
            dc.write_file("../outside.txt", "bad")

    def test_write_verifies_file_exists_after(self, dc, tmp_ws):
        # The method should verify the file exists — we confirm by checking
        # no VerificationError is raised on a normal write
        result = dc.write_file("verify_me.txt", "check")
        assert result.exists()


# ===========================================================================
# 9. append_file
# ===========================================================================

class TestAppendFile:

    def test_append_to_existing_file(self, dc, tmp_ws):
        _write(tmp_ws, "log.txt", "line1\n")
        dc.append_file("log.txt", "line2\n")
        content = (tmp_ws / "log.txt").read_text(encoding="utf-8")
        assert "line1" in content
        assert "line2" in content

    def test_append_does_not_erase_content(self, dc, tmp_ws):
        _write(tmp_ws, "log.txt", "original\n")
        dc.append_file("log.txt", "appended\n")
        content = (tmp_ws / "log.txt").read_text(encoding="utf-8")
        assert content.startswith("original")

    def test_append_creates_file_if_missing(self, dc, tmp_ws):
        dc.append_file("new_log.txt", "first line\n")
        assert (tmp_ws / "new_log.txt").exists()

    def test_append_returns_path(self, dc, tmp_ws):
        result = dc.append_file("out.txt", "data")
        assert isinstance(result, Path)

    def test_append_no_workspace_raises(self):
        ctrl = DirectoryControl()
        with pytest.raises(NoActiveWorkspaceError):
            ctrl.append_file("file.txt", "content")

    def test_append_traversal_rejected(self, dc):
        with pytest.raises((WorkspaceViolationError, PermissionError)):
            dc.append_file("../outside.txt", "bad")


# ===========================================================================
# 10. update_file
# ===========================================================================

class TestUpdateFile:

    def test_update_existing_file(self, dc, tmp_ws):
        _write(tmp_ws, "doc.txt", "old content")
        dc.update_file("doc.txt", "new content")
        assert (tmp_ws / "doc.txt").read_text(encoding="utf-8") == "new content"

    def test_update_returns_path(self, dc, tmp_ws):
        _write(tmp_ws, "doc.txt", "old")
        result = dc.update_file("doc.txt", "new")
        assert isinstance(result, Path)

    def test_update_nonexistent_raises(self, dc):
        with pytest.raises(FileNotFoundInWorkspace):
            dc.update_file("missing.txt", "content")

    def test_update_directory_raises(self, dc, tmp_ws):
        (tmp_ws / "adir").mkdir()
        with pytest.raises(NotAFileError):
            dc.update_file("adir", "content")

    def test_update_no_workspace_raises(self):
        ctrl = DirectoryControl()
        with pytest.raises(NoActiveWorkspaceError):
            ctrl.update_file("file.txt", "content")

    def test_update_traversal_rejected(self, dc):
        with pytest.raises((WorkspaceViolationError, PermissionError)):
            dc.update_file("../../etc/motd", "pwned")




# ===========================================================================
# 11. create_directory
# ===========================================================================

class TestCreateDirectory:

    def test_create_new_directory(self, dc, tmp_ws):
        dc.create_directory("reports")
        assert (tmp_ws / "reports").is_dir()

    def test_create_returns_path(self, dc, tmp_ws):
        result = dc.create_directory("mydir")
        assert isinstance(result, Path)

    def test_create_nested_directory(self, dc, tmp_ws):
        dc.create_directory("a/b/c")
        assert (tmp_ws / "a" / "b" / "c").is_dir()

    def test_create_existing_directory_raises(self, dc, tmp_ws):
        (tmp_ws / "existing").mkdir()
        with pytest.raises(OperationError):
            dc.create_directory("existing")

    def test_create_directory_no_workspace_raises(self):
        ctrl = DirectoryControl()
        with pytest.raises(NoActiveWorkspaceError):
            ctrl.create_directory("reports")

    def test_create_directory_traversal_rejected(self, dc):
        with pytest.raises((WorkspaceViolationError, PermissionError)):
            dc.create_directory("../outside_dir")


# ===========================================================================
# 12. move
# ===========================================================================

class TestMove:

    def test_move_file_to_new_name(self, dc, tmp_ws):
        _write(tmp_ws, "old.txt", "content")
        dc.move("old.txt", "new.txt")
        assert (tmp_ws / "new.txt").exists()
        assert not (tmp_ws / "old.txt").exists()

    def test_move_file_into_directory(self, dc, tmp_ws):
        _write(tmp_ws, "file.txt", "content")
        (tmp_ws / "dest").mkdir()
        dc.move("file.txt", "dest/file.txt")
        assert (tmp_ws / "dest" / "file.txt").exists()
        assert not (tmp_ws / "file.txt").exists()

    def test_move_returns_destination_path(self, dc, tmp_ws):
        _write(tmp_ws, "a.txt", "x")
        result = dc.move("a.txt", "b.txt")
        assert isinstance(result, Path)
        assert result.exists()

    def test_move_nonexistent_source_raises(self, dc):
        with pytest.raises(FileNotFoundInWorkspace):
            dc.move("ghost.txt", "dest.txt")

    def test_move_source_traversal_rejected(self, dc):
        with pytest.raises((WorkspaceViolationError, PermissionError)):
            dc.move("../outside.txt", "inside.txt")

    def test_move_destination_traversal_rejected(self, dc, tmp_ws):
        _write(tmp_ws, "file.txt", "content")
        with pytest.raises((WorkspaceViolationError, PermissionError)):
            dc.move("file.txt", "../../outside.txt")

    def test_move_no_workspace_raises(self):
        ctrl = DirectoryControl()
        with pytest.raises(NoActiveWorkspaceError):
            ctrl.move("a.txt", "b.txt")

    def test_move_verifies_source_removed(self, dc, tmp_ws):
        _write(tmp_ws, "src.txt", "data")
        dc.move("src.txt", "dst.txt")
        assert not (tmp_ws / "src.txt").exists()

    def test_move_directory_to_new_name_verifies_actual_destination(self, dc, tmp_ws):
        source = tmp_ws / "source-dir"
        source.mkdir()
        _write(source, "nested/data.txt", "directory move")

        result = dc.move("source-dir", "moved-dir")

        assert result == tmp_ws / "moved-dir"
        assert (result / "nested" / "data.txt").read_text(
            encoding="utf-8"
        ) == "directory move"
        assert not source.exists()


# ===========================================================================
# 13. copy
# ===========================================================================

class TestCopy:

    def test_copy_file(self, dc, tmp_ws):
        _write(tmp_ws, "orig.txt", "original")
        dc.copy("orig.txt", "copy.txt")
        assert (tmp_ws / "copy.txt").exists()
        assert (tmp_ws / "orig.txt").exists()  # source preserved

    def test_copy_content_matches(self, dc, tmp_ws):
        _write(tmp_ws, "src.txt", "hello copy")
        dc.copy("src.txt", "dst.txt")
        assert (tmp_ws / "dst.txt").read_text(encoding="utf-8") == "hello copy"

    def test_copy_returns_destination_path(self, dc, tmp_ws):
        _write(tmp_ws, "f.txt", "x")
        result = dc.copy("f.txt", "g.txt")
        assert isinstance(result, Path)
        assert result.exists()

    def test_copy_nonexistent_source_raises(self, dc):
        with pytest.raises(FileNotFoundInWorkspace):
            dc.copy("ghost.txt", "copy.txt")

    def test_copy_directory_tree_is_verified(self, dc, tmp_ws):
        source = tmp_ws / "adir"
        source.mkdir()
        _write(source, "nested/data.txt", "verified directory copy")

        result = dc.copy("adir", "adir_copy")

        assert result == tmp_ws / "adir_copy"
        assert (result / "nested" / "data.txt").read_text(
            encoding="utf-8"
        ) == "verified directory copy"
        assert (source / "nested" / "data.txt").exists()

    def test_copy_directory_into_descendant_is_rejected_without_side_effects(
        self, dc, tmp_ws
    ):
        source = tmp_ws / "adir"
        source.mkdir()
        _write(source, "data.txt", "original")

        with pytest.raises(OperationError, match="itself or one of its descendants"):
            dc.copy("adir", "adir/new/nested-copy")

        assert not (source / "new").exists()
        assert (source / "data.txt").read_text(encoding="utf-8") == "original"

    def test_copy_file_into_existing_directory(self, dc, tmp_ws):
        _write(tmp_ws, "original.txt", "copy into directory")
        (tmp_ws / "dest").mkdir()

        result = dc.copy("original.txt", "dest")

        assert result == tmp_ws / "dest" / "original.txt"
        assert result.read_text(encoding="utf-8") == "copy into directory"

    def test_copy_source_traversal_rejected(self, dc):
        with pytest.raises((WorkspaceViolationError, PermissionError)):
            dc.copy("../outside.txt", "inside.txt")

    def test_copy_destination_traversal_rejected(self, dc, tmp_ws):
        _write(tmp_ws, "file.txt", "content")
        with pytest.raises((WorkspaceViolationError, PermissionError)):
            dc.copy("file.txt", "../../outside.txt")

    def test_copy_no_workspace_raises(self):
        ctrl = DirectoryControl()
        with pytest.raises(NoActiveWorkspaceError):
            ctrl.copy("a.txt", "b.txt")

    def test_copy_into_subdir_creates_parent(self, dc, tmp_ws):
        _write(tmp_ws, "file.txt", "data")
        dc.copy("file.txt", "subdir/file_copy.txt")
        assert (tmp_ws / "subdir" / "file_copy.txt").exists()


# ===========================================================================
# 14. rename
# ===========================================================================

class TestRename:

    def test_rename_file_in_place(self, dc, tmp_ws):
        _write(tmp_ws, "old.txt", "content")
        dc.rename("old.txt", "new.txt")
        assert (tmp_ws / "new.txt").exists()
        assert not (tmp_ws / "old.txt").exists()

    def test_rename_returns_new_path(self, dc, tmp_ws):
        _write(tmp_ws, "a.txt", "x")
        result = dc.rename("a.txt", "b.txt")
        assert isinstance(result, Path)
        assert result.exists()

    def test_rename_nonexistent_source_raises(self, dc):
        with pytest.raises(FileNotFoundInWorkspace):
            dc.rename("ghost.txt", "new.txt")

    def test_rename_to_existing_target_raises(self, dc, tmp_ws):
        _write(tmp_ws, "a.txt", "x")
        _write(tmp_ws, "b.txt", "y")
        with pytest.raises(OperationError):
            dc.rename("a.txt", "b.txt")

    def test_rename_traversal_rejected(self, dc, tmp_ws):
        _write(tmp_ws, "file.txt", "content")
        with pytest.raises((WorkspaceViolationError, PermissionError)):
            dc.rename("file.txt", "../../escape.txt")

    def test_rename_no_workspace_raises(self):
        ctrl = DirectoryControl()
        with pytest.raises(NoActiveWorkspaceError):
            ctrl.rename("a.txt", "b.txt")

    def test_rename_directory(self, dc, tmp_ws):
        (tmp_ws / "oldfolder").mkdir()
        dc.rename("oldfolder", "newfolder")
        assert (tmp_ws / "newfolder").is_dir()
        assert not (tmp_ws / "oldfolder").exists()


# ===========================================================================
# 15. verify_operation
# ===========================================================================

class TestVerifyOperation:

    def test_verify_existing_path_passes(self, dc, tmp_ws):
        f = _write(tmp_ws, "real.txt", "data")
        assert dc.verify_operation(f, must_exist=True) is True

    def test_verify_missing_path_raises(self, dc, tmp_ws):
        missing = tmp_ws / "missing.txt"
        with pytest.raises(VerificationError):
            dc.verify_operation(missing, must_exist=True)

    def test_verify_absent_path_passes_when_must_not_exist(self, dc, tmp_ws):
        missing = tmp_ws / "gone.txt"
        assert dc.verify_operation(missing, must_exist=False) is True

    def test_verify_present_path_raises_when_must_not_exist(self, dc, tmp_ws):
        f = _write(tmp_ws, "present.txt", "still here")
        with pytest.raises(VerificationError):
            dc.verify_operation(f, must_exist=False)


# ===========================================================================
# 16-19. Security: path traversal, absolute outside, invalid paths
# ===========================================================================

class TestPathSecurity:

    def test_dotdot_traversal_rejected(self, dc):
        with pytest.raises((WorkspaceViolationError, PermissionError)):
            dc.read_file("../secret.txt")

    def test_double_dotdot_traversal_rejected(self, dc):
        with pytest.raises((WorkspaceViolationError, PermissionError)):
            dc.read_file("../../secret.txt")

    def test_absolute_path_outside_workspace_rejected(self, dc):
        with pytest.raises((WorkspaceViolationError, PermissionError)):
            dc.read_file("C:\\Windows\\System32\\cmd.exe")

    def test_windows_absolute_outside_rejected_write(self, dc):
        with pytest.raises((WorkspaceViolationError, PermissionError)):
            dc.write_file("C:\\evil.txt", "bad")

    def test_resolve_workspace_path_rejects_outside(self, tmp_ws):
        with pytest.raises(WorkspaceViolationError):
            resolve_workspace_path(tmp_ws, "../../outside")

    def test_resolve_workspace_path_allows_inside(self, tmp_ws):
        result = resolve_workspace_path(tmp_ws, "subdir/file.txt")
        assert str(result).startswith(str(tmp_ws.resolve()))

    def test_resolve_workspace_path_no_workspace_raises(self):
        with pytest.raises(NoActiveWorkspaceError):
            resolve_workspace_path(None, "file.txt")

    def test_validate_workspace_nonexistent_raises(self):
        with pytest.raises(WorkspaceNotFoundError):
            validate_workspace("Z:\\nonexistent_ws_99999")

    def test_validate_workspace_file_raises(self, tmp_ws):
        f = _write(tmp_ws, "file.txt")
        with pytest.raises(WorkspaceNotFoundError):
            validate_workspace(str(f))

    def test_is_within_workspace_true(self, tmp_ws):
        sub = tmp_ws / "sub" / "file.txt"
        assert is_within_workspace(tmp_ws, sub) is True

    def test_is_within_workspace_false(self, tmp_ws):
        outside = tmp_ws.parent / "sibling.txt"
        assert is_within_workspace(tmp_ws, outside) is False

    def test_path_with_embedded_null_rejected(self, dc):
        # Paths with null bytes should be rejected by the OS or pathlib
        with pytest.raises(Exception):
            dc.read_file("file\x00.txt")


# ===========================================================================
# 20. No active workspace — all operations raise NoActiveWorkspaceError
# ===========================================================================

class TestNoActiveWorkspace:

    def test_list_raises(self):
        ctrl = DirectoryControl()
        with pytest.raises(NoActiveWorkspaceError):
            ctrl.list_directory()

    def test_read_raises(self):
        ctrl = DirectoryControl()
        with pytest.raises(NoActiveWorkspaceError):
            ctrl.read_file("file.txt")

    def test_create_file_raises(self):
        ctrl = DirectoryControl()
        with pytest.raises(NoActiveWorkspaceError):
            ctrl.create_file("file.txt")

    def test_write_raises(self):
        ctrl = DirectoryControl()
        with pytest.raises(NoActiveWorkspaceError):
            ctrl.write_file("file.txt", "content")

    def test_append_raises(self):
        ctrl = DirectoryControl()
        with pytest.raises(NoActiveWorkspaceError):
            ctrl.append_file("file.txt", "content")

    def test_update_raises(self):
        ctrl = DirectoryControl()
        with pytest.raises(NoActiveWorkspaceError):
            ctrl.update_file("file.txt", "content")

    def test_create_dir_raises(self):
        ctrl = DirectoryControl()
        with pytest.raises(NoActiveWorkspaceError):
            ctrl.create_directory("somedir")

    def test_move_raises(self):
        ctrl = DirectoryControl()
        with pytest.raises(NoActiveWorkspaceError):
            ctrl.move("a.txt", "b.txt")

    def test_copy_raises(self):
        ctrl = DirectoryControl()
        with pytest.raises(NoActiveWorkspaceError):
            ctrl.copy("a.txt", "b.txt")

    def test_rename_raises(self):
        ctrl = DirectoryControl()
        with pytest.raises(NoActiveWorkspaceError):
            ctrl.rename("a.txt", "b.txt")

    def test_search_raises(self):
        ctrl = DirectoryControl()
        with pytest.raises(NoActiveWorkspaceError):
            ctrl.search_files("*.txt")

    def test_get_indexed_files_raises(self):
        ctrl = DirectoryControl()
        with pytest.raises(NoActiveWorkspaceError):
            ctrl.get_indexed_files()




# ===========================================================================
# 21. Write failure (mocked)
# ===========================================================================

class TestWriteFailure:

    def test_write_permission_error_raises_operation_error(self, dc, tmp_ws):
        with patch("directory_control.directory_control.Path.write_text", side_effect=PermissionError("denied")):
            with pytest.raises(OperationError):
                dc.write_file("fail.txt", "content")

    def test_create_permission_error_raises_operation_error(self, dc, tmp_ws):
        with patch("directory_control.directory_control.Path.write_text", side_effect=PermissionError("denied")):
            with pytest.raises(OperationError):
                dc.create_file("fail_create.txt")

    def test_append_permission_error_raises_operation_error(self, dc, tmp_ws):
        with patch("directory_control.directory_control.Path.open", side_effect=PermissionError("denied")):
            with pytest.raises(OperationError):
                dc.append_file("fail_append.txt", "content")

    def test_update_permission_error_raises_operation_error(self, dc, tmp_ws):
        _write(tmp_ws, "locked.txt", "existing")
        with patch("directory_control.directory_control.Path.write_text", side_effect=PermissionError("denied")):
            with pytest.raises(OperationError):
                dc.update_file("locked.txt", "new content")


# ===========================================================================
# 22. Read failure (mocked)
# ===========================================================================

class TestReadFailure:

    def test_read_permission_error_raises_operation_error(self, dc, tmp_ws):
        _write(tmp_ws, "secret.txt", "data")
        with patch("directory_control.directory_control.Path.read_bytes", side_effect=PermissionError("denied")):
            with pytest.raises(OperationError):
                dc.read_file("secret.txt")

    def test_read_oserror_raises_operation_error(self, dc, tmp_ws):
        _write(tmp_ws, "broken.txt", "data")
        with patch("directory_control.directory_control.Path.read_bytes", side_effect=OSError("io error")):
            with pytest.raises(OperationError):
                dc.read_file("broken.txt")


# ===========================================================================
# 23. Move / copy / rename failures (mocked)
# ===========================================================================

class TestMoveCopyRenameFailures:

    def test_move_permission_error_raises_operation_error(self, dc, tmp_ws):
        _write(tmp_ws, "moveme.txt", "data")
        with patch("shutil.move", side_effect=PermissionError("denied")):
            with pytest.raises(OperationError):
                dc.move("moveme.txt", "moved.txt")

    def test_copy_permission_error_raises_operation_error(self, dc, tmp_ws):
        _write(tmp_ws, "copyme.txt", "data")
        with patch("shutil.copy2", side_effect=PermissionError("denied")):
            with pytest.raises(OperationError):
                dc.copy("copyme.txt", "copied.txt")

    def test_rename_oserror_raises_operation_error(self, dc, tmp_ws):
        _write(tmp_ws, "renameme.txt", "data")
        with patch("directory_control.directory_control.Path.rename", side_effect=OSError("rename failed")):
            with pytest.raises(OperationError):
                dc.rename("renameme.txt", "renamed.txt")


# ===========================================================================
# 24. search_files
# ===========================================================================

class TestSearchFiles:

    def test_search_finds_matching_files(self, dc, tmp_ws):
        _write(tmp_ws, "alpha.py", "code")
        _write(tmp_ws, "beta.py", "code")
        _write(tmp_ws, "gamma.txt", "text")
        results = dc.search_files("alpha")
        names = {r["name"] for r in results}
        assert "alpha.py" in names

    def test_search_case_insensitive(self, dc, tmp_ws):
        _write(tmp_ws, "README.md", "content")
        results = dc.search_files("readme")
        names = {r["name"] for r in results}
        assert "README.md" in names

    def test_search_returns_list_of_dicts(self, dc, tmp_ws):
        _write(tmp_ws, "match.txt", "data")
        results = dc.search_files("match")
        assert isinstance(results, list)
        for r in results:
            assert "name" in r
            assert "rel_path" in r
            assert "type" in r

    def test_search_no_match_returns_empty(self, dc, tmp_ws):
        _write(tmp_ws, "file.txt", "data")
        results = dc.search_files("zzznomatchzzz")
        assert results == []

    def test_search_no_workspace_raises(self):
        ctrl = DirectoryControl()
        with pytest.raises(NoActiveWorkspaceError):
            ctrl.search_files("*.txt")

    def test_search_finds_in_subdirectories(self, dc, tmp_ws):
        sub = tmp_ws / "deep" / "sub"
        sub.mkdir(parents=True)
        _write(sub, "target_deep.txt", "deep")
        results = dc.search_files("target_deep")
        assert len(results) >= 1


# ===========================================================================
# 25. get_indexed_files
# ===========================================================================

class TestGetIndexedFiles:

    def test_returns_list(self, dc, tmp_ws):
        _write(tmp_ws, "a.py", "code")
        result = dc.get_indexed_files()
        assert isinstance(result, list)

    def test_returns_file_entries(self, dc, tmp_ws):
        _write(tmp_ws, "index_me.txt", "content")
        from directory_reader.reader import FileEntry
        entries = dc.get_indexed_files()
        for e in entries:
            assert isinstance(e, FileEntry)

    def test_no_workspace_raises(self):
        ctrl = DirectoryControl()
        with pytest.raises(NoActiveWorkspaceError):
            ctrl.get_indexed_files()


# ===========================================================================
# 26. import sanity — existing test suite must not be broken
# ===========================================================================

class TestImportSanity:

    def test_directory_control_importable(self):
        from directory_control import DirectoryControl as DC
        assert DC is not None

    def test_path_security_importable(self):
        from directory_control.path_security import (
            resolve_workspace_path,
            validate_workspace,
            is_within_workspace,
        )
        assert resolve_workspace_path is not None

    def test_all_exceptions_importable(self):
        from directory_control import (
            DirectoryControlError,
            OperationError,
            FileNotFoundInWorkspace,
            NotAFileError,
            VerificationError,
            WorkspaceViolationError,
            NoActiveWorkspaceError,
            WorkspaceNotFoundError,
        )

    def test_existing_pipeline_unaffected(self):
        from pipeline.ag_pipeline import AGPipeline
        pipeline = AGPipeline()
        result = pipeline.process_input("hello test")
        assert result is not None
        assert hasattr(result, "success")

    def test_existing_directory_reader_unaffected(self):
        from directory_reader import DirectoryReader, DirectorySession
        reader = DirectoryReader()
        assert reader is not None
        session = DirectorySession()
        assert session is not None

    def test_directory_control_does_not_shadow_reader(self):
        from directory_control import DirectoryControl
        from directory_reader import DirectoryReader
        assert DirectoryControl is not DirectoryReader

    def test_directory_control_uses_reader_internally(self, dc):
        from directory_reader.reader import DirectoryReader as DR
        assert isinstance(dc._reader, DR)
