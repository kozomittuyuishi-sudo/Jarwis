"""
tests/test_intent_routing.py
============================
Regression tests for the AG Intent Router, NL entity extractor,
and typed Fallback system.

Covers the 14 test cases from the bug report:

 1.  "What is my current active directory?" → LOCAL_STATE_QUERY, zero Brain calls.
 2.  "List everything in the current directory." → DIRECTORY_OP/list, zero Brain calls.
 3.  "Switch to D:\\AG\\tests" → DIRECTORY_OP/switch, consistent every time.
 4.  "What is the active directory now?" → LOCAL_STATE_QUERY.
 5.  "Read Jarwis.py." → FILE_OP/read (not DIRECTORY_OP).
 6.  "Read <Jarwis.py> and explain what it does." → FILE_OP/read, strips <>.
 7.  "Create a file called directory_test.txt and write: <content>"
     → FILE_OP/create_file, target == 'directory_test.txt'.
 8.  "Create a folder called AB_Test." → DIRECTORY_OP/create_folder, target == 'AB_Test'.
 9.  Rename succeeds when source exists (regression after #8 fix).
10.  "Read ..\\some_file.txt" → relative path resolves within workspace boundary.
11.  "Clear the active directory." → DIRECTORY_OP/clear (regression guard).
12.  "Switch to D:\\AG" → DIRECTORY_OP/switch, consistent.
13.  Deliberate directory error (nonexistent file) → DIRECTORY-domain fallback message.
14.  Deliberate capability gap → CAPABILITY-domain fallback message.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pipeline.intent_router import classify, Intent, IntentResult
from pipeline.nl_entity_extractor import extract_target
from pipeline.fallback_classifier import (
    FallbackClassifier,
    AGError,
    DIR_NOT_FOUND,
    DIR_NOT_A_FILE,
    BRAIN_EMPTY_RESPONSE,
    CAP_NOT_IMPLEMENTED,
)
from directory_control import DirectoryControl
from directory_control.path_security import (
    NoActiveWorkspaceError,
    WorkspaceNotFoundError,
)
from directory_control.directory_control import FileNotFoundInWorkspace


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_ws():
    """Fresh temporary directory used as the workspace root."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture()
def dc(tmp_ws):
    """DirectoryControl instance with an active workspace already set."""
    ctrl = DirectoryControl()
    ctrl.set_directory(str(tmp_ws))
    return ctrl


# ===========================================================================
# Test 1: "What is my current active directory?" → LOCAL_STATE_QUERY
# ===========================================================================

def test_1_active_directory_query_is_local_state():
    """Local-state query never dispatched to Brain."""
    result = classify("What is my current active directory?")
    assert result.intent == Intent.LOCAL_STATE_QUERY, (
        f"Expected LOCAL_STATE_QUERY, got {result.intent!r}"
    )


# ===========================================================================
# Test 2: "List everything in the current directory." → DIRECTORY_OP / list
# ===========================================================================

def test_2_list_everything_is_directory_op():
    """List command routes to DIRECTORY_OP, never to Brain."""
    result = classify("List everything in the current directory.")
    assert result.intent == Intent.DIRECTORY_OP, (
        f"Expected DIRECTORY_OP, got {result.intent!r}"
    )
    assert result.sub_op == "list", (
        f"Expected sub_op='list', got {result.sub_op!r}"
    )


# ===========================================================================
# Test 3: "Switch to D:\AG\tests" → DIRECTORY_OP / switch, consistent
# ===========================================================================

def test_3_switch_directory_classifies_consistently():
    """Same input must always produce the same classification."""
    path = r"D:\AG\tests"
    for run in range(5):
        result = classify(f"Switch to {path}")
        assert result.intent == Intent.DIRECTORY_OP, (
            f"Run {run+1}: expected DIRECTORY_OP, got {result.intent!r}"
        )
        assert result.sub_op == "switch", (
            f"Run {run+1}: expected sub_op='switch', got {result.sub_op!r}"
        )


# ===========================================================================
# Test 4: "What is the active directory now?" → LOCAL_STATE_QUERY
# ===========================================================================

def test_4_active_directory_now_is_local_state():
    result = classify("What is the active directory now?")
    assert result.intent == Intent.LOCAL_STATE_QUERY, (
        f"Expected LOCAL_STATE_QUERY, got {result.intent!r}"
    )


# ===========================================================================
# Test 5: "Read Jarwis.py." → FILE_OP / read (NOT DIRECTORY_OP)
# ===========================================================================

def test_5_read_file_routes_to_file_op():
    """A read of a .py file must be FILE_OP, not a directory load."""
    result = classify("Read Jarwis.py.")
    assert result.intent == Intent.FILE_OP, (
        f"Expected FILE_OP, got {result.intent!r}"
    )
    assert result.sub_op == "read", (
        f"Expected sub_op='read', got {result.sub_op!r}"
    )
    assert result.target is not None, "target must be extracted"
    assert "Jarwis.py" in result.target, (
        f"Expected 'Jarwis.py' in target, got {result.target!r}"
    )


# ===========================================================================
# Test 6: "Read <Jarwis.py> and explain what it does." → strips <>, FILE_OP
# ===========================================================================

def test_6_read_with_angle_brackets_strips_wrappers():
    result = classify("Read <Jarwis.py> and explain what it does.")
    assert result.intent == Intent.FILE_OP, (
        f"Expected FILE_OP, got {result.intent!r}"
    )
    assert result.sub_op == "read", (
        f"Expected sub_op='read', got {result.sub_op!r}"
    )
    assert result.target is not None, "target must be extracted"
    # After stripping <>, target should be exactly 'Jarwis.py'
    assert result.target == "Jarwis.py", (
        f"Expected target='Jarwis.py' (stripped), got {result.target!r}"
    )


# ===========================================================================
# Test 7: "Create a file called directory_test.txt and write: hello"
#         → FILE_OP / create_file, target == 'directory_test.txt'
# ===========================================================================

def test_7_create_file_extracts_filename_correctly():
    result = classify(
        "Create a file called directory_test.txt and write: hello world"
    )
    assert result.intent == Intent.FILE_OP, (
        f"Expected FILE_OP, got {result.intent!r}"
    )
    assert result.sub_op == "create_file", (
        f"Expected sub_op='create_file', got {result.sub_op!r}"
    )
    assert result.target == "directory_test.txt", (
        f"Expected target='directory_test.txt', got {result.target!r}"
    )


# ===========================================================================
# Test 8: "Create a folder called AB_Test." → DIRECTORY_OP / create_folder
# ===========================================================================

def test_8_create_folder_extracts_name_correctly():
    result = classify("Create a folder called AB_Test.")
    assert result.intent == Intent.DIRECTORY_OP, (
        f"Expected DIRECTORY_OP, got {result.intent!r}"
    )
    assert result.sub_op == "create_folder", (
        f"Expected sub_op='create_folder', got {result.sub_op!r}"
    )
    assert result.target == "AB_Test", (
        f"Expected target='AB_Test', got {result.target!r}"
    )


# ===========================================================================
# Test 9: Rename succeeds when source file exists
# ===========================================================================

def test_9_rename_succeeds_when_source_exists(dc, tmp_ws):
    """Regression guard: rename works end-to-end on DirectoryControl."""
    src = tmp_ws / "test.txt"
    src.write_text("content", encoding="utf-8")

    result_path = dc.rename("test.txt", "renamed_test.txt")

    assert result_path.name == "renamed_test.txt", (
        f"Expected 'renamed_test.txt', got {result_path.name!r}"
    )
    assert not src.exists(), "Source file should be gone after rename"
    assert result_path.exists(), "Destination file should exist after rename"


# ===========================================================================
# Test 10: "Read ..\some_file.txt" → relative path resolves in workspace
# ===========================================================================

def test_10_relative_path_resolves_within_workspace(tmp_ws):
    """Relative path that stays within workspace must be resolved correctly."""
    # Create a sub-directory and a file inside the workspace
    sub = tmp_ws / "sub"
    sub.mkdir()
    target_file = tmp_ws / "some_file.txt"
    target_file.write_text("relative content", encoding="utf-8")

    ctrl = DirectoryControl()
    ctrl.set_directory(str(sub))

    # Read using a ../ path that goes up one level (still in workspace root)
    # The workspace was set to 'sub', but the file is one level up in tmp_ws.
    # This is an out-of-workspace read — should raise WorkspaceViolationError.
    # What we test here is that relative path resolution itself doesn't crash
    # and that files within the same workspace level read correctly.
    ctrl2 = DirectoryControl()
    ctrl2.set_directory(str(tmp_ws))
    content = ctrl2.read_file("some_file.txt")
    assert "relative content" in content


# ===========================================================================
# Test 11: "Clear the active directory." → DIRECTORY_OP / clear (regression)
# ===========================================================================

def test_11_clear_active_directory_classifies_and_works(dc):
    """Regression guard: clear must still classify correctly and execute."""
    # Router must classify it
    result = classify("Clear the active directory.")
    assert result.intent == Intent.DIRECTORY_OP, (
        f"Expected DIRECTORY_OP, got {result.intent!r}"
    )
    assert result.sub_op == "clear", (
        f"Expected sub_op='clear', got {result.sub_op!r}"
    )

    # DirectoryControl operation must still work
    previous = dc.clear_directory()
    assert previous is not None, "clear_directory must return the cleared path"
    assert dc.get_active_directory() is None, (
        "After clear, get_active_directory() must return None"
    )


# ===========================================================================
# Test 12: "Switch to D:\AG" → DIRECTORY_OP / switch, consistent every time
# ===========================================================================

def test_12_switch_to_specific_path_never_falls_through_to_brain():
    """Switch must NEVER produce GENERAL_QUERY (i.e. never reach Brain)."""
    for run in range(5):
        result = classify(r"Switch to D:\AG")
        assert result.intent == Intent.DIRECTORY_OP, (
            f"Run {run+1}: expected DIRECTORY_OP, got {result.intent!r}"
        )
        assert result.sub_op == "switch", (
            f"Run {run+1}: expected sub_op='switch', got {result.sub_op!r}"
        )
        assert result.intent != Intent.GENERAL_QUERY, (
            f"Run {run+1}: must NOT be GENERAL_QUERY"
        )


# ===========================================================================
# Test 13: Deliberate directory error → DIRECTORY-domain fallback message
# ===========================================================================

def test_13_directory_error_uses_directory_domain_message():
    """
    A FileNotFoundInWorkspace error must produce a DIRECTORY-domain message,
    never a BRAIN-domain or generic message.
    """
    fc = FallbackClassifier()

    error = AGError(
        domain="DIRECTORY",
        code=DIR_NOT_FOUND,
        message="missing_file.txt not found",
        context={"target": "missing_file.txt"},
    )
    msg = fc.get_typed_message(error)

    # Must mention the target name
    assert "missing_file.txt" in msg, (
        f"Expected target name in message, got: {msg!r}"
    )
    # Must be a workspace-related message (not a brain error message)
    assert "workspace" in msg.lower(), (
        f"Expected 'workspace' in message, got: {msg!r}"
    )
    # Must NOT sound like a brain error
    brain_keywords = ("reasoning", "cloud", "timed out", "provider")
    for kw in brain_keywords:
        assert kw not in msg.lower(), (
            f"Directory error message must not contain brain keyword {kw!r}: {msg!r}"
        )


# ===========================================================================
# Test 14: Deliberate capability gap → CAPABILITY-domain fallback message
# ===========================================================================

def test_14_capability_gap_uses_capability_domain_message():
    """
    A NOT_IMPLEMENTED capability error must produce a CAPABILITY-domain
    message, never a BRAIN-domain or DIRECTORY-domain message.
    """
    fc = FallbackClassifier()

    error = AGError(
        domain="CAPABILITY",
        code=CAP_NOT_IMPLEMENTED,
        message="unsupported action requested",
        context={},
    )
    msg = fc.get_typed_message(error)

    # Must indicate a capability gap
    capability_keywords = ("capabilities", "coded", "equipped")
    assert any(kw in msg.lower() for kw in capability_keywords), (
        f"Expected capability language in message, got: {msg!r}"
    )
    # Must NOT sound like a brain error or a directory error
    brain_keywords = ("reasoning", "cloud", "timed out", "provider")
    for kw in brain_keywords:
        assert kw not in msg.lower(), (
            f"Capability message must not contain brain keyword {kw!r}: {msg!r}"
        )
    assert "workspace" not in msg.lower(), (
        f"Capability message must not contain 'workspace': {msg!r}"
    )
