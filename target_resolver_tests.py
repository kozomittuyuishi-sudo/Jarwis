"""
target_resolver_tests.py
=========================
Independent regression tests for target_resolver.py / target_resolution_models.py.

Runs entirely standalone — no dependency on QUIPS, Preprocessor Brain,
or any other AG subsystem. `python3 target_resolver_tests.py` runs
everything and prints a pass/fail summary; every check is a real
`assert` so a regression fails loudly rather than needing to be
eyeballed.

TEST STRATEGY — WINDOWS PATHS ON A NON-WINDOWS HOST
-----------------------------------------------------
Two genuinely different things are being tested, and they are kept
deliberately separate:

1. STRING-LEVEL normalization/classification (normalize_target_string,
   _looks_absolute) — these are pure functions with no filesystem
   dependency, implemented on `ntpath`/`posixpath` specifically so they
   behave identically on any host. These ARE tested directly against
   literal Windows-style strings like "D:\\AG\\test.txt", including the
   explicit spaces-in-paths edge case, with no real filesystem involved.

2. END-TO-END resolve() behavior that depends on real existence checks
   (EXISTS_FILE / EXISTS_DIRECTORY / AMBIGUOUS / smart search, etc.) —
   these necessarily use `pathlib.Path`, which is host-native. This
   suite runs on whatever host it's invoked on, so these tests build
   real temporary directories using THIS host's native path dialect
   (POSIX, if run in this environment) as stand-ins for what would be
   "D:\\AG\\..." on a real Windows deployment. The resolution LOGIC
   being exercised (absolute/relative/workspace-relative/named-scope/
   smart-search/ambiguous/outside-scope) is identical either way; only
   the literal separator character differs. This is noted explicitly
   wherever it matters, and is also called out in the final report.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from target_resolution_models import (
    ActionHint,
    ExistenceState,
    ResolutionMethod,
    ResolutionMode,
    ResolutionStatus,
    TargetReference,
    TargetType,
)
from target_resolver import TargetResolver, normalize_target_string, _looks_absolute


_FAILURES = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if not condition:
        _FAILURES.append(f"{label}: {detail}")


# =============================================================================
# 1. Pure string normalization (host-independent)
# =============================================================================

def test_string_normalization() -> None:
    check("normalize backslash drive path",
          normalize_target_string("D:\\AG\\test.txt") == "D:\\AG\\test.txt")
    check("normalize forward-slash drive path",
          normalize_target_string("D:/AG/test.txt") == "D:\\AG\\test.txt",
          normalize_target_string("D:/AG/test.txt"))
    check("normalize redundant separators",
          normalize_target_string("D:\\AG\\\\backup\\\\test.txt") == "D:\\AG\\backup\\test.txt",
          normalize_target_string("D:\\AG\\\\backup\\\\test.txt"))
    check("normalize dot segment",
          normalize_target_string("D:\\AG\\.\\test.txt") == "D:\\AG\\test.txt",
          normalize_target_string("D:\\AG\\.\\test.txt"))
    check("normalize parent traversal collapses",
          normalize_target_string("D:\\AG\\backup\\..\\test.txt") == "D:\\AG\\test.txt",
          normalize_target_string("D:\\AG\\backup\\..\\test.txt"))
    check("normalize preserves spaces in path",
          normalize_target_string("D:\\My Projects\\AG\\test file.txt")
          == "D:\\My Projects\\AG\\test file.txt",
          normalize_target_string("D:\\My Projects\\AG\\test file.txt"))

    check("absolute detection: drive backslash", _looks_absolute("D:\\AG\\test.txt"))
    check("absolute detection: drive forward-slash", _looks_absolute("D:/AG/test.txt"))
    check("absolute detection: UNC", _looks_absolute("\\\\server\\share\\file.txt"))
    check("absolute detection: posix", _looks_absolute("/home/user/file.txt"))
    check("absolute detection: relative is NOT absolute", not _looks_absolute("test.txt"))
    check("absolute detection: relative nested is NOT absolute",
          not _looks_absolute("backup\\test.txt"))


# =============================================================================
# 2. End-to-end resolve() — built on a real, host-native fixture tree
# =============================================================================

def _build_fixture():
    """
    Builds (on THIS host's native path dialect):

        <tmp>/AG/                      <- workspace root, also registered scope "AG"
            test.txt
            Brain.py
            backup/
                test.txt
                configs/
                    test.json
            brains/
                Brain.py                <- duplicate name -> ambiguity with root Brain.py
            "folder with spaces"/
                "file with spaces.txt"
        <tmp>/Other/                   <- sibling, NOT inside the workspace
            outside.txt
    """
    tmp = tempfile.mkdtemp(prefix="target_resolver_test_")
    root = Path(tmp)
    ws = root / "AG"
    ws.mkdir()
    (ws / "test.txt").write_text("hello", encoding="utf-8")
    (ws / "Brain.py").write_text("# root brain\n", encoding="utf-8")
    (ws / "backup").mkdir()
    (ws / "backup" / "test.txt").write_text("backup copy", encoding="utf-8")
    (ws / "backup" / "configs").mkdir()
    (ws / "backup" / "configs" / "test.json").write_text("{}", encoding="utf-8")
    (ws / "brains").mkdir()
    (ws / "brains" / "Brain.py").write_text("# nested brain\n", encoding="utf-8")
    spaced_dir = ws / "folder with spaces"
    spaced_dir.mkdir()
    (spaced_dir / "file with spaces.txt").write_text("spaced", encoding="utf-8")

    other = root / "Other"
    other.mkdir()
    (other / "outside.txt").write_text("outside", encoding="utf-8")

    return root, ws, other


def test_absolute_paths(resolver: TargetResolver, ws: Path) -> None:
    ref = TargetReference(
        original_target=str(ws / "test.txt"),
        resolution_mode=ResolutionMode.ABSOLUTE_PATH,
        action=ActionHint.READ,
    )
    result = resolver.resolve(ref)
    check("absolute: resolved", result.status == ResolutionStatus.RESOLVED, result.to_dict())
    check("absolute: method", result.method == ResolutionMethod.ABSOLUTE_PATH)
    check("absolute: exists as file", result.existence_state == ExistenceState.EXISTS_FILE)
    check("absolute: target_type FILE", result.target_type == TargetType.FILE)


def test_relative_paths(resolver: TargetResolver, ws: Path) -> None:
    ref = TargetReference(original_target="test.txt", resolution_mode=ResolutionMode.RELATIVE_PATH)
    result = resolver.resolve(ref)
    check("relative: resolved via active workspace", result.status == ResolutionStatus.RESOLVED,
          result.to_dict())
    check("relative: correct path", Path(result.resolved_path) == ws / "test.txt")

    nested_ref = TargetReference(original_target=str(Path("backup") / "test.txt"),
                                  resolution_mode=ResolutionMode.RELATIVE_PATH)
    nested_result = resolver.resolve(nested_ref)
    check("relative nested: resolved", nested_result.status == ResolutionStatus.RESOLVED)
    check("relative nested: correct path", Path(nested_result.resolved_path) == ws / "backup" / "test.txt")

    explicit_base_ref = TargetReference(
        original_target="outside.txt", resolution_mode=ResolutionMode.RELATIVE_PATH,
        context={"base_dir": str(ws.parent / "Other")},
    )
    explicit_result = resolver.resolve(explicit_base_ref)
    check("relative: explicit context.base_dir honored",
          explicit_result.status == ResolutionStatus.OUTSIDE_SCOPE,
          "explicit base outside the workspace should be boundary-checked against "
          "the workspace, not silently allowed: " + str(explicit_result.to_dict()))


def test_nested_paths(resolver: TargetResolver, ws: Path) -> None:
    ref = TargetReference(original_target=str(Path("backup") / "configs" / "test.json"),
                           resolution_mode=ResolutionMode.WORKSPACE_RELATIVE)
    result = resolver.resolve(ref)
    check("nested: resolved", result.status == ResolutionStatus.RESOLVED, result.to_dict())
    check("nested: correct path", Path(result.resolved_path) == ws / "backup" / "configs" / "test.json")
    check("nested: exists as file", result.existence_state == ExistenceState.EXISTS_FILE)


def test_parent_traversal(resolver: TargetResolver, ws: Path) -> None:
    # ".." alone, from the workspace root, escapes the workspace -> OUTSIDE_SCOPE.
    ref = TargetReference(original_target="..", resolution_mode=ResolutionMode.WORKSPACE_RELATIVE)
    result = resolver.resolve(ref)
    check("parent traversal '..': rejected as outside scope",
          result.status == ResolutionStatus.OUTSIDE_SCOPE, result.to_dict())

    ref2 = TargetReference(original_target=str(Path("..") / "Other" / "outside.txt"),
                            resolution_mode=ResolutionMode.WORKSPACE_RELATIVE)
    result2 = resolver.resolve(ref2)
    check("parent traversal '..\\folder': rejected as outside scope",
          result2.status == ResolutionStatus.OUTSIDE_SCOPE, result2.to_dict())

    # A traversal that stays INSIDE the workspace must still be allowed.
    ref3 = TargetReference(original_target=str(Path("backup") / ".." / "test.txt"),
                            resolution_mode=ResolutionMode.WORKSPACE_RELATIVE)
    result3 = resolver.resolve(ref3)
    check("parent traversal staying inside workspace: resolved",
          result3.status == ResolutionStatus.RESOLVED, result3.to_dict())
    check("parent traversal staying inside workspace: correct path",
          Path(result3.resolved_path) == ws / "test.txt")


def test_workspace_relative(resolver: TargetResolver, ws: Path) -> None:
    ref = TargetReference(original_target=str(Path("backup") / "test.txt"),
                           resolution_mode=ResolutionMode.WORKSPACE_RELATIVE)
    result = resolver.resolve(ref)
    check("workspace-relative: resolved", result.status == ResolutionStatus.RESOLVED)
    check("workspace-relative: correct path", Path(result.resolved_path) == ws / "backup" / "test.txt")
    check("workspace-relative: method", result.method == ResolutionMethod.WORKSPACE_RELATIVE)


def test_named_scope(resolver: TargetResolver, ws: Path) -> None:
    # "test.txt in AG" -- "AG" matches the active workspace's own name.
    ref = TargetReference(original_target="test.txt", scope="AG",
                           resolution_mode=ResolutionMode.NAMED_SCOPE)
    result = resolver.resolve(ref)
    check("named scope (workspace name): resolved", result.status == ResolutionStatus.RESOLVED,
          result.to_dict())
    check("named scope (workspace name): correct path", Path(result.resolved_path) == ws / "test.txt")
    check("named scope (workspace name): method", result.method == ResolutionMethod.NAMED_SCOPE)

    # "Brain.py in brains" -- "brains" is a subdirectory of the active workspace.
    ref2 = TargetReference(original_target="Brain.py", scope="brains",
                            resolution_mode=ResolutionMode.NAMED_SCOPE)
    result2 = resolver.resolve(ref2)
    check("named scope (subdirectory): resolved", result2.status == ResolutionStatus.RESOLVED,
          result2.to_dict())
    check("named scope (subdirectory): correct path",
          Path(result2.resolved_path) == ws / "brains" / "Brain.py")

    # An explicitly registered project scope.
    resolver.register_scope("Backups", str(ws / "backup"))
    ref3 = TargetReference(original_target="test.txt", scope="Backups",
                            resolution_mode=ResolutionMode.NAMED_SCOPE)
    result3 = resolver.resolve(ref3)
    check("named scope (registered project): resolved", result3.status == ResolutionStatus.RESOLVED)
    check("named scope (registered project): correct path",
          Path(result3.resolved_path) == ws / "backup" / "test.txt")


def test_missing_scope(resolver: TargetResolver) -> None:
    ref = TargetReference(original_target="test.txt", scope="NoSuchProject",
                           resolution_mode=ResolutionMode.NAMED_SCOPE)
    result = resolver.resolve(ref)
    check("missing scope: SCOPE_NOT_FOUND", result.status == ResolutionStatus.SCOPE_NOT_FOUND,
          result.to_dict())
    check("missing scope: no resolved_path", result.resolved_path is None)


def test_smart_search(resolver: TargetResolver, ws: Path) -> None:
    # Unique name -> exactly one match.
    ref = TargetReference(original_target="test.json", resolution_mode=ResolutionMode.SMART_SEARCH)
    result = resolver.resolve(ref)
    check("smart search (unique): resolved", result.status == ResolutionStatus.RESOLVED,
          result.to_dict())
    check("smart search (unique): correct path",
          Path(result.resolved_path) == ws / "backup" / "configs" / "test.json")
    check("smart search (unique): method", result.method == ResolutionMethod.SMART_SEARCH)

    # Ambiguous: "Brain.py" exists at both the workspace root and in brains/.
    ambiguous_ref = TargetReference(original_target="Brain.py",
                                     resolution_mode=ResolutionMode.SMART_SEARCH)
    ambiguous_result = resolver.resolve(ambiguous_ref)
    check("smart search (ambiguous): status", ambiguous_result.status == ResolutionStatus.AMBIGUOUS,
          ambiguous_result.to_dict())
    check("smart search (ambiguous): no single resolved_path", ambiguous_result.resolved_path is None)
    check("smart search (ambiguous): two candidates", len(ambiguous_result.candidates) == 2,
          ambiguous_result.candidates)
    expected = {str(ws / "Brain.py"), str(ws / "brains" / "Brain.py")}
    check("smart search (ambiguous): correct candidates", set(ambiguous_result.candidates) == expected,
          ambiguous_result.candidates)

    # Nonexistent -> NOT_FOUND.
    missing_ref = TargetReference(original_target="does_not_exist.xyz",
                                   resolution_mode=ResolutionMode.SMART_SEARCH)
    missing_result = resolver.resolve(missing_ref)
    check("smart search (missing): NOT_FOUND", missing_result.status == ResolutionStatus.NOT_FOUND,
          missing_result.to_dict())


def test_existing_and_nonexisting_targets(resolver: TargetResolver, ws: Path) -> None:
    existing = resolver.resolve(TargetReference(original_target="test.txt",
                                                  resolution_mode=ResolutionMode.WORKSPACE_RELATIVE))
    check("existing target: EXISTS_FILE", existing.existence_state == ExistenceState.EXISTS_FILE)
    check("existing target: still RESOLVED", existing.status == ResolutionStatus.RESOLVED)

    # Default action (UNSPECIFIED) -> a missing target is still RESOLVED,
    # just flagged NOT_EXISTS (required for CREATE-type callers).
    nonexisting = resolver.resolve(TargetReference(original_target="new_file.txt",
                                                      resolution_mode=ResolutionMode.WORKSPACE_RELATIVE))
    check("nonexisting target (no action hint): RESOLVED",
          nonexisting.status == ResolutionStatus.RESOLVED, nonexisting.to_dict())
    check("nonexisting target (no action hint): NOT_EXISTS",
          nonexisting.existence_state == ExistenceState.NOT_EXISTS)

    # READ action on a missing target -> downgraded to NOT_FOUND.
    nonexisting_read = resolver.resolve(TargetReference(
        original_target="new_file.txt", resolution_mode=ResolutionMode.WORKSPACE_RELATIVE,
        action=ActionHint.READ,
    ))
    check("nonexisting target (READ action): NOT_FOUND",
          nonexisting_read.status == ResolutionStatus.NOT_FOUND, nonexisting_read.to_dict())

    # CREATE action on a missing target -> stays RESOLVED (the whole point).
    nonexisting_create = resolver.resolve(TargetReference(
        original_target="new_file.txt", resolution_mode=ResolutionMode.WORKSPACE_RELATIVE,
        action=ActionHint.CREATE,
    ))
    check("nonexisting target (CREATE action): stays RESOLVED",
          nonexisting_create.status == ResolutionStatus.RESOLVED, nonexisting_create.to_dict())


def test_invalid_targets(resolver: TargetResolver) -> None:
    empty = resolver.resolve(TargetReference(original_target="",
                                                resolution_mode=ResolutionMode.RELATIVE_PATH))
    check("invalid target (empty): INVALID_TARGET", empty.status == ResolutionStatus.INVALID_TARGET)

    null_byte = resolver.resolve(TargetReference(original_target="test\x00.txt",
                                                    resolution_mode=ResolutionMode.RELATIVE_PATH))
    check("invalid target (null byte): INVALID_TARGET",
          null_byte.status == ResolutionStatus.INVALID_TARGET)


def test_outside_workspace(resolver: TargetResolver, ws: Path, other: Path) -> None:
    ref = TargetReference(original_target=str(other / "outside.txt"),
                           resolution_mode=ResolutionMode.ABSOLUTE_PATH)
    result = resolver.resolve(ref)
    check("outside workspace (absolute): OUTSIDE_SCOPE",
          result.status == ResolutionStatus.OUTSIDE_SCOPE, result.to_dict())
    check("outside workspace: resolved_path still reported for diagnostics",
          result.resolved_path is not None)


def test_windows_edge_cases_with_spaces(resolver: TargetResolver, ws: Path) -> None:
    ref = TargetReference(original_target=str(Path("folder with spaces") / "file with spaces.txt"),
                           resolution_mode=ResolutionMode.WORKSPACE_RELATIVE)
    result = resolver.resolve(ref)
    check("spaces in path: resolved", result.status == ResolutionStatus.RESOLVED, result.to_dict())
    check("spaces in path: exists", result.existence_state == ExistenceState.EXISTS_FILE)
    check("spaces in path: correct path",
          Path(result.resolved_path) == ws / "folder with spaces" / "file with spaces.txt")

    # Pure Windows-dialect string form, verified at the normalization
    # level (see module docstring for why this is split from the
    # existence-dependent check above).
    windows_form = normalize_target_string("D:\\My Projects\\AG\\folder with spaces\\file.txt")
    check("spaces in path (windows-dialect string, normalization only)",
          windows_form == "D:\\My Projects\\AG\\folder with spaces\\file.txt", windows_form)


def test_unspecified_mode_classification(resolver: TargetResolver, ws: Path, other: Path) -> None:
    abs_ref = TargetReference(original_target=str(ws / "test.txt"))
    check("unspecified -> absolute classified correctly",
          resolver.resolve(abs_ref).method == ResolutionMethod.ABSOLUTE_PATH)

    scope_ref = TargetReference(original_target="test.txt", scope="AG")
    check("unspecified + scope -> named scope classified correctly",
          resolver.resolve(scope_ref).method == ResolutionMethod.NAMED_SCOPE)

    plain_ref = TargetReference(original_target="test.txt")
    check("unspecified + active workspace -> workspace-relative classified correctly",
          resolver.resolve(plain_ref).method == ResolutionMethod.WORKSPACE_RELATIVE)


def test_no_active_workspace_behaves_safely() -> None:
    bare_resolver = TargetResolver(workspace_source=None)
    result = bare_resolver.resolve(TargetReference(original_target="test.txt",
                                                      resolution_mode=ResolutionMode.WORKSPACE_RELATIVE))
    check("no workspace: MISSING_CONTEXT, not a guess",
          result.status == ResolutionStatus.MISSING_CONTEXT, result.to_dict())


def test_duck_typed_workspace_sources(ws: Path) -> None:
    # Plain string source.
    r1 = TargetResolver(workspace_source=str(ws))
    check("duck-typed source: plain string",
          r1.resolve(TargetReference(original_target="test.txt")).status == ResolutionStatus.RESOLVED)

    # Zero-arg callable source.
    r2 = TargetResolver(workspace_source=lambda: str(ws))
    check("duck-typed source: callable",
          r2.resolve(TargetReference(original_target="test.txt")).status == ResolutionStatus.RESOLVED)

    # Object exposing get_active_directory(), mirroring AG's real
    # DirectoryControl / Preprocessor Brain's DirectoryOperations
    # without importing either.
    class _FakeDirectoryControl:
        def get_active_directory(self) -> str:
            return str(ws)

    r3 = TargetResolver(workspace_source=_FakeDirectoryControl())
    check("duck-typed source: get_active_directory() object",
          r3.resolve(TargetReference(original_target="test.txt")).status == ResolutionStatus.RESOLVED)


# =============================================================================
# Runner
# =============================================================================

def run_all() -> None:
    test_string_normalization()

    root, ws, other = _build_fixture()
    try:
        resolver = TargetResolver(workspace_source=str(ws), scope_registry={"AG": str(ws)})

        test_absolute_paths(resolver, ws)
        test_relative_paths(resolver, ws)
        test_nested_paths(resolver, ws)
        test_parent_traversal(resolver, ws)
        test_workspace_relative(resolver, ws)
        test_named_scope(resolver, ws)
        test_missing_scope(resolver)
        test_smart_search(resolver, ws)
        test_existing_and_nonexisting_targets(resolver, ws)
        test_invalid_targets(resolver)
        test_outside_workspace(resolver, ws, other)
        test_windows_edge_cases_with_spaces(resolver, ws)
        test_unspecified_mode_classification(resolver, ws, other)
        test_no_active_workspace_behaves_safely()
        test_duck_typed_workspace_sources(ws)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print("=" * 70)
    if _FAILURES:
        print(f"TARGET RESOLVER TESTS: {len(_FAILURES)} FAILURE(S)\n")
        for f in _FAILURES:
            print(f"  - {f}")
        raise SystemExit(1)
    print("TARGET RESOLVER TESTS: ALL PASSED")


if __name__ == "__main__":
    run_all()