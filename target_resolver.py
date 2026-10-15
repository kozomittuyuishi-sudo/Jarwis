"""
target_resolver.py
===================
Deterministic Target Resolver for AG.

ROLE
----
    QUIPS            "What does the user mean?"
    Target Resolver  "Where does that target actually exist?"   <- this file
    Directory Control "Perform the filesystem operation."

This module converts a structured TargetReference (produced by QUIPS)
into a structured ResolvedTarget (consumed by Directory Control). It is
NOT a natural-language parser and does not duplicate QUIPS's job: it
trusts `TargetReference.resolution_mode` to say which strategy applies,
rather than trying absolute -> relative -> workspace -> search -> guess
as a fallback cascade. See `resolve()` below for the one narrow,
documented exception (UNSPECIFIED mode).

INTEGRATION BOUNDARY (intentional, for now)
--------------------------------------------
This file does NOT import QUIPS, Preprocessor Brain, or Directory
Control. It is being developed independently so its contract can
stabilize before integration (per the current project phase). It
communicates with the rest of AG purely through the dataclasses in
target_resolution_models.py and through plain duck-typed objects passed
into its constructor (see WorkspaceProvider below) — so when
integration does happen, AG's real DirectoryControl/workspace object
can be handed in directly with zero changes here.

WINDOWS PATH HANDLING ON ANY HOST
-----------------------------------
AG's real deployment target is Windows (D:\\AG). String-level path
manipulation (normalizing separators, `.`/`..`, detecting whether a
string IS absolute) is implemented using the stdlib `ntpath` /
`posixpath` modules directly rather than relying on `pathlib.Path`'s
host-native behavior — `ntpath`/`posixpath` are pure string-logic
modules with no OS dependency, so Windows-style paths like
"D:\\AG\\test.txt" normalize and classify correctly even when this code
runs on a non-Windows development machine. Actual filesystem I/O
(existence checks) necessarily uses `pathlib.Path` and is therefore
host-native — see the module docstring note in target_resolver_tests.py
for how the test suite handles that split.

DO NOT confuse normalization with existence validation (see section
10.A of the design spec): a path can be structurally RESOLVED while not
existing yet — required for CREATE-type operations.
"""

from __future__ import annotations

import ntpath
import posixpath
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from target_resolution_models import (
    ActionHint,
    ExistenceState,
    ResolutionMethod,
    ResolutionMode,
    ResolutionStatus,
    ResolvedTarget,
    TargetReference,
    TargetType,
)


# =============================================================================
# SECTION 1 — Pure string-level path helpers (host-independent)
# =============================================================================

_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")
_WINDOWS_UNC_RE = re.compile(r"^\\\\")


def _looks_windows(raw: str) -> bool:
    """True if `raw` structurally looks like a Windows-style path."""
    return bool(_WINDOWS_DRIVE_RE.match(raw)) or bool(_WINDOWS_UNC_RE.match(raw)) or "\\" in raw


def looks_absolute(raw: str) -> bool:
    """
    OS-independent absolute-path detection. Deliberately NOT
    `pathlib.Path(raw).is_absolute()`, which gives the wrong answer for
    Windows-style strings when running on a POSIX host (and vice
    versa).

    Public on purpose: this is the one canonical implementation of
    "does this string look like an absolute path" for AG. QUIPS reuses
    it directly (rather than re-deriving its own copy) so the two
    subsystems can never classify the same string differently — pure
    string classification, not resolution, so this does not blur the
    QUIPS/Target-Resolver boundary.
    """
    if _WINDOWS_DRIVE_RE.match(raw) or _WINDOWS_UNC_RE.match(raw):
        return True
    if raw.startswith("//"):
        return True
    if raw.startswith("/"):
        return True
    return False


# Backward-compatible internal alias.
_looks_absolute = looks_absolute


def normalize_target_string(raw: str) -> str:
    """
    Normalize separators, `.`, `..`, and redundant separators — pure
    string transformation, no filesystem access, no existence check.
    Uses `ntpath` for Windows-style input and `posixpath` otherwise, so
    behavior is identical regardless of which OS this code runs on.
    """
    raw = raw.strip()
    if not raw:
        return raw
    if _looks_windows(raw):
        return ntpath.normpath(raw)
    return posixpath.normpath(raw)


def _join_and_normalize(base: str, rel: str) -> str:
    """Join + normalize, choosing the path dialect from whichever of
    `base`/`rel` looks Windows-style (if either does, both are treated
    as Windows-style — AG's paths are consistently one dialect in
    practice)."""
    if _looks_windows(base) or _looks_windows(rel):
        return ntpath.normpath(ntpath.join(base, rel))
    return posixpath.normpath(posixpath.join(base, rel))


def _basename(raw: str) -> str:
    """Extract the final path component, Windows- or POSIX-style."""
    if _looks_windows(raw):
        return ntpath.basename(raw.rstrip("\\/"))
    return posixpath.basename(raw.rstrip("/"))


def _is_within(root: str, candidate: str) -> bool:
    """
    OS-independent "is candidate inside root" check on the NORMALIZED
    STRING form. Used for workspace-boundary enforcement without
    relying on pathlib (which would mis-evaluate cross-dialect strings
    on the wrong host).
    """
    dialect_windows = _looks_windows(root) or _looks_windows(candidate)
    norm = ntpath.normpath if dialect_windows else posixpath.normpath
    sep = "\\" if dialect_windows else "/"

    root_n = norm(root)
    cand_n = norm(candidate)
    if dialect_windows:
        root_n, cand_n = root_n.lower(), cand_n.lower()

    if cand_n == root_n:
        return True
    root_with_sep = root_n if root_n.endswith(sep) else root_n + sep
    return cand_n.startswith(root_with_sep)


# =============================================================================
# SECTION 2 — Workspace adapter (duck-typed, zero coupling)
# =============================================================================

WorkspaceSource = Union[None, str, Callable[[], Optional[str]], Any]


class WorkspaceProvider:
    """
    Adapts ANY of the following into a single `get_active_directory()`
    call, so TargetResolver never has to import or know about AG's real
    DirectoryControl, the Preprocessor Brain's standalone fallback, or
    any future equivalent:

        - None                           -> no active workspace
        - a plain string                 -> that path, always
        - a zero-arg callable            -> called each time
        - any object exposing a callable
          `get_active_directory()`       -> that method is called
    """

    def __init__(self, source: WorkspaceSource):
        self._source = source

    def get_active_directory(self) -> Optional[str]:
        src = self._source
        if src is None:
            return None
        if isinstance(src, str):
            return src
        getter = getattr(src, "get_active_directory", None)
        if callable(getter):
            return getter()
        if callable(src):
            return src()
        return None


# =============================================================================
# SECTION 3 — Existence-required action policy
# =============================================================================

# Actions for which a resolver result pointing at a non-existent target
# should be reported as a resolution FAILURE (NOT_FOUND) rather than a
# successfully-resolved-but-absent path. Deliberately a short, closed
# list: everything else (CREATE, WRITE, APPEND, and any destination of
# a MOVE/COPY/RENAME) is expected to legitimately not exist yet, and
# business-rule enforcement (e.g. "destination must not already exist")
# belongs to Directory Control, not the resolver.
_EXISTENCE_REQUIRED_ACTIONS = frozenset({
    ActionHint.ACCESS,
    ActionHint.READ,
    ActionHint.LIST,
    ActionHint.DELETE,
    ActionHint.MOVE_SOURCE,
    ActionHint.COPY_SOURCE,
    ActionHint.RENAME_SOURCE,
})


# =============================================================================
# SECTION 4 — TargetResolver
# =============================================================================

class TargetResolver:
    """
    Deterministic target resolver. See module docstring for the overall
    contract. Construct once per session/request context and reuse;
    holds no per-call mutable state other than the registered scopes.
    """

    def __init__(
        self,
        workspace_source: WorkspaceSource = None,
        scope_registry: Optional[Dict[str, str]] = None,
        max_search_depth: int = 3,
    ) -> None:
        self._workspace = WorkspaceProvider(workspace_source)
        self._scope_registry: Dict[str, str] = dict(scope_registry or {})
        self._max_search_depth = max(0, max_search_depth)

    # ------------------------------------------------------------------
    # Public configuration
    # ------------------------------------------------------------------

    def register_scope(self, name: str, path: str) -> None:
        """Register a known named scope/project (e.g. "AG" -> "D:\\AG").
        Deterministic, explicit — never inferred or guessed."""
        self._scope_registry[name] = path

    def active_workspace(self) -> Optional[str]:
        return self._workspace.get_active_directory()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def resolve(self, ref: TargetReference) -> ResolvedTarget:
        """Never raises — every expected condition maps to a
        ResolutionStatus; only a genuinely unexpected internal error
        falls back to INVALID_TARGET with a diagnostic message."""
        try:
            return self._resolve_inner(ref)
        except Exception as exc:  # absolute safety net
            return ResolvedTarget(
                original_target=getattr(ref, "original_target", ""),
                resolved_path=None,
                target_type=getattr(ref, "target_type", TargetType.UNKNOWN),
                status=ResolutionStatus.INVALID_TARGET,
                method=ResolutionMethod.NONE,
                existence_state=ExistenceState.UNKNOWN,
                error=f"Unexpected resolver error: {exc}",
            )

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _resolve_inner(self, ref: TargetReference) -> ResolvedTarget:
        if ref is None or not isinstance(ref.original_target, str) or not ref.original_target.strip():
            return ResolvedTarget(
                original_target=getattr(ref, "original_target", "") or "",
                resolved_path=None,
                target_type=TargetType.UNKNOWN,
                status=ResolutionStatus.INVALID_TARGET,
                method=ResolutionMethod.NONE,
                error="Empty or missing target.",
            )

        if "\x00" in ref.original_target:
            return ResolvedTarget(
                original_target=ref.original_target,
                resolved_path=None,
                target_type=ref.target_type,
                status=ResolutionStatus.INVALID_TARGET,
                method=ResolutionMethod.NONE,
                error="Target contains an invalid character.",
            )

        mode = ref.resolution_mode or ResolutionMode.UNSPECIFIED

        if mode == ResolutionMode.UNSPECIFIED:
            mode = self._classify_mode(ref)

        if mode == ResolutionMode.ABSOLUTE_PATH:
            result = self._resolve_absolute(ref)
        elif mode == ResolutionMode.RELATIVE_PATH:
            result = self._resolve_relative(ref)
        elif mode == ResolutionMode.WORKSPACE_RELATIVE:
            result = self._resolve_workspace_relative(ref)
        elif mode == ResolutionMode.NAMED_SCOPE:
            result = self._resolve_named_scope(ref)
        elif mode == ResolutionMode.SMART_SEARCH:
            result = self._resolve_smart_search(ref)
        elif mode == ResolutionMode.CONTEXT_REFERENCE:
            result = self._resolve_context_reference(ref)
        else:
            result = ResolvedTarget(
                original_target=ref.original_target,
                resolved_path=None,
                target_type=ref.target_type,
                status=ResolutionStatus.UNSUPPORTED_REFERENCE,
                method=ResolutionMethod.NONE,
                error=f"Unrecognized resolution_mode: {mode!r}",
            )

        return self._apply_action_policy(ref, result)

    def _classify_mode(self, ref: TargetReference) -> str:
        """
        The ONE deterministic classification step used when QUIPS does
        not specify a mode. This is NOT a guessing cascade (see module
        docstring / design spec section 5): it inspects the shape of
        the reference exactly once and picks exactly one mode based on
        unambiguous structural signals, in a fixed priority order. It
        never retries with a different mode after a failure.
        """
        if _looks_absolute(ref.original_target):
            return ResolutionMode.ABSOLUTE_PATH
        if ref.scope:
            return ResolutionMode.NAMED_SCOPE
        if self.active_workspace() is not None:
            return ResolutionMode.WORKSPACE_RELATIVE
        return ResolutionMode.RELATIVE_PATH

    # ------------------------------------------------------------------
    # B. Absolute targets
    # ------------------------------------------------------------------

    def _resolve_absolute(self, ref: TargetReference) -> ResolvedTarget:
        normalized = normalize_target_string(ref.original_target)
        return self._finalize(ref, normalized, ResolutionMethod.ABSOLUTE_PATH,
                               enforce_boundary=True)

    # ------------------------------------------------------------------
    # C. Relative targets
    # ------------------------------------------------------------------

    def _resolve_relative(self, ref: TargetReference) -> ResolvedTarget:
        base = (ref.context or {}).get("base_dir") or self.active_workspace()
        if not base:
            return ResolvedTarget(
                original_target=ref.original_target,
                resolved_path=None,
                target_type=ref.target_type,
                status=ResolutionStatus.MISSING_CONTEXT,
                method=ResolutionMethod.NONE,
                error="No base directory available to resolve a relative target "
                      "(no explicit context.base_dir and no active workspace).",
            )
        combined = _join_and_normalize(base, ref.original_target)
        # The boundary is always the active workspace — NOT `base` itself.
        # If boundary_root were `base`, an explicit context.base_dir that
        # sits outside the workspace would trivially pass its own check
        # (anything joined under X is, tautologically, "within X"). The
        # workspace is the actual security boundary regardless of which
        # base a relative path was anchored to; with no active workspace
        # at all, there is nothing to bound against (consistent with "do
        # not arbitrarily invent a base directory").
        workspace = self.active_workspace()
        return self._finalize(ref, combined, ResolutionMethod.RELATIVE_PATH,
                               enforce_boundary=bool(workspace), boundary_root=workspace)

    # ------------------------------------------------------------------
    # D. Workspace-relative targets
    # ------------------------------------------------------------------

    def _resolve_workspace_relative(self, ref: TargetReference) -> ResolvedTarget:
        workspace = self.active_workspace()
        if not workspace:
            return ResolvedTarget(
                original_target=ref.original_target,
                resolved_path=None,
                target_type=ref.target_type,
                status=ResolutionStatus.MISSING_CONTEXT,
                method=ResolutionMethod.NONE,
                error="No active workspace to resolve a workspace-relative target.",
            )
        combined = _join_and_normalize(workspace, ref.original_target)
        ambiguous = self._extension_collision_candidates(ref, combined, workspace)
        if ambiguous:
            return ResolvedTarget(
                original_target=ref.original_target,
                resolved_path=None,
                target_type=TargetType.UNKNOWN,
                status=ResolutionStatus.AMBIGUOUS,
                method=ResolutionMethod.WORKSPACE_RELATIVE,
                candidates=ambiguous,
                error=f"Multiple exact/stem matches for {ref.original_target!r}; "
                      "disambiguation required.",
                metadata={"workspace": workspace},
            )
        return self._finalize(ref, combined, ResolutionMethod.WORKSPACE_RELATIVE,
                               enforce_boundary=True, boundary_root=workspace)

    @staticmethod
    def _extension_collision_candidates(
        ref: TargetReference, normalized_path: str, workspace: str
    ) -> List[str]:
        """Preserve ambiguity between an exact path and same-stem files."""
        if ref.target_type != TargetType.UNKNOWN or Path(ref.original_target).suffix:
            return []
        if not _is_within(workspace, normalized_path):
            return []

        exact = Path(normalized_path)
        try:
            if not exact.exists():
                return []
            candidates = [str(exact)] if exact.is_file() or exact.is_dir() else []
            for sibling in exact.parent.iterdir():
                if (sibling != exact and sibling.is_file()
                        and sibling.stem.casefold() == exact.name.casefold()):
                    candidates.append(str(sibling))
        except OSError:
            return []
        return sorted(set(candidates)) if len(candidates) > 1 else []

    # ------------------------------------------------------------------
    # E. Named scope
    # ------------------------------------------------------------------

    def _resolve_named_scope(self, ref: TargetReference) -> ResolvedTarget:
        if not ref.scope:
            return ResolvedTarget(
                original_target=ref.original_target,
                resolved_path=None,
                target_type=ref.target_type,
                status=ResolutionStatus.MISSING_CONTEXT,
                method=ResolutionMethod.NONE,
                error="resolution_mode is NAMED_SCOPE but no scope was provided.",
            )

        scope_root = self._lookup_scope(ref.scope)
        if scope_root is None:
            return ResolvedTarget(
                original_target=ref.original_target,
                resolved_path=None,
                target_type=ref.target_type,
                status=ResolutionStatus.SCOPE_NOT_FOUND,
                method=ResolutionMethod.NONE,
                error=f"Scope {ref.scope!r} is not a registered scope, the active "
                      f"workspace, or a subdirectory of it.",
                metadata={"scope": ref.scope},
            )

        combined = _join_and_normalize(scope_root, ref.original_target)
        result = self._finalize(ref, combined, ResolutionMethod.NAMED_SCOPE,
                                 enforce_boundary=True, boundary_root=scope_root)
        result.metadata["scope"] = ref.scope
        result.metadata["scope_root"] = scope_root
        return result

    def _lookup_scope(self, name: str) -> Optional[str]:
        """Deterministic scope lookup — exact match only, three fixed
        sources checked in order, no fuzzy matching, no guessing."""
        name_lower = name.lower()

        # 1. Explicitly registered named scopes/projects.
        for key, path in self._scope_registry.items():
            if key.lower() == name_lower:
                return path

        workspace = self.active_workspace()
        if workspace:
            ws_name = _basename(workspace.rstrip("\\/"))
            # 2. The scope name IS the active workspace's own name.
            if ws_name.lower() == name_lower:
                return workspace
            # 3. A direct subdirectory of the active workspace.
            candidate = _join_and_normalize(workspace, name)
            try:
                if Path(candidate).is_dir():
                    return candidate
            except OSError:
                pass

        return None

    # ------------------------------------------------------------------
    # F. Smart target resolution (scoped, exact-name, non-fuzzy)
    # ------------------------------------------------------------------

    def _resolve_smart_search(self, ref: TargetReference) -> ResolvedTarget:
        roots: List[str] = list((ref.context or {}).get("search_roots") or [])
        if not roots:
            workspace = self.active_workspace()
            if workspace:
                roots = [workspace]

        if not roots:
            return ResolvedTarget(
                original_target=ref.original_target,
                resolved_path=None,
                target_type=ref.target_type,
                status=ResolutionStatus.MISSING_CONTEXT,
                method=ResolutionMethod.NONE,
                error="No search roots available (no context.search_roots and no "
                      "active workspace).",
            )

        target_name = _basename(ref.original_target)
        matches: List[str] = []
        for root in roots:
            root_path = Path(root)
            try:
                if not root_path.is_dir():
                    continue
            except OSError:
                continue
            matches.extend(self._scoped_walk(root_path, target_name, self._max_search_depth))

        unique_sorted = sorted(set(matches))

        if not unique_sorted:
            return ResolvedTarget(
                original_target=ref.original_target,
                resolved_path=None,
                target_type=ref.target_type,
                status=ResolutionStatus.NOT_FOUND,
                method=ResolutionMethod.SMART_SEARCH,
                existence_state=ExistenceState.NOT_EXISTS,
                error=f"No match for {target_name!r} found within {len(roots)} "
                      f"search root(s) (max depth {self._max_search_depth}).",
                metadata={"search_roots": roots},
            )

        if len(unique_sorted) > 1:
            return ResolvedTarget(
                original_target=ref.original_target,
                resolved_path=None,
                target_type=ref.target_type,
                status=ResolutionStatus.AMBIGUOUS,
                method=ResolutionMethod.SMART_SEARCH,
                candidates=unique_sorted,
                error=f"Multiple matches for {target_name!r}; disambiguation required.",
                metadata={"search_roots": roots},
            )

        result = self._finalize(ref, unique_sorted[0], ResolutionMethod.SMART_SEARCH,
                                 enforce_boundary=False)
        result.metadata["search_roots"] = roots
        return result

    def _scoped_walk(self, root: Path, name: str, max_depth: int) -> List[str]:
        """Depth-limited, exact (case-insensitive) filename match. No
        fuzzy matching, no ranking — see design spec section 13."""
        name_lower = name.lower()
        found: List[str] = []
        stack: List[Any] = [(root, 0)]
        while stack:
            current, depth = stack.pop()
            try:
                entries = list(current.iterdir())
            except OSError:
                continue
            for entry in entries:
                if entry.name.lower() == name_lower:
                    found.append(str(entry))
                if depth < max_depth:
                    try:
                        if entry.is_dir():
                            stack.append((entry, depth + 1))
                    except OSError:
                        continue
        return found

    # ------------------------------------------------------------------
    # H. Context reference
    # ------------------------------------------------------------------

    def _resolve_context_reference(self, ref: TargetReference) -> ResolvedTarget:
        """
        The resolver does NOT track conversation state itself (that
        remains QUIPS/ConversationManager's responsibility). It only
        accepts an already-dereferenced concrete value handed to it via
        `context`, and resolves THAT deterministically.
        """
        referenced = None
        if ref.context:
            referenced = ref.context.get("reference_path") or ref.context.get("referenced_target")

        if not referenced:
            return ResolvedTarget(
                original_target=ref.original_target,
                resolved_path=None,
                target_type=ref.target_type,
                status=ResolutionStatus.MISSING_CONTEXT,
                method=ResolutionMethod.NONE,
                error="resolution_mode is CONTEXT_REFERENCE but no concrete "
                      "context.reference_path / context.referenced_target was "
                      "supplied by the caller.",
            )

        delegated_ref = replace(ref, original_target=referenced)
        if _looks_absolute(referenced):
            result = self._resolve_absolute(delegated_ref)
        elif self.active_workspace():
            result = self._resolve_workspace_relative(delegated_ref)
        else:
            result = self._resolve_relative(delegated_ref)

        result.original_target = ref.original_target  # preserve the caller's wording
        result.method = ResolutionMethod.CONTEXT_REFERENCE
        result.metadata["dereferenced_to"] = referenced
        return result

    # ------------------------------------------------------------------
    # Shared finalize step: boundary check + existence/type detection
    # ------------------------------------------------------------------

    def _finalize(
        self,
        ref: TargetReference,
        normalized_path: str,
        method: str,
        *,
        enforce_boundary: bool,
        boundary_root: Optional[str] = None,
    ) -> ResolvedTarget:
        workspace = boundary_root or self.active_workspace()
        if enforce_boundary and workspace and not _is_within(workspace, normalized_path):
            return ResolvedTarget(
                original_target=ref.original_target,
                resolved_path=normalized_path,
                target_type=ref.target_type,
                status=ResolutionStatus.OUTSIDE_SCOPE,
                method=method,
                error=f"Resolved path falls outside the permitted workspace "
                      f"({workspace!r}).",
                metadata={"workspace": workspace},
            )

        existence_state, target_type = self._probe(normalized_path, ref.target_type)

        return ResolvedTarget(
            original_target=ref.original_target,
            resolved_path=normalized_path,
            target_type=target_type,
            status=ResolutionStatus.RESOLVED,
            method=method,
            existence_state=existence_state,
        )

    @staticmethod
    def _probe(normalized_path: str, type_hint: str) -> "tuple[str, str]":
        """
        Host-native existence probe. On a host whose native path
        dialect doesn't match `normalized_path` (e.g. a Windows-style
        string evaluated on a POSIX host), this correctly/harmlessly
        reports NOT_EXISTS rather than raising — see module docstring.
        """
        try:
            p = Path(normalized_path)
            if p.is_file():
                return ExistenceState.EXISTS_FILE, TargetType.FILE
            if p.is_dir():
                return ExistenceState.EXISTS_DIRECTORY, TargetType.DIRECTORY
            return ExistenceState.NOT_EXISTS, (type_hint or TargetType.UNKNOWN)
        except OSError:
            return ExistenceState.UNKNOWN, (type_hint or TargetType.UNKNOWN)

    # ------------------------------------------------------------------
    # H. Operation-aware existence policy (applied to every outcome)
    # ------------------------------------------------------------------

    def _apply_action_policy(self, ref: TargetReference, result: ResolvedTarget) -> ResolvedTarget:
        """
        Resolution and operation validation remain separate concerns:
        this is the ONLY place existence requirements are enforced, and
        it only ever downgrades an already-RESOLVED-but-absent result
        to NOT_FOUND for a small, explicit set of actions that
        structurally require the target to already exist. It never
        upgrades a failure, never invents a path, and never applies to
        AMBIGUOUS/SCOPE_NOT_FOUND/etc. results.
        """
        if result.status != ResolutionStatus.RESOLVED:
            return result
        if result.existence_state != ExistenceState.NOT_EXISTS:
            return result
        action = ref.action or ActionHint.UNSPECIFIED
        if action not in _EXISTENCE_REQUIRED_ACTIONS:
            return result

        result.status = ResolutionStatus.NOT_FOUND
        result.error = f"Target does not exist (required for action={action})."
        result.metadata["action_policy_applied"] = action
        return result