"""
target_resolution_models.py
============================
Structured contracts between QUIPS and the Target Resolver.

ROLE
----
This file defines data only. It contains NO resolution logic, NO
filesystem access, and NO natural-language parsing. Its entire job is
to give QUIPS and Target Resolver a shared, stable vocabulary so they
can evolve independently without either one re-interpreting the
other's output.

    QUIPS            "What does the user mean?"   -> produces TargetReference
    Target Resolver  "Where does that exist?"      -> produces ResolvedTarget
    Directory Control "Perform the operation."      -> consumes ResolvedTarget

Nothing in this file imports from QUIPS, Target Resolver, Directory
Control, or any other AG subsystem. It is intentionally a leaf module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# =============================================================================
# Target type — what kind of filesystem object is being referred to.
# =============================================================================

class TargetType:
    FILE = "FILE"
    DIRECTORY = "DIRECTORY"
    UNKNOWN = "UNKNOWN"


# =============================================================================
# Resolution mode — QUIPS's declaration of WHAT KIND of reference this is.
# This is a hint TO the resolver, not something the resolver infers by
# trying strategies until one works (see target_resolver.py's module
# docstring for why).
# =============================================================================

class ResolutionMode:
    ABSOLUTE_PATH = "ABSOLUTE_PATH"
    RELATIVE_PATH = "RELATIVE_PATH"
    WORKSPACE_RELATIVE = "WORKSPACE_RELATIVE"
    NAMED_SCOPE = "NAMED_SCOPE"
    SMART_SEARCH = "SMART_SEARCH"
    CONTEXT_REFERENCE = "CONTEXT_REFERENCE"
    UNSPECIFIED = "UNSPECIFIED"


# =============================================================================
# Resolution method — the resolver's record of HOW a target was actually
# resolved. Distinct from ResolutionMode: mode is the request going IN,
# method is the outcome coming OUT (they will usually match, but method
# is the ground truth for diagnostics).
# =============================================================================

class ResolutionMethod:
    ABSOLUTE_PATH = "ABSOLUTE_PATH"
    RELATIVE_PATH = "RELATIVE_PATH"
    WORKSPACE_RELATIVE = "WORKSPACE_RELATIVE"
    NAMED_SCOPE = "NAMED_SCOPE"
    SMART_SEARCH = "SMART_SEARCH"
    CONTEXT_REFERENCE = "CONTEXT_REFERENCE"
    NONE = "NONE"  # nothing could be applied (e.g. invalid/missing input)


# =============================================================================
# Resolution status — explicit structured outcomes. The resolver returns
# one of these instead of raising for any expected condition; an
# exception means something genuinely unexpected happened internally.
# =============================================================================

class ResolutionStatus:
    RESOLVED = "RESOLVED"
    NOT_FOUND = "NOT_FOUND"
    AMBIGUOUS = "AMBIGUOUS"
    INVALID_TARGET = "INVALID_TARGET"
    INVALID_SCOPE = "INVALID_SCOPE"
    SCOPE_NOT_FOUND = "SCOPE_NOT_FOUND"
    OUTSIDE_SCOPE = "OUTSIDE_SCOPE"
    MISSING_CONTEXT = "MISSING_CONTEXT"
    UNSUPPORTED_REFERENCE = "UNSUPPORTED_REFERENCE"


# =============================================================================
# Existence state — what the resolver actually observed on disk, kept
# separate from ResolutionStatus. A target can be RESOLVED and still
# NOT_EXIST (that is expected and required for CREATE-type operations).
# =============================================================================

class ExistenceState:
    EXISTS_FILE = "EXISTS_FILE"
    EXISTS_DIRECTORY = "EXISTS_DIRECTORY"
    NOT_EXISTS = "NOT_EXISTS"
    UNKNOWN = "UNKNOWN"  # could not be determined (e.g. permission error)


# =============================================================================
# Action hint — QUIPS's declaration of what the user intends to DO with
# the target. The resolver uses this ONLY to decide whether a missing
# target should be reported as a resolution failure (NOT_FOUND) or as a
# successfully resolved-but-absent path (fine for CREATE/WRITE/APPEND).
# It is intentionally a small, closed vocabulary — see
# target_resolver.py's _EXISTENCE_REQUIRED_ACTIONS for the policy table.
# =============================================================================

class ActionHint:
    ACCESS = "ACCESS"            # generic open/read, same existence policy as READ
    READ = "READ"
    LIST = "LIST"
    CREATE = "CREATE"
    WRITE = "WRITE"
    APPEND = "APPEND"
    DELETE = "DELETE"
    MOVE_SOURCE = "MOVE_SOURCE"
    MOVE_DESTINATION = "MOVE_DESTINATION"
    COPY_SOURCE = "COPY_SOURCE"
    COPY_DESTINATION = "COPY_DESTINATION"
    RENAME_SOURCE = "RENAME_SOURCE"
    RENAME_DESTINATION = "RENAME_DESTINATION"
    UNSPECIFIED = "UNSPECIFIED"


# =============================================================================
# TargetReference — what QUIPS hands to the Target Resolver.
# =============================================================================

@dataclass
class TargetReference:
    """
    Structured description of what the user is referring to, as
    determined by QUIPS. The resolver consumes this; it does not
    re-derive any of it from raw text.

    Fields
    ------
    original_target:
        The target exactly as QUIPS extracted it (e.g. "test.txt",
        "backup\\test.txt", "D:\\AG\\test.txt"). Never pre-mangled by
        QUIPS with surrounding natural-language filler.
    target_type:
        QUIPS's best declared guess at FILE vs DIRECTORY vs UNKNOWN.
        A hint — the resolver may refine this once it can see what's
        actually on disk.
    scope:
        A named scope/workspace/project QUIPS identified (e.g. "AG",
        "brains"), or None if no scope was referenced.
    resolution_mode:
        One of ResolutionMode.* — tells the resolver which strategy to
        apply. Required for deterministic behavior; UNSPECIFIED is
        supported but triggers a single deterministic classification
        step rather than a guessing cascade (see target_resolver.py).
    action:
        One of ActionHint.* — what the user intends to do with the
        target. Used only for existence-required policy (see above).
    context:
        Free-form supplementary information QUIPS has already resolved
        at the semantic level and wants to pass through, e.g.:
            {"base_dir": "..."}               for RELATIVE_PATH
            {"search_roots": ["..."]}         for SMART_SEARCH
            {"reference_path": "..."}         for CONTEXT_REFERENCE
        The resolver treats this as already-structured input, never as
        text to parse.
    raw_text:
        Optional original user utterance, carried through purely for
        diagnostics/logging. Never parsed by the resolver.
    """
    original_target: str
    target_type: str = TargetType.UNKNOWN
    scope: Optional[str] = None
    resolution_mode: str = ResolutionMode.UNSPECIFIED
    action: str = ActionHint.UNSPECIFIED
    context: Dict[str, Any] = field(default_factory=dict)
    raw_text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_target": self.original_target,
            "target_type": self.target_type,
            "scope": self.scope,
            "resolution_mode": self.resolution_mode,
            "action": self.action,
            "context": self.context,
            "raw_text": self.raw_text,
        }


# =============================================================================
# ResolvedTarget — what the Target Resolver hands to Directory Control.
# =============================================================================

@dataclass
class ResolvedTarget:
    """
    The structured outcome of resolving a TargetReference. Directory
    Control should trust this completely and never re-interpret
    `original_target` itself.

    Fields
    ------
    original_target:
        Carried through unchanged from the TargetReference, for
        traceability.
    resolved_path:
        The normalized, concrete path string, or None if resolution did
        not produce one (e.g. AMBIGUOUS, MISSING_CONTEXT).
    target_type:
        The resolver's best determination of FILE vs DIRECTORY vs
        UNKNOWN — based on actual existence when observable, falling
        back to the TargetReference's hint when the target does not
        (yet) exist.
    status:
        One of ResolutionStatus.* — the authoritative outcome.
    method:
        One of ResolutionMethod.* — how resolution was actually
        performed.
    existence_state:
        One of ExistenceState.* — what was actually observed on disk,
        independent of whether `status` counts that as a failure.
    candidates:
        Populated only when status == AMBIGUOUS: every candidate path
        found, for the caller (or a human) to disambiguate. Never
        auto-selected from.
    error:
        Short internal diagnostic string. Never fabricated detail —
        only set when something concrete is known (e.g. which scope
        name was not found). Not guaranteed to be user-facing-safe
        wording; callers should map status -> their own message.
    metadata:
        Additional diagnostics (e.g. which scope path matched, which
        search roots were used, whether an action-existence policy was
        applied). Safe to ignore; present for debugging.
    """
    original_target: str
    resolved_path: Optional[str]
    target_type: str
    status: str
    method: str
    existence_state: str = ExistenceState.UNKNOWN
    candidates: List[str] = field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_resolved(self) -> bool:
        return self.status == ResolutionStatus.RESOLVED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_target": self.original_target,
            "resolved_path": self.resolved_path,
            "target_type": self.target_type,
            "status": self.status,
            "method": self.method,
            "existence_state": self.existence_state,
            "candidates": self.candidates,
            "error": self.error,
            "metadata": self.metadata,
        }