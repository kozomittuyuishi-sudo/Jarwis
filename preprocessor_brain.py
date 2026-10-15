"""
preprocessor_brain.py
======================
AG Preprocessor Brain — deterministic offline processing layer.

WHAT THIS IS
------------
The Preprocessor Brain sits in front of AG's LLM brains (Local Brain /
Cloud Brain) and handles the requests that do NOT require an LLM at all:

    - "What are your capabilities?"          (self-info)
    - "What is my current directory?"        (directory)
    - "Read brain.py."                       (directory)
    - "Create a file called X and write: Y"  (directory)

It is NOT an LLM, NOT a chatbot, and NOT a replacement for the Local or
Cloud Brain. It is a fast, offline, deterministic classifier/parser/
executor that either:

    (a) handles the request completely on its own and returns a
        structured, machine-readable result, or
    (b) declares that it cannot handle the request and that deeper LLM
        reasoning is required (``requires_llm=True``).

THE CORE PRINCIPLE
-------------------
    LLM:            "Here is what I think the user means."
    Preprocessor:   "Here is the structured operation I recognize."
    Local subsystem:"Here is what actually happened."
    Jarwis:             "Here is the human-readable response."

The Preprocessor Brain is the middle two steps. It NEVER lets an LLM
perform or claim to perform a filesystem action, and it NEVER guesses
at a dangerous/irreversible operation from a weak signal — when it is
not confident, it says so (UNKNOWN + clarification) instead of acting.

SCOPE OF THIS FILE (Phase 1 — first five features only)
---------------------------------------------------------
    1. Self-capability / self-info processing   -> CapabilityRegistry
    2. Request classification                   -> RequestClassifier
    3. Intent detection                          -> IntentDetector
    4. Command parsing (NL -> structured command) -> CommandParser
    5. Directory control integration              -> DirectoryOperations /
                                                       DirectoryOperationExecutor

Everything else described in the Preprocessor Brain design doc (path
resolution as its own layer, device control, LLM request preparation,
etc.) is intentionally OUT of scope for this file and is left as
clearly-marked extension points.

INTEGRATION
-----------
AG (or ag_pipeline.py) is expected to eventually do something like::

    processor = PreprocessorBrain(directory_control=existing_directory_control,
                                   introspection_source=existing_introspection_engine)
    result = processor.process(user_input)

    if result.requires_llm:
        # fall through to Local/Cloud Brain as today
        ...
    else:
        # result.success / result.result / result.error is the truth
        ...

If no real ``DirectoryControl`` instance is supplied, a minimal, fully
self-contained fallback (``DirectoryOperations``) is used instead so
that this module can be developed, run, and tested completely on its
own (see the ``__main__`` block at the bottom of this file) without
importing anything from the rest of the AG codebase.

DESIGN RULES
------------
- No LLM calls anywhere in this file. No network calls. No shell
  commands. pathlib / stdlib only.
- Never raises out of the public API — every internal error becomes a
  PreprocessorResult with success=False and a structured error code.
- Error domains are kept separate (SELF_INFO / DIRECTORY /
  PREPROCESSOR / FILESYSTEM), mirroring AG's existing
  fallback_classifier.py error-isolation principle, so a failed
  filesystem operation can never be mistaken for a brain failure.
- Windows and POSIX paths both work (pathlib-based); no Linux-only
  assumptions, no shell-outs.
"""

from __future__ import annotations

import re
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# =============================================================================
# SECTION 0 — Domain / Intent / Capability constants
# =============================================================================

class Domain:
    """Top-level request domains (Feature 2)."""
    SELF_INFO = "SELF_INFO"
    DIRECTORY = "DIRECTORY"
    DEVICE = "DEVICE"          # not implemented yet — reserved
    GENERAL_LLM = "GENERAL_LLM"
    UNKNOWN = "UNKNOWN"


class Intent:
    """Structured intents (Feature 3)."""
    # --- Self-info intents ---
    GET_IDENTITY = "GET_IDENTITY"
    GET_CAPABILITIES = "GET_CAPABILITIES"
    GET_LIMITATIONS = "GET_LIMITATIONS"
    GET_ARCHITECTURE = "GET_ARCHITECTURE"
    GET_BRAIN_INFO = "GET_BRAIN_INFO"

    # --- Directory intents ---
    CURRENT_DIRECTORY = "CURRENT_DIRECTORY"
    LIST_DIRECTORY = "LIST_DIRECTORY"
    READ_FILE = "READ_FILE"
    CREATE_FILE = "CREATE_FILE"
    CREATE_FOLDER = "CREATE_FOLDER"
    WRITE_FILE = "WRITE_FILE"
    APPEND_FILE = "APPEND_FILE"
    SWITCH_DIRECTORY = "SWITCH_DIRECTORY"
    # NAVIGATE is the deterministic navigation contract (Phase 2).
    # It handles "go to", "navigate to", "open", "move into", "..", "parent",
    # relative paths, and absolute paths — updating the one authoritative
    # active-directory state in DirectoryControl.
    NAVIGATE = "NAVIGATE"
    INSPECT = "INSPECT"
    RENAME = "RENAME"
    COPY = "COPY"
    MOVE = "MOVE"
    DELETE = "DELETE"
    SEARCH = "SEARCH"

    UNKNOWN = "UNKNOWN"


class CapabilityState:
    """Verified capability states (Feature 1)."""
    SUPPORTED = "SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    PARTIAL = "PARTIAL"
    UNAVAILABLE = "UNAVAILABLE"


class ErrorDomain:
    """Isolated error domains — never conflated with each other."""
    SELF_INFO = "SELF_INFO"
    PREPROCESSOR = "PREPROCESSOR"
    DIRECTORY_CONTROL = "DIRECTORY_CONTROL"
    FILESYSTEM = "FILESYSTEM"


# Directory error codes — deliberately mirror the codes already used by
# AG's fallback_classifier.py / directory_control.py so a downstream
# caller that already understands those codes needs no translation layer.
class DirErrorCode:
    NOT_FOUND = "NOT_FOUND"
    NOT_A_FILE = "NOT_A_FILE"
    NOT_A_DIRECTORY = "NOT_A_DIRECTORY"
    SOURCE_MISSING = "SOURCE_MISSING"
    OUT_OF_WORKSPACE = "OUT_OF_WORKSPACE_BOUNDS"
    NO_ACTIVE_WORKSPACE = "NO_ACTIVE_WORKSPACE"
    ALREADY_EXISTS = "ALREADY_EXISTS"
    OPERATION_FAILED = "OPERATION_FAILED"
    CAPABILITY_UNAVAILABLE = "CAPABILITY_UNAVAILABLE"
    AMBIGUOUS_REQUEST = "AMBIGUOUS_REQUEST"


# =============================================================================
# SECTION 1 — Structured result object
# =============================================================================

@dataclass
class PreprocessorResult:
    """
    Machine-readable outcome of PreprocessorBrain.process().

    This is the ONLY thing AG should trust as "what actually happened" —
    never a string the LLM produced.
    """
    success: bool
    domain: str
    intent: str
    operation: Optional[str] = None
    target: Optional[str] = None
    result: Any = None
    error: Optional[str] = None
    error_domain: Optional[str] = None
    confidence: float = 0.0
    requires_llm: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "domain": self.domain,
            "intent": self.intent,
            "operation": self.operation,
            "target": self.target,
            "result": self.result,
            "error": self.error,
            "error_domain": self.error_domain,
            "confidence": self.confidence,
            "requires_llm": self.requires_llm,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


@dataclass
class ClassificationResult:
    domain: str
    confidence: float
    reason: str = ""


@dataclass
class IntentResult:
    intent: str
    confidence: float
    reason: str = ""


@dataclass
class ParsedCommand:
    """Structured command produced by Feature 4 (CommandParser)."""
    domain: str
    intent: str
    target: Optional[str] = None
    source: Optional[str] = None
    destination: Optional[str] = None
    content: Optional[str] = None
    args: Dict[str, Any] = field(default_factory=dict)
    complete: bool = True
    missing_fields: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "intent": self.intent,
            "target": self.target,
            "source": self.source,
            "destination": self.destination,
            "content": self.content,
            "args": self.args,
        }


# =============================================================================
# SECTION 2 — Small text utilities shared by the classifier / parser
# =============================================================================

# Punctuation that is safe to strip from the END of an extracted
# target/path/filename (sentence punctuation the user typed, not part of
# the actual filesystem name). NEVER applied to extracted `content`.
_TRAILING_PUNCT = ".,!?;:\"')]}\u2019\u201d"
_LEADING_PUNCT = "\"'(\u2018\u201c"


def _strip_target_punct(value: str) -> str:
    value = value.strip()
    while value and value[-1] in _TRAILING_PUNCT:
        value = value[:-1]
    while value and value[0] in _LEADING_PUNCT:
        value = value[1:]
    return value.strip()


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


# A filler-word set used ONLY as a last-resort fallback when no stronger
# structural signal (quotes, "called X", explicit extension) is present.
_FILLER_WORDS = {
    "a", "an", "the", "please", "file", "folder", "directory", "called",
    "named", "titled", "with", "name", "of",
}


def _last_resort_target(remainder: str) -> str:
    """
    Fallback target extraction: strip leading filler/verb words token by
    token until something meaningful-looking remains. Used only when no
    quoted string, "called X", or extension pattern was found — this is
    intentionally conservative so it never eats real path content.
    """
    tokens = remainder.strip().split()
    while tokens and tokens[0].lower().strip(".,!?;:") in _FILLER_WORDS:
        tokens.pop(0)
    return _strip_target_punct(" ".join(tokens))


# Matches a filename or path ending in a plausible extension, including
# windows drive-letter paths and posix/nested paths. This is the single
# strongest, least-ambiguous signal for "this token is a path".
_PATH_WITH_EXTENSION_RE = re.compile(
    r"(?:[A-Za-z]:[\\/]|\\\\|/|\.{1,2}[/\\]|~[/\\])?"    # optional drive/UNC/rel/home prefix
    r"[\w.\-]+(?:[/\\][\w.\-]+)*"                        # path segments
    r"\.[A-Za-z0-9]{1,8}"                                # extension
)

# A path with NO required extension — used for directory targets, which
# (unlike files) usually have no extension at all. Deliberately requires
# either a drive letter or an actual path separator, so it only ever
# matches something that structurally looks like a path, never a bare
# English word like "brains" or "workspace".
_EXPLICIT_DIR_PATH_RE = re.compile(
    r"[A-Za-z]:[\\/][^\s,.;!?]*(?:[\\/][^\s,.;!?]*)*"    # C:\... or C:/...
    r"|(?:\.{1,2}|~)?/[^\s,.;!?]+(?:/[^\s,.;!?]+)*"        # /a/b, ./a/b, ~/a/b
    r"|[\w.\-]+(?:[\\/][\w.\-]+)+"                        # a/b/c (relative, multi-segment)
)

# Sentinel value meaning "the request refers to the workspace/directory
# that is already active" — e.g. "list the current workspace", "read
# files in the current workspace" — as opposed to a specific named
# sub-path. Kept distinguishable from a literal "." or a real path so
# both the parser's intent and any caller inspecting ParsedCommand.target
# can tell the two apart without re-parsing English.
ACTIVE_WORKSPACE = "ACTIVE_WORKSPACE"

# Phrases that refer to the active workspace itself rather than naming a
# specific sub-directory — never treated as a literal path/filename.
_ACTIVE_WORKSPACE_PHRASES = {
    "everything", "the files", "files", "contents", "the contents",
    "the current directory", "the active directory", "this directory",
    "this workspace", "here", "the directory", "the folder", "the workspace",
    "current workspace", "the current workspace", "active workspace",
    "the active workspace", "my workspace", "this folder", "workspace",
}

_CALLED_NAMED_RE = re.compile(
    r"\b(?:called|named|titled)\s+"
    r"(?P<quoted>\"[^\"]+\"|'[^']+')"
    r"|"
    r"\b(?:called|named|titled)\s+(?P<bare>[^\s]+)",
    re.IGNORECASE,
)

_QUOTED_RE = re.compile(r"\"([^\"]+)\"|'([^']+)'")


def extract_named_target(zone: str) -> Optional[str]:
    """
    Extract a filename/foldername/path from a text zone using — in order
    of confidence — quoted strings, "called/named X", an explicit
    extension pattern, then nothing (caller decides on a last-resort
    fallback). This is the fix for the historical bug where filler text
    like "called directory_test.txt and write:" was captured whole as a
    filename instead of just "directory_test.txt".
    """
    if not zone or not zone.strip():
        return None

    m = _CALLED_NAMED_RE.search(zone)
    if m:
        if m.group("quoted"):
            return _strip_quotes(m.group("quoted"))
        if m.group("bare"):
            return _strip_target_punct(m.group("bare"))

    q = _QUOTED_RE.search(zone)
    if q:
        return q.group(1) if q.group(1) is not None else q.group(2)

    ext = _PATH_WITH_EXTENSION_RE.search(zone)
    if ext:
        return _strip_target_punct(ext.group(0))

    return None


# Markers that separate a "target description" zone from a "content"
# zone, e.g. "...called test.txt AND WRITE: hello" — everything at/after
# the marker is content, never part of the filename.
_CONTENT_MARKER_RE = re.compile(
    r"\b(?:and\s+write|and\s+add|and\s+append|write|add|append|containing|"
    r"with\s+(?:the\s+)?content(?:s)?|contents?)\s*:?\s*",
    re.IGNORECASE,
)


def split_target_and_content(raw: str) -> Tuple[str, Optional[str]]:
    """
    Split raw input into (target_zone, content_or_None) using the first
    content marker found AFTER the first quarter of the string (so that a
    leading "Write ..." verb doesn't itself get mistaken for the split
    point when the sentence begins with it).
    """
    matches = list(_CONTENT_MARKER_RE.finditer(raw))
    if not matches:
        return raw, None

    # Prefer a marker that looks like "and write:" / "write:" occurring
    # after some target description; if the very first token IS the verb
    # (e.g. "Write 'hello' to test.txt"), that case is handled by the
    # dedicated WRITE_FILE parser instead, so here we just take the LAST
    # marker match — content markers rarely repeat, and the last one is
    # the one immediately preceding the actual content in "create ...
    # and write: ..." style commands.
    m = matches[-1]
    target_zone = raw[: m.start()]
    content_zone = raw[m.end():].strip()
    if not content_zone:
        content_zone = None
    return target_zone, content_zone


# =============================================================================
# SECTION 3 — Feature 1: Self-Capability & Self-Info Processing
# =============================================================================

@dataclass
class Capability:
    name: str
    domain: str
    state: str
    description: str = ""


class CapabilityRegistry:
    """
    The single verified source of truth for "what can AG actually do"
    that the Preprocessor Brain (and, transitively, the LLM prompt layer)
    is allowed to state.

    It NEVER invents capabilities. Directory capabilities are derived by
    checking, at call time, whether the attached directory-operations
    object actually exposes the corresponding method — so if a method is
    added or removed from DirectoryControl, this registry's answers
    change automatically with zero duplication of truth.

    If AG already has a live IntrospectionEngine / RuntimeSnapshot, pass
    it in as ``introspection_source`` (either the engine, an object with
    a ``.to_dict()`` method shaped like RuntimeSnapshot.to_dict(), or a
    plain dict with "capabilities" / "limitations" keys). Its verified
    strings are folded in as SUPPLEMENTARY evidence, never used to
    contradict the directly-verified method-presence checks below.
    """

    # intent -> the directory-operations method name that implements it
    _DIRECTORY_INTENT_METHODS: Dict[str, str] = {
        Intent.CURRENT_DIRECTORY: "get_active_directory",
        Intent.LIST_DIRECTORY: "list_directory",
        Intent.READ_FILE: "read_file",
        Intent.CREATE_FILE: "create_file",
        Intent.CREATE_FOLDER: "create_directory",
        Intent.WRITE_FILE: "write_file",
        Intent.APPEND_FILE: "append_file",
        Intent.SWITCH_DIRECTORY: "switch_directory",
        Intent.NAVIGATE: "navigate_directory",
        Intent.INSPECT: "inspect_target",
        Intent.RENAME: "rename",
        Intent.COPY: "copy",
        Intent.MOVE: "move",
        Intent.DELETE: "delete",          # intentionally NOT implemented today
        Intent.SEARCH: "search_files",
    }

    _FRIENDLY_NAMES: Dict[str, str] = {
        Intent.CURRENT_DIRECTORY: "tell you the active directory",
        Intent.LIST_DIRECTORY: "list a directory's contents",
        Intent.READ_FILE: "read files",
        Intent.CREATE_FILE: "create files",
        Intent.CREATE_FOLDER: "create folders",
        Intent.WRITE_FILE: "write file contents",
        Intent.APPEND_FILE: "append to files",
        Intent.SWITCH_DIRECTORY: "switch the active directory",
        Intent.NAVIGATE: "navigate to a directory",
        Intent.INSPECT: "inspect a file or directory",
        Intent.RENAME: "rename files/folders",
        Intent.COPY: "copy files",
        Intent.MOVE: "move files/folders",
        Intent.DELETE: "delete files/folders",
        Intent.SEARCH: "search for files",
    }

    def __init__(
        self,
        directory_ops: Optional[Any] = None,
        introspection_source: Optional[Any] = None,
    ) -> None:
        self._directory_ops = directory_ops
        self._introspection_source = introspection_source

    def attach_directory_ops(self, directory_ops: Any) -> None:
        self._directory_ops = directory_ops

    # ------------------------------------------------------------------

    def directory_capability(self, intent: str) -> Capability:
        method_name = self._DIRECTORY_INTENT_METHODS.get(intent)
        friendly = self._FRIENDLY_NAMES.get(intent, intent)

        if method_name is None:
            return Capability(
                name=intent, domain=Domain.DIRECTORY,
                state=CapabilityState.UNAVAILABLE,
                description=f"No known directory operation maps to '{intent}'.",
            )

        if self._directory_ops is None:
            return Capability(
                name=intent, domain=Domain.DIRECTORY,
                state=CapabilityState.UNAVAILABLE,
                description="No directory subsystem is currently connected, "
                            f"so I can't verify whether I can {friendly}.",
            )

        method = getattr(self._directory_ops, method_name, None)
        if callable(method):
            return Capability(
                name=intent, domain=Domain.DIRECTORY,
                state=CapabilityState.SUPPORTED,
                description=f"Yes — I can {friendly} "
                             f"(implemented via {method_name}()).",
            )

        return Capability(
            name=intent, domain=Domain.DIRECTORY,
            state=CapabilityState.UNSUPPORTED,
            description=f"No — I cannot {friendly} yet "
                        f"(no {method_name}() implementation is connected).",
        )

    def all_directory_capabilities(self) -> List[Capability]:
        return [self.directory_capability(i) for i in self._DIRECTORY_INTENT_METHODS]

    def summarize_directory_capabilities(self) -> Dict[str, List[str]]:
        buckets: Dict[str, List[str]] = {
            CapabilityState.SUPPORTED: [],
            CapabilityState.UNSUPPORTED: [],
            CapabilityState.PARTIAL: [],
            CapabilityState.UNAVAILABLE: [],
        }
        for cap in self.all_directory_capabilities():
            buckets[cap.state].append(cap.name)
        return buckets

    # ------------------------------------------------------------------
    # Introspection integration (supplementary, never authoritative over
    # the directly-verified method checks above)
    # ------------------------------------------------------------------

    def _resolve_introspection_dict(self) -> Optional[Dict[str, Any]]:
        src = self._introspection_source
        if src is None:
            return None
        try:
            if callable(src) and not hasattr(src, "to_dict"):
                src = src()
            if hasattr(src, "build_snapshot") and not hasattr(src, "to_dict"):
                src = src.build_snapshot()
            if hasattr(src, "to_dict"):
                return src.to_dict()
            if isinstance(src, dict):
                return src
        except Exception:
            return None
        return None

    def introspection_capabilities(self) -> List[str]:
        d = self._resolve_introspection_dict()
        if not d:
            return []
        return list(d.get("capabilities", []))

    def introspection_limitations(self) -> List[str]:
        d = self._resolve_introspection_dict()
        if not d:
            return []
        return list(d.get("limitations", []))

    # ------------------------------------------------------------------
    # Human-readable answer builders
    # ------------------------------------------------------------------

    def describe_capabilities(self) -> str:
        buckets = self.summarize_directory_capabilities()
        lines: List[str] = []
        if buckets[CapabilityState.SUPPORTED]:
            supported = ", ".join(
                self._FRIENDLY_NAMES.get(i, i) for i in buckets[CapabilityState.SUPPORTED]
            )
            lines.append(f"I can currently: {supported}.")
        if buckets[CapabilityState.UNSUPPORTED]:
            unsupported = ", ".join(
                self._FRIENDLY_NAMES.get(i, i) for i in buckets[CapabilityState.UNSUPPORTED]
            )
            lines.append(f"Not yet implemented: {unsupported}.")
        if buckets[CapabilityState.UNAVAILABLE] and self._directory_ops is None:
            lines.append("No directory subsystem is currently connected.")
        extra = self.introspection_capabilities()
        if extra:
            lines.append("Additional verified capabilities: " + "; ".join(extra[:12]))
        if not lines:
            lines.append("I don't currently have verified capability information available.")
        return " ".join(lines)

    def describe_limitations(self) -> str:
        buckets = self.summarize_directory_capabilities()
        lines: List[str] = []
        unsupported = buckets[CapabilityState.UNSUPPORTED] + buckets[CapabilityState.UNAVAILABLE]
        if unsupported:
            named = ", ".join(self._FRIENDLY_NAMES.get(i, i) for i in unsupported)
            lines.append(f"Current limitations: {named}.")
        extra = self.introspection_limitations()
        if extra:
            lines.append("Additional verified limitations: " + "; ".join(extra[:8]))
        if not lines:
            lines.append("No specific limitations are currently recorded.")
        return " ".join(lines)


# Static, verified architecture facts (Section 2/3/4 of the design doc).
# These describe what has ACTUALLY been built, per the current design
# doc, not invented behavior. Override via PreprocessorBrain(...) kwargs
# if AG's real architecture facts should take precedence at runtime.
DEFAULT_BRAIN_INFO: Dict[str, Any] = {
    "local_brain": "Ollama-based, runs locally, currently uses qwen2.5:3b, "
                   "no internet required.",
    "cloud_brain": "OpenRouter-based external LLM, used for deeper/general "
                   "reasoning, includes adaptive preparation/validation/retry logic.",
    "mode": "Auto mode: local first, cloud fallback.",
}

DEFAULT_ARCHITECTURE_INFO: str = (
    "AG has two evolutionary execution paths: (1) the primary live path — "
    "Jarwis.py -> intent routing -> brain -> response, and (2) the formal "
    "pipeline — ag_pipeline.py -> CognitiveEngine -> BrainDispatcher -> "
    "ResponseProcessor -> ExecutiveLayer. The Preprocessor Brain is a "
    "deterministic offline processing layer that sits in front of both "
    "brain paths, handling self-info and directory requests without "
    "requiring an LLM call."
)


class SelfInfoProcessor:
    """Answers self-info questions using ONLY verified data (Feature 1)."""

    _INTENT_PATTERNS: List[Tuple[re.Pattern, str]] = [
        (re.compile(r"\b(?:limitation\w*|don't.*support|do not.*support|unsupported)\b"), Intent.GET_LIMITATIONS),
        (re.compile(r"\barchitecture\b|\bhow (?:are|is) you (?:built|structured|designed)\b|"
                     r"\bpipeline\b|\bhow (?:do )?you work\b|\bwhat happens internally\b|"
                     r"\bwhat systems do you have\b"), Intent.GET_ARCHITECTURE),
        (re.compile(r"\bbrains?\b"), Intent.GET_BRAIN_INFO),
        (re.compile(r"\bcapabilit|\bcapable of\b|\bwhat can you (?:currently )?do\b|"
                    r"\bwhat you can do\b|\bwhat functionality do you have\b|"
                    r"\bwhat functions? (?:do you have|do you support)\b|"
                    r"\bwhat can you help me with\b|\bare you able to\b|\bcan you\b"),
         Intent.GET_CAPABILITIES),
        (re.compile(r"\bwho you are\b|\babout yourself\b|\bknow about yourself\b|\byour identity\b"),
         Intent.GET_IDENTITY),
    ]

    def __init__(self, capability_registry: CapabilityRegistry,
                 brain_info: Optional[Dict[str, Any]] = None,
                 architecture_info: Optional[str] = None):
        self.capabilities = capability_registry
        self.brain_info = brain_info or DEFAULT_BRAIN_INFO
        self.architecture_info = architecture_info or DEFAULT_ARCHITECTURE_INFO

    def detect_intent(self, text_lower: str) -> IntentResult:
        for pattern, intent in self._INTENT_PATTERNS:
            if pattern.search(text_lower):
                return IntentResult(intent=intent, confidence=0.9,
                                     reason=f"Matched pattern for {intent}.")
        return IntentResult(intent=Intent.GET_CAPABILITIES, confidence=0.5,
                             reason="Self-info domain with no specific sub-intent; "
                                    "defaulting to capabilities overview.")

    def answer(self, intent: str) -> str:
        if intent == Intent.GET_IDENTITY:
            from self_info import SelfInfoEngine

            return SelfInfoEngine().answer("who are you").context_block
        if intent == Intent.GET_LIMITATIONS:
            return self.capabilities.describe_limitations()
        if intent == Intent.GET_ARCHITECTURE:
            return self.architecture_info
        if intent == Intent.GET_BRAIN_INFO:
            return (f"Local Brain: {self.brain_info.get('local_brain')} "
                    f"Cloud Brain: {self.brain_info.get('cloud_brain')} "
                    f"{self.brain_info.get('mode', '')}").strip()
        # default / GET_CAPABILITIES
        return self.capabilities.describe_capabilities()


# =============================================================================
# SECTION 4 — Feature 2: Request Classification
# =============================================================================

class RequestClassifier:
    """
    Deterministic domain classifier. Self-info and directory patterns are
    checked first and strongly implemented, per spec; everything else
    either falls through to GENERAL_LLM (well-formed but non-deterministic
    question) or UNKNOWN (not enough signal to say anything at all).
    """

    _SELF_INFO_PATTERNS: List[re.Pattern] = [re.compile(p, re.IGNORECASE) for p in [
        r"\bwho you are\b|\btell me about yourself\b|\bknow about yourself\b|\byour identity\b",
        r"\bwhat are your capabilit",
        r"\bwhat can you (?:currently )?do\b|\bwhat you can do\b",
        r"\bwhat functionality do you have\b|\bwhat functions? (?:do you have|do you support)\b",
        r"\bwhat can you help me with\b",
        r"\bcan you (?:read|create|write|edit|delete|remove|handle|automate|manage|switch)\b",
        r"\byour (?:capabilities|limitations|architecture)\b",
        r"\bwhat (?:are you|you're|you are) (?:capable|able) of\b",
        r"\bwhat operations? .*?(?:don't|do not).*support\b",
        r"\b(?:how you work(?: internally)?|what happens internally|what systems do you have)\b",
        r"\bwhat is your architecture\b",
        r"\bhow (?:are|is) you (?:built|structured|designed)\b",
        r"\b(?:which|what) brains?\b|\bhow many brains\b",
        r"\btell me about yourself\b",
        r"\bwhat are your limitations\b",
        r"\bdo you support\b",
    ]]

    # Verb-anchored / structural patterns for directory intents — checked
    # BEFORE the generic noun fallback below, so an explicit action is
    # always preferred over a vague noun match.
    _DIRECTORY_PATTERNS: List[re.Pattern] = [re.compile(p, re.IGNORECASE) for p in [
        r"^\s*(?:list|show)\b",
        r"\b(?:current|active)\s+(?:directory|workspace)\b",
        r"\bwhat (?:directory|folder)\b.*\b(?:am i in|are we in)\b|"
        r"\bwhat directory\b|\bwhat workspace\b|\bworking in\b|\bwhere am i\b|\bwhere are you\b",
        r"\bset\s+(?:the\s+)?workspace\s+to\b|\bswitch\s+workspace(?:s)?\s+to\b|\bswitch to\b|"
        r"\bchange (?:the )?(?:directory|workspace) to\b|^\s*cd\s|\bwork\s+in\b|"
        r"\bmake\b.*\bmy\s+workspace\b",
        r"\buse\s+(?:[A-Za-z]:[\\/]|\.{1,2}[\\/]|~[\\/]|/)",
        # --- NAVIGATE patterns (Phase 2) ---
        # "go to", "navigate to", "move into", "move to", "open" as navigation
        r"\bgo\s+to\b",
        r"\bnavigate\s+to\b",
        r"\bmove\s+into\b",
        r"\bmove\s+to\b",
        r"\bopen\b.*\b(?:folder|directory)\b",
        # Parent / current navigation
        r"\bgo\s+(?:up|back)\b",
        r"\bgo\s+to\s+(?:\.\.|parent)\b",
        r"\bnavigate\s+to\s+(?:parent|\.\.)\b",
        r"\bmove\s+to\s+(?:parent|\.\.)\b",
        # --- end NAVIGATE patterns ---
        r"\brename\b",
        r"\bcopy\b.*\bto\b|^\s*cp\s",
        r"\bmove\b.*\bto\b|^\s*mv\s",
        r"\bdelete\b|\bremove\b|^\s*rm\s",
        r"\bappend\b",
        r"\bcreate\b.*\b(?:folder|directory)\b|\bmake\b.*\b(?:folder|directory)\b|\bmkdir\b",
        r"\bcreate\b.*\bfile\b|\bmake\b.*\bfile\b",
        r"\bwrite\b",
        r"\bsearch\b|\bfind\b.*\bfile\b",
        r"\bread\b|\bopen file\b|^\s*cat\s",
        # --- INSPECT patterns ---
        r"\binspect\b",
        r"\bshow\s+(?:information|info)\s+about\b",
        r"\btell\s+me\s+about\b.*\b(?:\.[a-z0-9]{1,8}|folder|directory|file)\b",
    ]]

    _DIRECTORY_NOUNS = re.compile(r"\b(?:file|folder|directory|workspace|path)\b", re.IGNORECASE)

    def classify(self, user_input: str) -> ClassificationResult:
        text = (user_input or "").strip()
        if not text:
            return ClassificationResult(Domain.UNKNOWN, 0.0, "Empty input.")

        for pattern in self._SELF_INFO_PATTERNS:
            if pattern.search(text):
                return ClassificationResult(Domain.SELF_INFO, 0.9,
                                             f"Matched self-info pattern: {pattern.pattern}")

        for pattern in self._DIRECTORY_PATTERNS:
            if pattern.search(text):
                return ClassificationResult(Domain.DIRECTORY, 0.9,
                                             f"Matched directory pattern: {pattern.pattern}")

        if self._DIRECTORY_NOUNS.search(text):
            # A directory-related noun is present but no confident action
            # verb/pattern matched — flag DIRECTORY at low confidence so
            # intent detection can (correctly) come back UNKNOWN and
            # trigger a clarification instead of guessing an operation.
            return ClassificationResult(Domain.DIRECTORY, 0.35,
                                         "Directory-related noun present without a "
                                         "confident action pattern.")

        if len(text.split()) >= 2 and re.search(r"[A-Za-z]{3,}", text):
            return ClassificationResult(Domain.GENERAL_LLM, 0.55,
                                         "No deterministic domain matched; routable "
                                         "to general LLM reasoning.")

        return ClassificationResult(Domain.UNKNOWN, 0.2, "Insufficient signal to classify.")


# =============================================================================
# SECTION 5 — Feature 3: Intent Detection
# =============================================================================

class IntentDetector:
    """
    Deterministic, ORDER-SENSITIVE intent detection for the DIRECTORY
    domain. Order matters — see inline comments — because natural
    language commands often contain multiple candidate keywords (e.g.
    "create a file ... and write: ..." contains both "create" and
    "write"; the more specific / more dangerous operation must win).
    """

    _RULES: List[Tuple[re.Pattern, str]] = [
        # 0. INSPECT — if the input literally starts with "inspect" (or a
        #    recognised "show information about / tell me about" variant),
        #    it is unambiguously an inspection request regardless of what
        #    comes after it ("inspect the current workspace",
        #    "inspect this directory", "inspect .").
        #    This rule MUST be first so it cannot be shadowed by the
        #    CURRENT_DIRECTORY or LIST_DIRECTORY rules below that also
        #    match on "current workspace" and "this directory".
        (re.compile(r"^\s*inspect\b", re.IGNORECASE), Intent.INSPECT),
        (re.compile(r"^\s*show\s+(?:information|info)\s+about\b", re.IGNORECASE), Intent.INSPECT),
        (re.compile(
            r"^\s*tell\s+me\s+about\b.*\b(?:\.[a-z0-9]{1,8}|folder|directory|file)\b",
            re.IGNORECASE,
        ), Intent.INSPECT),

        # 1. Unambiguous "list contents" phrasing always wins first —
        #    this must be narrow (verb + files/contents/everything, or an
        #    explicit "what's in"), NOT a bare list/show verb, so it can
        #    never shadow a workspace-location query like "show my active
        #    workspace" (see rule 3 below; the bare list/show catch-all
        #    is intentionally the LAST rule in this table).
        (re.compile(r"\blist (?:the )?(?:files|contents|everything)\b", re.IGNORECASE),
         Intent.LIST_DIRECTORY),
        (re.compile(r"^\s*(?:list|show)\s+(?:me\s+)?(?:the\s+)?(?:files|contents|everything)\b",
                     re.IGNORECASE), Intent.LIST_DIRECTORY),
        (re.compile(r"\bwhat'?s in\b", re.IGNORECASE), Intent.LIST_DIRECTORY),

        # 3a-3c. Listing cues — checked BEFORE the location-query rules
        # below, because phrases like "current workspace" / "this
        # directory" appear in BOTH a listing request ("list the current
        # workspace", "read files in the current workspace") and a bare
        # location query ("what is my current workspace?"). The presence
        # of a listing cue (a leading "list" verb, "files ... in", or an
        # explicit "this directory/folder") disambiguates in favor of
        # LIST_DIRECTORY; its absence falls through to the location-query
        # rules that follow.
        (re.compile(r"^\s*list\b", re.IGNORECASE), Intent.LIST_DIRECTORY),
        (re.compile(r"\bfiles?\s+(?:are\s+)?in\b", re.IGNORECASE), Intent.LIST_DIRECTORY),
        (re.compile(r"\bthis\s+(?:directory|folder)\b", re.IGNORECASE), Intent.LIST_DIRECTORY),

        # 2. Location query — "current/active directory|workspace", plus
        #    the natural variants ("what workspace are you working in",
        #    "show my active workspace", "where are you working").
        #    Checked BEFORE the generic switch/list fallbacks so these
        #    read as queries, not commands.
        (re.compile(r"\b(?:current|active)\s+(?:directory|workspace)\b", re.IGNORECASE),
         Intent.CURRENT_DIRECTORY),
        (re.compile(r"\bwhere am i\b|\bwhere are you\b|\bwhat (?:directory|folder)\b.*\b(?:am i in|are we in)\b|"
                     r"\bwhat directory\b|\bwhat workspace\b|\bworking in\b", re.IGNORECASE),
         Intent.CURRENT_DIRECTORY),

        # 3. Switch/set workspace — imperative phrasing. "work in" and
        #    "use <path>" require an explicit path-like target so they
        #    don't collide with the query phrasing above.
        (re.compile(r"\bset\s+(?:the\s+)?workspace\s+to\b|\bswitch\s+workspace(?:s)?\s+to\b|"
                     r"\bswitch to\b|\bchange (?:the )?(?:directory|workspace) to\b|"
                     r"^\s*cd\s|\bwork\s+in\b|\bmake\b.*\bmy\s+workspace\b|"
                     r"\buse\s+(?:[A-Za-z]:[\\/]|\.{1,2}[\\/]|~[\\/]|/)",
                     re.IGNORECASE),
         Intent.SWITCH_DIRECTORY),

        # 3a. NAVIGATE — go to / navigate to / move into / open <folder>.
        #     Checked AFTER switch/set workspace so those more-explicit
        #     phrasings ("switch workspace to") win when present.
        #     These rules must come BEFORE the rename/copy/move action verbs
        #     (rules 4+) to prevent "move into backup" being classified as MOVE.
        #
        #     Parent navigation: "go up", "go back", "go to ..", "go to parent",
        #     "navigate to parent", "move to parent".
        #     Current navigation: "go to .".
        #     Named navigation: "go to backup", "navigate to D:/AG".
        (re.compile(
            r"\bgo\s+(?:up|back)\b"
            r"|\bgo\s+to\s+(?:\.\.|parent\b)"
            r"|\bnavigate\s+to\s+(?:parent\b|\.\.)"
            r"|\bmove\s+to\s+(?:parent\b|\.\.)",
            re.IGNORECASE,
        ), Intent.NAVIGATE),
        (re.compile(
            r"\bgo\s+to\b"
            r"|\bnavigate\s+to\b"
            r"|\bmove\s+into\b",
            re.IGNORECASE,
        ), Intent.NAVIGATE),

        # 4. Rename / copy / move / delete — explicit verbs, unambiguous.
        (re.compile(r"\brename\b", re.IGNORECASE), Intent.RENAME),
        (re.compile(r"\bcopy\b|^\s*cp\s", re.IGNORECASE), Intent.COPY),
        (re.compile(r"\bmove\b|^\s*mv\s", re.IGNORECASE), Intent.MOVE),
        (re.compile(r"\bdelete\b|\bremove\b|^\s*rm\s", re.IGNORECASE), Intent.DELETE),

        # 5. Append (checked before generic write/create).
        (re.compile(r"\bappend\b|\badd\b.*\bto\b.*\bfile\b", re.IGNORECASE),
         Intent.APPEND_FILE),

        # 6. Create folder vs create file — folder keyword must be
        #    checked first since "create a file called notes and make a
        #    folder" style ambiguity is rare, but "folder"/"directory"
        #    is the stronger, less ambiguous signal.
        (re.compile(r"\bcreate\b.*\b(?:folder|directory)\b|\bmake\b.*\b(?:folder|directory)\b|"
                     r"^\s*mkdir\s", re.IGNORECASE), Intent.CREATE_FOLDER),
        (re.compile(r"\bcreate\b.*\bfile\b|\bmake\b.*\bfile\b", re.IGNORECASE),
         Intent.CREATE_FILE),

        # 7. Write (only reached if "create"/"make" + "file" didn't
        #    already match above — so "create a file ... and write: ..."
        #    is correctly classified as CREATE_FILE, not WRITE_FILE).
        (re.compile(r"\bwrite\b", re.IGNORECASE), Intent.WRITE_FILE),

        # 8. Search.
        (re.compile(r"\bsearch\b|\bfind\b.*\bfile\b", re.IGNORECASE), Intent.SEARCH),

        # 9. "Open <folder>" = list that folder's contents.
        (re.compile(r"\bopen\b.*\b(?:folder|directory)\b", re.IGNORECASE),
         Intent.LIST_DIRECTORY),

        # 10. Read file — checked last among action verbs since "read"/
        #     "open" are common but shouldn't shadow the more specific
        #     rules above.
        (re.compile(r"\bread\b|\bopen file\b|^\s*cat\s|\bcontents of\b", re.IGNORECASE),
         Intent.READ_FILE),

        # 11. Bare list/show verb — last resort, only reached if nothing
        #     more specific above matched (e.g. "show test.txt" would
        #     already have hit READ_FILE-ish rules if it named a file;
        #     this exists for plain "list"/"show" with no other signal).
        (re.compile(r"^\s*(?:list|show)\b", re.IGNORECASE), Intent.LIST_DIRECTORY),
    ]

    def detect(self, text: str) -> IntentResult:
        for pattern, intent in self._RULES:
            if pattern.search(text):
                return IntentResult(intent=intent, confidence=0.85,
                                     reason=f"Matched rule for {intent}.")
        return IntentResult(intent=Intent.UNKNOWN, confidence=0.0,
                             reason="No directory intent pattern matched.")


# =============================================================================
# SECTION 6 — Feature 4: Command Parsing (NL -> structured command)
# =============================================================================

class CommandParser:
    """
    Converts (intent, raw user text) into a ParsedCommand.

    This is where the historical Directory Control bug is fixed: filler
    text such as "called directory_test.txt and write:" must never
    become part of the filename. See extract_named_target() /
    split_target_and_content() in Section 2 for the extraction logic.
    """

    _LEADING_VERB_RE = re.compile(
        r"^\s*(?:please\s+)?(?:list|show|read|open|create|make|write|append|add|"
        r"switch|change|cd|rename|copy|cp|move|mv|delete|remove|rm|search|find|mkdir|cat)\b",
        re.IGNORECASE,
    )

    def parse(self, intent: str, raw_text: str) -> ParsedCommand:
        raw = raw_text.strip()

        if intent == Intent.CURRENT_DIRECTORY:
            return ParsedCommand(domain=Domain.DIRECTORY, intent=intent)

        if intent == Intent.LIST_DIRECTORY:
            return self._parse_list_directory(raw)

        if intent == Intent.SWITCH_DIRECTORY:
            return self._parse_switch_directory(raw)

        if intent == Intent.NAVIGATE:
            return self._parse_navigate_directory(raw)

        if intent == Intent.INSPECT:
            return self._parse_inspect(raw)

        if intent == Intent.READ_FILE:
            return self._parse_single_path_command(
                raw, intent,
                prefixes=("read file ", "read ", "open file ", "open ", "cat "),
            )

        if intent == Intent.CREATE_FOLDER:
            return self._parse_create_folder(raw)

        if intent == Intent.CREATE_FILE:
            return self._parse_create_file(raw)

        if intent == Intent.WRITE_FILE:
            return self._parse_write_or_append(raw, intent)

        if intent == Intent.APPEND_FILE:
            return self._parse_write_or_append(raw, intent)

        if intent in (Intent.RENAME, Intent.COPY, Intent.MOVE):
            return self._parse_source_dest_command(raw, intent)

        if intent == Intent.DELETE:
            return self._parse_single_path_command(
                raw, intent,
                prefixes=("delete ", "remove ", "rm "),
            )

        if intent == Intent.SEARCH:
            return self._parse_search(raw)

        return ParsedCommand(domain=Domain.DIRECTORY, intent=Intent.UNKNOWN,
                              complete=False, missing_fields=["intent"])

    # ------------------------------------------------------------------

    def _strip_leading_verb_phrase(self, raw: str, prefixes: Tuple[str, ...]) -> str:
        lower = raw.lower()
        for prefix in prefixes:
            if lower.startswith(prefix):
                return raw[len(prefix):]
        # Fall back to stripping a single generic leading verb token.
        m = self._LEADING_VERB_RE.match(raw)
        if m:
            return raw[m.end():]
        return raw

    def _parse_navigate_directory(self, raw: str) -> ParsedCommand:
        """
        Parse a NAVIGATE intent into a ParsedCommand with a ``target``.

        Navigation targets (in priority order):

        1. Parent navigation keywords → target = ".."
           "go up", "go back", "go to ..", "go to parent",
           "navigate to parent", "move to parent", "navigate to .."

        2. Current-directory token → target = "."
           "go to ."

        3. Absolute Windows path → target = the path as-given (forward or
           back slashes; drive letter required)

        4. Quoted path/name → target = the quoted string

        5. Explicit relative path (contains separator) → target = the path

        6. Bare name (no separator) — e.g. "go to backup" →
           target = the bare name (resolved as relative at execution time)

        If no target can be extracted, returns an incomplete ParsedCommand
        so the caller can ask for clarification.
        """
        text = raw.strip()
        text_lower = text.lower()

        # ---- 1. Parent navigation ----------------------------------------
        _PARENT_PATTERNS = [
            re.compile(r"\bgo\s+(?:up|back)\b", re.IGNORECASE),
            re.compile(r"\bgo\s+to\s+\.\.", re.IGNORECASE),
            re.compile(r"\bgo\s+to\s+parent\b", re.IGNORECASE),
            re.compile(r"\bnavigate\s+to\s+(?:parent\b|\.\.)", re.IGNORECASE),
            re.compile(r"\bmove\s+to\s+(?:parent\b|\.\.)", re.IGNORECASE),
        ]
        for pat in _PARENT_PATTERNS:
            if pat.search(text):
                return ParsedCommand(domain=Domain.DIRECTORY, intent=Intent.NAVIGATE,
                                      target="..")

        # ---- 2. Current-directory ". " token ----------------------------
        _CURRENT_PATTERNS = [
            re.compile(r"\bgo\s+to\s+\.\s*$", re.IGNORECASE),
        ]
        for pat in _CURRENT_PATTERNS:
            if pat.search(text):
                return ParsedCommand(domain=Domain.DIRECTORY, intent=Intent.NAVIGATE,
                                      target=".")

        # ---- Strip the navigation verb phrase to get a raw target zone --
        # Remove leading "go to", "navigate to", "move into", "open",
        # plus optional trailing punctuation.
        _NAV_VERB_RE = re.compile(
            r"^\s*(?:please\s+)?"
            r"(?:go\s+to|navigate\s+to|move\s+into|open)\s+",
            re.IGNORECASE,
        )
        m = _NAV_VERB_RE.match(text)
        target_zone = text[m.end():].strip() if m else text

        # Strip surrounding quotes
        target_zone = _strip_quotes(target_zone)
        target_zone = _strip_target_punct(target_zone)

        if not target_zone:
            return ParsedCommand(domain=Domain.DIRECTORY, intent=Intent.NAVIGATE,
                                  complete=False, missing_fields=["target"])

        # ---- 3. Absolute Windows path -----------------------------------
        # e.g. "D:/AG", "D:\AG", "D:/AG/backup"
        abs_match = re.match(
            r"^[A-Za-z]:[/\\].*",
            target_zone,
        )
        if abs_match:
            return ParsedCommand(domain=Domain.DIRECTORY, intent=Intent.NAVIGATE,
                                  target=_strip_target_punct(target_zone))

        # ---- 4. UNC or posix absolute path ------------------------------
        if target_zone.startswith("\\\\") or target_zone.startswith("/"):
            return ParsedCommand(domain=Domain.DIRECTORY, intent=Intent.NAVIGATE,
                                  target=_strip_target_punct(target_zone))

        # ---- 5. Explicit relative path (contains separator) -------------
        # e.g. "backup/pre_preprocessor_migration", "backup\subdir"
        if "/" in target_zone or "\\" in target_zone:
            return ParsedCommand(domain=Domain.DIRECTORY, intent=Intent.NAVIGATE,
                                  target=_strip_target_punct(target_zone))

        # ---- 6. Bare name (single directory component) ------------------
        # e.g. "backup", "automation timeline", "pre_preprocessor_migration"
        # Accept it as-is — relative resolution happens in execute().
        cleaned = _strip_target_punct(target_zone)
        if cleaned:
            return ParsedCommand(domain=Domain.DIRECTORY, intent=Intent.NAVIGATE,
                                  target=cleaned)

        return ParsedCommand(domain=Domain.DIRECTORY, intent=Intent.NAVIGATE,
                              complete=False, missing_fields=["target"])

    def _parse_inspect(self, raw: str) -> ParsedCommand:
        """
        Parse an INSPECT intent into a ParsedCommand with a ``target``.

        Recognised forms:
          inspect <target>
          inspect the current workspace / inspect . / inspect here
          show information about <target>
          show info about <target>
          tell me about <target>

        The ACTIVE_WORKSPACE sentinel is returned for phrases that mean
        "inspect wherever I currently am".
        """
        text = raw.strip()

        # ---- Strip the verb phrase to get the target zone ---------------
        _INSPECT_VERB_RE = re.compile(
            r"^\s*(?:please\s+)?"
            r"(?:inspect"
            r"|show\s+(?:information|info)\s+about"
            r"|tell\s+me\s+about"
            r")\s+",
            re.IGNORECASE,
        )
        m = _INSPECT_VERB_RE.match(text)
        target_zone = text[m.end():].strip() if m else text

        # Strip surrounding quotes only (not trailing punct yet) so that
        # the special tokens "." and ".." survive the next steps.
        target_zone = _strip_quotes(target_zone).strip()

        if not target_zone:
            return ParsedCommand(domain=Domain.DIRECTORY, intent=Intent.INSPECT,
                                  complete=False, missing_fields=["target"])

        # ---- Special path tokens — must be handled before _strip_target_punct
        #      because "." is in _TRAILING_PUNCT and would be wiped out.
        if target_zone == ".":
            return ParsedCommand(domain=Domain.DIRECTORY, intent=Intent.INSPECT,
                                  target=ACTIVE_WORKSPACE)
        if target_zone == "..":
            return ParsedCommand(domain=Domain.DIRECTORY, intent=Intent.INSPECT,
                                  target="..")

        # Now safe to apply trailing-punctuation stripping
        target_zone = _strip_target_punct(target_zone)

        if not target_zone:
            return ParsedCommand(domain=Domain.DIRECTORY, intent=Intent.INSPECT,
                                  complete=False, missing_fields=["target"])

        # ---- Context phrases → ACTIVE_WORKSPACE sentinel ----------------
        _lower = target_zone.lower()
        _INSPECT_WORKSPACE_PHRASES = {
            ".", "here", "this", "current", "this directory", "this folder",
            "this file", "this workspace", "the workspace", "the directory",
            "the folder", "the current workspace", "current workspace",
            "active workspace", "the active workspace", "my workspace",
            "the current directory", "current directory", "the active directory",
            "active directory",
        }
        if _lower in _INSPECT_WORKSPACE_PHRASES:
            return ParsedCommand(domain=Domain.DIRECTORY, intent=Intent.INSPECT,
                                  target=ACTIVE_WORKSPACE)

        # ---- Absolute Windows path --------------------------------------
        if re.match(r"^[A-Za-z]:[/\\]", target_zone):
            return ParsedCommand(domain=Domain.DIRECTORY, intent=Intent.INSPECT,
                                  target=_strip_target_punct(target_zone))

        # ---- UNC or POSIX absolute path ---------------------------------
        if target_zone.startswith("\\\\") or target_zone.startswith("/"):
            return ParsedCommand(domain=Domain.DIRECTORY, intent=Intent.INSPECT,
                                  target=_strip_target_punct(target_zone))

        # ---- Bare name or relative path (contains separator or bare word) ---
        cleaned = _strip_target_punct(target_zone)
        if cleaned:
            return ParsedCommand(domain=Domain.DIRECTORY, intent=Intent.INSPECT,
                                  target=cleaned)

        return ParsedCommand(domain=Domain.DIRECTORY, intent=Intent.INSPECT,
                              complete=False, missing_fields=["target"])

    # Checked in order; the first structural match wins. "make X my
    # workspace" is handled separately because the path sits in the
    # MIDDLE of the sentence, not after a fixed prefix, so the generic
    # prefix-stripping approach used elsewhere can't extract it.
    _SWITCH_PATTERNS: Tuple[re.Pattern, ...] = (
        re.compile(r"\bmake\s+(?P<path>.+?)\s+my\s+workspace\b", re.IGNORECASE),
        re.compile(
            r"\b(?:set\s+(?:the\s+)?workspace\s+to|switch\s+workspace(?:s)?\s+to|"
            r"switch\s+to|change\s+(?:the\s+)?(?:directory|workspace)\s+to|"
            r"work\s+in|use)\s+(?P<path>.+?)\s*[.?!]*\s*$",
            re.IGNORECASE,
        ),
        re.compile(r"^\s*cd\s+(?P<path>.+?)\s*[.?!]*\s*$", re.IGNORECASE),
    )

    def _parse_switch_directory(self, raw: str) -> ParsedCommand:
        for pattern in self._SWITCH_PATTERNS:
            m = pattern.search(raw)
            if m:
                path = _strip_target_punct(m.group("path").strip())
                if path:
                    return ParsedCommand(domain=Domain.DIRECTORY,
                                          intent=Intent.SWITCH_DIRECTORY, target=path)
        return ParsedCommand(domain=Domain.DIRECTORY, intent=Intent.SWITCH_DIRECTORY,
                              complete=False, missing_fields=["target"])

    def _parse_single_path_command(self, raw: str, intent: str,
                                    prefixes: Tuple[str, ...]) -> ParsedCommand:
        remainder = self._strip_leading_verb_phrase(raw, prefixes)
        target = extract_named_target(remainder)
        if target is None:
            target = _last_resort_target(remainder)
        target = _strip_target_punct(target) if target else None

        if not target:
            return ParsedCommand(domain=Domain.DIRECTORY, intent=intent,
                                  complete=False, missing_fields=["target"])
        return ParsedCommand(domain=Domain.DIRECTORY, intent=intent, target=target)

    def _parse_list_directory(self, raw: str) -> ParsedCommand:
        # Tier 1: an explicit path (drive letter, or containing an actual
        # path separator) always wins, regardless of surrounding filler
        # like "files in" / "the directory" — this is what fixes targets
        # such as "files in D:/AG" incorrectly becoming the whole phrase.
        m = _EXPLICIT_DIR_PATH_RE.search(raw)
        if m:
            target = _strip_target_punct(m.group(0))
            if target:
                return ParsedCommand(domain=Domain.DIRECTORY, intent=Intent.LIST_DIRECTORY,
                                      target=target)

        # Tier 2: "open the X folder" mid-sentence pattern.
        mo = re.search(r"\bopen\s+(?:the\s+)?(.+?)\s+(?:folder|directory)\b", raw, re.IGNORECASE)
        if mo:
            cleaned = _strip_target_punct(raw[mo.start(1):mo.end(1)])
            if cleaned and cleaned.lower() not in _ACTIVE_WORKSPACE_PHRASES:
                return ParsedCommand(domain=Domain.DIRECTORY, intent=Intent.LIST_DIRECTORY,
                                      target=cleaned)

        # Tier 3: strip the leading verb + common filler; if a short,
        # specific-looking bare name remains (e.g. "brains"), treat it as
        # a relative sub-path within the active workspace.
        remainder = re.sub(
            r"^\s*(?:please\s+)?(?:list|show|read)\s+(?:me\s+)?(?:what'?s\s+)?(?:the\s+)?",
            "", raw, flags=re.IGNORECASE,
        )
        remainder = re.sub(r"\b(?:files?|contents?)\s+(?:are\s+)?in\b", "", remainder,
                            flags=re.IGNORECASE)
        remainder = re.sub(r"^\s*in\b", "", remainder, flags=re.IGNORECASE).strip()
        cleaned = _strip_target_punct(remainder)
        if (cleaned and cleaned.lower() not in _ACTIVE_WORKSPACE_PHRASES
                and len(cleaned.split()) <= 3):
            return ParsedCommand(domain=Domain.DIRECTORY, intent=Intent.LIST_DIRECTORY,
                                  target=cleaned)

        # Tier 4: nothing specific was named — the request refers to the
        # workspace that is already active, not a literal path.
        return ParsedCommand(domain=Domain.DIRECTORY, intent=Intent.LIST_DIRECTORY,
                              target=ACTIVE_WORKSPACE)

    def _parse_create_folder(self, raw: str) -> ParsedCommand:
        remainder = self._strip_leading_verb_phrase(
            raw, ("create a folder ", "create folder ", "make a folder ", "make folder ",
                  "create a directory ", "create directory ", "make a directory ",
                  "make directory ", "mkdir "))
        target = extract_named_target(remainder) or _last_resort_target(remainder)
        target = _strip_target_punct(target) if target else None
        if not target:
            return ParsedCommand(domain=Domain.DIRECTORY, intent=Intent.CREATE_FOLDER,
                                  complete=False, missing_fields=["target"])
        return ParsedCommand(domain=Domain.DIRECTORY, intent=Intent.CREATE_FOLDER, target=target)

    def _parse_create_file(self, raw: str) -> ParsedCommand:
        target_zone, content = split_target_and_content(raw)
        target = extract_named_target(target_zone) or _last_resort_target(
            self._strip_leading_verb_phrase(
                target_zone, ("create a file ", "create file ", "make a file ", "make file "))
        )
        target = _strip_target_punct(target) if target else None
        content_clean = _strip_quotes(content) if content else None

        missing = []
        if not target:
            missing.append("target")
        return ParsedCommand(
            domain=Domain.DIRECTORY, intent=Intent.CREATE_FILE,
            target=target, content=content_clean,
            complete=not missing, missing_fields=missing,
        )

    def _parse_write_or_append(self, raw: str, intent: str) -> ParsedCommand:
        # Canonical form: "write <content> to <target>" / "append <content> to <target>"
        verb = "write" if intent == Intent.WRITE_FILE else "(?:append|add)"
        m = re.search(
            rf"\b{verb}\s+(?P<content>.+?)\s+(?:to|into)\s+(?P<target>.+?)\s*\.?\s*$",
            raw, re.IGNORECASE,
        )
        if m:
            content = _strip_quotes(m.group("content").strip())
            target_raw = m.group("target").strip()
            target = extract_named_target(target_raw) or _strip_target_punct(target_raw)
            return ParsedCommand(domain=Domain.DIRECTORY, intent=intent,
                                  target=target, content=content)

        # "write to <target>" / "append to <target>" with no inline content —
        # structurally incomplete; content must come from a follow-up turn.
        m2 = re.search(
            rf"\b{verb}\s+(?:this\s+|that\s+)?(?:to|into)\s+(?P<target>.+?)\s*\.?\s*$",
            raw, re.IGNORECASE,
        )
        if m2:
            target_raw = m2.group("target").strip()
            target = extract_named_target(target_raw) or _strip_target_punct(target_raw)
            return ParsedCommand(domain=Domain.DIRECTORY, intent=intent, target=target,
                                  content=None, complete=False, missing_fields=["content"])

        return ParsedCommand(domain=Domain.DIRECTORY, intent=intent,
                              complete=False, missing_fields=["target", "content"])

    def _parse_source_dest_command(self, raw: str, intent: str) -> ParsedCommand:
        remainder = self._strip_leading_verb_phrase(
            raw, ("rename file ", "rename ", "copy ", "cp ", "move ", "mv "))
        m = re.search(r"^(?P<source>.+?)\s+to\s+(?P<destination>.+?)\s*\.?\s*$",
                       remainder, re.IGNORECASE)
        if not m:
            return ParsedCommand(domain=Domain.DIRECTORY, intent=intent,
                                  complete=False, missing_fields=["source", "destination"])

        source_raw = m.group("source").strip()
        dest_raw = m.group("destination").strip()
        source = extract_named_target(source_raw) or _strip_target_punct(source_raw)
        destination = extract_named_target(dest_raw) or _strip_target_punct(dest_raw)

        missing = []
        if not source:
            missing.append("source")
        if not destination:
            missing.append("destination")
        return ParsedCommand(domain=Domain.DIRECTORY, intent=intent,
                              source=source, destination=destination,
                              complete=not missing, missing_fields=missing)

    def _parse_search(self, raw: str) -> ParsedCommand:
        m = re.search(r"\bsearch\s+(?:for\s+)?(?P<pattern>.+?)\s*\.?\s*$", raw, re.IGNORECASE)
        if not m:
            m = re.search(r"\bfind\s+(?:a\s+)?(?:file\s+)?(?:called\s+|named\s+)?"
                           r"(?P<pattern>.+?)\s*\.?\s*$", raw, re.IGNORECASE)
        if not m:
            return ParsedCommand(domain=Domain.DIRECTORY, intent=Intent.SEARCH,
                                  complete=False, missing_fields=["target"])
        pattern = extract_named_target(m.group("pattern")) or _strip_target_punct(
            m.group("pattern"))
        if not pattern:
            return ParsedCommand(domain=Domain.DIRECTORY, intent=Intent.SEARCH,
                                  complete=False, missing_fields=["target"])
        return ParsedCommand(domain=Domain.DIRECTORY, intent=Intent.SEARCH, target=pattern)


# =============================================================================
# SECTION 7 — Feature 5: Directory Control integration
# =============================================================================
#
# PreprocessorBrain integrates with AG's REAL DirectoryControl by duck
# typing: any object exposing the method names below is used as-is (this
# is exactly what CapabilityRegistry checks for too, so capability
# answers and actual execution can never disagree).
#
# For standalone development/testing (Section 17 of the design doc),
# a minimal, self-contained fallback implementation is provided below.
# It is intentionally NOT a competing architecture — same method
# surface, pure pathlib, single active-workspace model — so swapping in
# the real DirectoryControl later is a drop-in replacement with zero
# changes to the rest of this file.
# =============================================================================

class DirectoryOpsError(Exception):
    """Base class for the standalone fallback directory operations."""


class NoActiveWorkspaceError(DirectoryOpsError):
    pass


class WorkspaceNotFoundError(DirectoryOpsError):
    pass


class WorkspaceNotADirectoryError(DirectoryOpsError):
    """The path exists but is a file, not a directory — distinct from
    WorkspaceNotFoundError so callers get a clear, specific error rather
    than a generic 'does not exist' message for a path that does."""
    pass


class WorkspaceViolationError(DirectoryOpsError):
    pass


class FileNotFoundInWorkspace(DirectoryOpsError):
    pass


class NotAFileError(DirectoryOpsError):
    pass


class OperationError(DirectoryOpsError):
    pass


class DirectoryOperations:
    """
    Minimal, self-contained, workspace-bounded filesystem backend used
    ONLY when no real DirectoryControl instance is supplied. Mirrors the
    method surface of AG's directory_control.DirectoryControl closely
    enough that PreprocessorBrain code never needs to know which one it
    is talking to.

    Deliberately does NOT implement delete — matches the currently
    verified, real DirectoryControl (see Section 6 of the design doc:
    "No deletion. No file execution."), so CapabilityRegistry correctly
    reports DELETE as UNSUPPORTED against this backend too.
    """

    MAX_READ_BYTES = 131_072

    def __init__(self) -> None:
        self._active_directory: Optional[Path] = None

    # -- workspace management ------------------------------------------------

    def set_directory(self, path: str) -> str:
        target = Path(path).expanduser()
        if not target.exists():
            raise WorkspaceNotFoundError(f"Workspace path does not exist: {path}")
        if not target.is_dir():
            raise WorkspaceNotADirectoryError(f"Workspace path is not a directory: {path}")
        # Only reassigned once both checks pass — a failed validation
        # above raises before this line, so the previously active
        # workspace is never overwritten by a rejected path.
        self._active_directory = target.resolve()
        return str(self._active_directory)

    def switch_directory(self, path: str) -> str:
        return self.set_directory(path)

    def navigate_directory(self, target: str) -> str:
        """
        Navigate to a target directory using the NAVIGATE contract.
        Mirrors the real DirectoryControl.navigate_directory() surface.
        """
        stripped = target.strip().strip('"').strip("'").strip()

        # Normalise "parent" keyword
        if stripped.lower() == "parent":
            stripped = ".."

        candidate = Path(stripped)
        if candidate.is_absolute():
            resolved = candidate.resolve()
            if not resolved.exists():
                raise WorkspaceNotFoundError(
                    f"Navigation target does not exist: {stripped}"
                )
            if not resolved.is_dir():
                raise WorkspaceNotADirectoryError(
                    f"Navigation target is not a directory: {stripped}"
                )
            self._active_directory = resolved
            return str(resolved)

        # Relative / special — require an active workspace
        current = self._require_workspace()

        if stripped == ".":
            return str(current)

        if stripped == "..":
            parent = current.parent
            if parent == current:
                raise WorkspaceNotFoundError(
                    "Already at the filesystem root — cannot navigate further up."
                )
            resolved = parent.resolve()
            if not resolved.exists() or not resolved.is_dir():
                raise WorkspaceNotFoundError(
                    f"Parent directory does not exist or is not a directory: {resolved}"
                )
            self._active_directory = resolved
            return str(resolved)

        candidate = (current / stripped).resolve()
        if not candidate.exists():
            raise WorkspaceNotFoundError(
                f"Navigation target does not exist: {stripped}"
            )
        if not candidate.is_dir():
            raise WorkspaceNotADirectoryError(
                f"Navigation target is not a directory: {stripped}"
            )
        self._active_directory = candidate
        return str(candidate)

    def clear_directory(self) -> Optional[str]:
        previous = str(self._active_directory) if self._active_directory else None
        self._active_directory = None
        return previous

    def get_active_directory(self) -> Optional[str]:
        return str(self._active_directory) if self._active_directory else None

    def _require_workspace(self) -> Path:
        if self._active_directory is None:
            raise NoActiveWorkspaceError("No active workspace is set.")
        return self._active_directory

    def _resolve(self, relative_or_absolute: str) -> Path:
        ws = self._require_workspace()
        candidate = Path(relative_or_absolute)
        target = candidate if candidate.is_absolute() else ws / candidate
        target = target.resolve()
        try:
            target.relative_to(ws)
        except ValueError:
            raise WorkspaceViolationError(
                f"Path is outside the active workspace boundary: {relative_or_absolute}"
            )
        return target

    # -- inspection ------------------------------------------------------

    def list_directory(self, sub_path: str = ".", *, files_only: bool = False,
                        dirs_only: bool = False) -> List[dict]:
        target = self._resolve(sub_path)
        if not target.exists():
            raise FileNotFoundInWorkspace(f"Directory does not exist: {sub_path}")
        if not target.is_dir():
            raise NotAFileError(f"Path is a file, not a directory: {sub_path}")

        entries = []
        ws = self._active_directory
        for item in sorted(target.iterdir()):
            kind = "dir" if item.is_dir() else "file"
            if files_only and kind != "file":
                continue
            if dirs_only and kind != "dir":
                continue
            size = 0
            if kind == "file":
                try:
                    size = item.stat().st_size
                except OSError:
                    pass
            try:
                rel = str(item.relative_to(ws))
            except ValueError:
                rel = item.name
            entries.append({"name": item.name, "rel_path": rel, "type": kind, "size_bytes": size})
        return entries

    def search_files(self, pattern: str) -> List[dict]:
        ws = self._require_workspace()
        pattern_lower = pattern.lower()
        results = []
        for item in ws.rglob("*"):
            if pattern_lower in item.name.lower():
                kind = "dir" if item.is_dir() else "file"
                size = 0
                if kind == "file":
                    try:
                        size = item.stat().st_size
                    except OSError:
                        pass
                try:
                    rel = str(item.relative_to(ws))
                except ValueError:
                    rel = item.name
                results.append({"name": item.name, "rel_path": rel, "type": kind,
                                 "size_bytes": size})
        return results

    # -- read --------------------------------------------------------------

    def read_file(self, path: str) -> str:
        target = self._resolve(path)
        if not target.exists():
            raise FileNotFoundInWorkspace(f"File does not exist: {path}")
        if not target.is_file():
            raise NotAFileError(f"Path is a directory, not a file: {path}")
        try:
            size = target.stat().st_size
            truncated = size > self.MAX_READ_BYTES
            content = target.read_bytes()[: self.MAX_READ_BYTES].decode("utf-8", errors="replace")
            if truncated:
                content += f"\n\n[NOTE: File truncated at {self.MAX_READ_BYTES} bytes.]"
            return content
        except PermissionError:
            raise OperationError(f"Permission denied reading: {path}")
        except OSError as exc:
            raise OperationError(f"Could not read file: {exc}")

    # -- create / write ------------------------------------------------------

    def create_file(self, path: str, content: str = "") -> Path:
        target = self._resolve(path)
        if target.exists() and target.is_file():
            raise OperationError(f"File already exists: {path}. Use write_file to replace it.")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        except PermissionError:
            raise OperationError(f"Permission denied creating file: {path}")
        except OSError as exc:
            raise OperationError(f"Could not create file: {exc}")
        return target

    def write_file(self, path: str, content: str) -> Path:
        target = self._resolve(path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        except PermissionError:
            raise OperationError(f"Permission denied writing to: {path}")
        except OSError as exc:
            raise OperationError(f"Could not write file: {exc}")
        return target

    def append_file(self, path: str, content: str) -> Path:
        target = self._resolve(path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as fh:
                fh.write(content)
        except PermissionError:
            raise OperationError(f"Permission denied appending to: {path}")
        except OSError as exc:
            raise OperationError(f"Could not append to file: {exc}")
        return target

    def create_directory(self, path: str) -> Path:
        target = self._resolve(path)
        if target.exists() and target.is_dir():
            raise OperationError(f"Directory already exists: {path}")
        try:
            target.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            raise OperationError(f"Directory already exists: {path}")
        except PermissionError:
            raise OperationError(f"Permission denied creating directory: {path}")
        except OSError as exc:
            raise OperationError(f"Could not create directory: {exc}")
        return target

    # -- move / copy / rename ------------------------------------------------

    def move(self, source: str, destination: str) -> Path:
        src = self._resolve(source)
        dst = self._resolve(destination)
        if not src.exists():
            raise FileNotFoundInWorkspace(f"Source does not exist: {source}")
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
        except PermissionError:
            raise OperationError(f"Permission denied moving: {source}")
        except OSError as exc:
            raise OperationError(f"Move failed: {exc}")
        return dst / src.name if dst.is_dir() and dst.name != src.name else dst

    def copy(self, source: str, destination: str) -> Path:
        src = self._resolve(source)
        dst = self._resolve(destination)
        if not src.exists():
            raise FileNotFoundInWorkspace(f"Source does not exist: {source}")
        if not src.is_file():
            raise NotAFileError(f"Source is a directory, not a file: {source}")
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), str(dst))
        except PermissionError:
            raise OperationError(f"Permission denied copying: {source}")
        except OSError as exc:
            raise OperationError(f"Copy failed: {exc}")
        return dst / src.name if dst.is_dir() else dst

    def rename(self, source: str, new_name: str) -> Path:
        src = self._resolve(source)
        if not src.exists():
            raise FileNotFoundInWorkspace(f"Source does not exist: {source}")
        if "/" in new_name or "\\" in new_name:
            dst = self._resolve(new_name)
        else:
            dst = self._resolve(str(Path(source).parent / new_name))
        if dst.exists():
            raise OperationError(f"Target already exists: {new_name}")
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.rename(dst)
        except PermissionError:
            raise OperationError(f"Permission denied renaming: {source}")
        except OSError as exc:
            raise OperationError(f"Rename failed: {exc}")
        return dst

    def inspect_target(self, path: str) -> dict:
        """
        Collect deterministic, read-only metadata for a file or directory.

        For a path that is the ACTIVE_WORKSPACE sentinel or an absolute
        path, resolve directly; for a relative path, resolve against the
        active workspace.

        Returns a dict with keys appropriate to the target type. Never
        modifies the filesystem.
        """
        import os as _os

        # Determine the resolved path
        candidate = Path(path)
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            ws = self._require_workspace()
            resolved = (ws / path).resolve()
            # Containment check — inspection is workspace-bounded
            try:
                resolved.relative_to(ws)
            except ValueError:
                raise WorkspaceViolationError(
                    f"Inspect target is outside the active workspace: {path}"
                )

        if not resolved.exists():
            raise FileNotFoundInWorkspace(f"Inspect target does not exist: {path}")

        stat = resolved.stat()
        mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()

        if resolved.is_file():
            is_readable = _os.access(resolved, _os.R_OK)
            is_writable = _os.access(resolved, _os.W_OK)
            result: dict = {
                "operation": "INSPECT",
                "target_type": "file",
                "name": resolved.name,
                "path": str(resolved),
                "extension": resolved.suffix,
                "size": stat.st_size,
                "modified": mtime,
                "readable": is_readable,
                "writable": is_writable,
            }
            return result

        if resolved.is_dir():
            file_count = 0
            dir_count = 0
            try:
                for item in resolved.iterdir():
                    if item.is_file():
                        file_count += 1
                    elif item.is_dir():
                        dir_count += 1
            except PermissionError:
                pass  # partial count if permission denied
            result = {
                "operation": "INSPECT",
                "target_type": "directory",
                "name": resolved.name,
                "path": str(resolved),
                "file_count": file_count,
                "directory_count": dir_count,
                "modified": mtime,
            }
            return result

        # Exists but is neither file nor directory (symlink edge case, etc.)
        raise OperationError(
            f"Inspect target is not a regular file or directory: {path}"
        )


class DirectoryOperationExecutor:
    """
    Executes a ParsedCommand against a duck-typed directory-operations
    object (real DirectoryControl OR the standalone DirectoryOperations
    fallback) and converts every outcome — success or failure — into a
    PreprocessorResult. Exceptions never leak past this class.
    """

    def __init__(self, directory_ops: Any):
        self.ops = directory_ops

    def execute(self, cmd: ParsedCommand) -> PreprocessorResult:
        try:
            return self._dispatch(cmd)
        except NoActiveWorkspaceError as exc:
            return self._error(cmd, DirErrorCode.NO_ACTIVE_WORKSPACE, str(exc))
        except WorkspaceNotFoundError as exc:
            return self._error(cmd, DirErrorCode.NOT_FOUND, str(exc))
        except WorkspaceNotADirectoryError as exc:
            return self._error(cmd, DirErrorCode.NOT_A_DIRECTORY, str(exc))
        except WorkspaceViolationError as exc:
            return self._error(cmd, DirErrorCode.OUT_OF_WORKSPACE, str(exc))
        except FileNotFoundInWorkspace as exc:
            return self._error(cmd, DirErrorCode.NOT_FOUND, str(exc))
        except NotAFileError as exc:
            lower = str(exc).lower()
            code = (DirErrorCode.NOT_A_DIRECTORY if "not a directory" in lower
                    else DirErrorCode.NOT_A_FILE)
            return self._error(cmd, code, str(exc))
        except OperationError as exc:
            return self._error(cmd, DirErrorCode.OPERATION_FAILED, str(exc))
        except AttributeError as exc:
            # The connected ops object doesn't implement this operation.
            return self._error(cmd, DirErrorCode.CAPABILITY_UNAVAILABLE, str(exc))
        except Exception as exc:
            # Last-resort isolation boundary — also handles exceptions from the
            # real DirectoryControl (e.g. path_security.WorkspaceNotFoundError
            # which is a different class from our local WorkspaceNotFoundError).
            # Classify by message content so NOT_FOUND / NOT_A_DIRECTORY remain
            # accurate even when crossing exception class boundaries.
            msg = str(exc)
            msg_lower = msg.lower()
            if "not a directory" in msg_lower or "is not a directory" in msg_lower:
                return self._error(cmd, DirErrorCode.NOT_A_DIRECTORY, msg)
            if (
                "does not exist" in msg_lower
                or "not exist" in msg_lower
                or "not found" in msg_lower
                or "no active workspace" in msg_lower
                or "workspace directory does not exist" in msg_lower
            ):
                if "no active workspace" in msg_lower:
                    return self._error(cmd, DirErrorCode.NO_ACTIVE_WORKSPACE, msg)
                return self._error(cmd, DirErrorCode.NOT_FOUND, msg)
            if "outside" in msg_lower or "boundary" in msg_lower:
                return self._error(cmd, DirErrorCode.OUT_OF_WORKSPACE, msg)
            return self._error(cmd, DirErrorCode.OPERATION_FAILED,
                                f"Unexpected filesystem error: {exc}")

    def _error(self, cmd: ParsedCommand, code: str, message: str) -> PreprocessorResult:
        return PreprocessorResult(
            success=False, domain=Domain.DIRECTORY, intent=cmd.intent,
            operation=cmd.intent, target=cmd.target or cmd.source,
            error=code, error_domain=ErrorDomain.DIRECTORY_CONTROL,
            confidence=1.0, requires_llm=False,
            metadata={"internal_message": message, "command": cmd.to_dict()},
        )

    def _ok(self, cmd: ParsedCommand, result: Any) -> PreprocessorResult:
        return PreprocessorResult(
            success=True, domain=Domain.DIRECTORY, intent=cmd.intent,
            operation=cmd.intent, target=cmd.target or cmd.source,
            result=result, confidence=1.0, requires_llm=False,
            metadata={"command": cmd.to_dict()},
        )

    def _dispatch(self, cmd: ParsedCommand) -> PreprocessorResult:
        intent = cmd.intent

        if intent == Intent.CURRENT_DIRECTORY:
            active = self.ops.get_active_directory()
            if active is None:
                return self._error(cmd, DirErrorCode.NO_ACTIVE_WORKSPACE,
                                    "No active workspace is set.")
            return self._ok(cmd, active)

        if intent == Intent.LIST_DIRECTORY:
            if cmd.target in (None, ACTIVE_WORKSPACE):
                sub_path = "."
                display_target = self.ops.get_active_directory()
            else:
                sub_path = cmd.target
                display_target = cmd.target
            entries = self.ops.list_directory(sub_path)
            return PreprocessorResult(
                success=True, domain=Domain.DIRECTORY, intent=cmd.intent,
                operation=cmd.intent, target=display_target, result=entries,
                confidence=1.0, requires_llm=False, metadata={"command": cmd.to_dict()},
            )

        if intent == Intent.READ_FILE:
            content = self.ops.read_file(cmd.target)
            return self._ok(cmd, content)

        if intent == Intent.CREATE_FILE:
            path = self.ops.create_file(cmd.target, cmd.content or "")
            return self._ok(cmd, str(path))

        if intent == Intent.CREATE_FOLDER:
            path = self.ops.create_directory(cmd.target)
            return self._ok(cmd, str(path))

        if intent == Intent.WRITE_FILE:
            path = self.ops.write_file(cmd.target, cmd.content or "")
            return self._ok(cmd, str(path))

        if intent == Intent.APPEND_FILE:
            path = self.ops.append_file(cmd.target, cmd.content or "")
            return self._ok(cmd, str(path))

        if intent == Intent.SWITCH_DIRECTORY:
            resolved = self.ops.switch_directory(cmd.target)
            return self._ok(cmd, resolved)

        if intent == Intent.NAVIGATE:
            resolved = self.ops.navigate_directory(cmd.target)
            return self._ok(cmd, resolved)

        if intent == Intent.INSPECT:
            # ACTIVE_WORKSPACE sentinel → inspect the active workspace itself
            if cmd.target in (None, ACTIVE_WORKSPACE):
                active = self.ops.get_active_directory()
                if active is None:
                    return self._error(cmd, DirErrorCode.NO_ACTIVE_WORKSPACE,
                                        "No active workspace is set.")
                inspection = self.ops.inspect_target(active)
            else:
                inspection = self.ops.inspect_target(cmd.target)
            return PreprocessorResult(
                success=True, domain=Domain.DIRECTORY, intent=cmd.intent,
                operation="INSPECT",
                target=inspection.get("path") or cmd.target,
                result=inspection,
                confidence=1.0, requires_llm=False,
                metadata={"command": cmd.to_dict()},
            )

        if intent == Intent.RENAME:
            path = self.ops.rename(cmd.source, cmd.destination)
            return self._ok(cmd, str(path))

        if intent == Intent.COPY:
            path = self.ops.copy(cmd.source, cmd.destination)
            return self._ok(cmd, str(path))

        if intent == Intent.MOVE:
            path = self.ops.move(cmd.source, cmd.destination)
            return self._ok(cmd, str(path))

        if intent == Intent.DELETE:
            # Never guess at destructive capability — only call it if the
            # connected ops object actually implements it.
            delete_fn = getattr(self.ops, "delete", None)
            if not callable(delete_fn):
                return self._error(cmd, DirErrorCode.CAPABILITY_UNAVAILABLE,
                                    "No delete() implementation is connected.")
            delete_fn(cmd.target)
            return self._ok(cmd, cmd.target)

        if intent == Intent.SEARCH:
            results = self.ops.search_files(cmd.target)
            return self._ok(cmd, results)

        return self._error(cmd, DirErrorCode.AMBIGUOUS_REQUEST,
                            f"No executor branch for intent {intent}.")


# =============================================================================
# SECTION 8 — Top-level orchestrator: PreprocessorBrain
# =============================================================================

class PreprocessorBrain:
    """
    Single entry point: ``PreprocessorBrain(...).process(user_input)``.

    Parameters
    ----------
    directory_control:
        A real DirectoryControl instance (or any duck-typed equivalent).
        If omitted, a standalone DirectoryOperations fallback is created
        automatically so this class works fully offline/out of the box.
    introspection_source:
        Optional IntrospectionEngine / RuntimeSnapshot / dict, folded
        into self-info answers as supplementary verified evidence.
    brain_info / architecture_info:
        Optional overrides for the static, verified architecture facts
        used to answer GET_BRAIN_INFO / GET_ARCHITECTURE.
    """

    def __init__(
        self,
        directory_control: Optional[Any] = None,
        introspection_source: Optional[Any] = None,
        brain_info: Optional[Dict[str, Any]] = None,
        architecture_info: Optional[str] = None,
    ) -> None:
        self.directory_ops: Any = directory_control or DirectoryOperations()
        self.classifier = RequestClassifier()
        self.intent_detector = IntentDetector()
        self.parser = CommandParser()
        self.capabilities = CapabilityRegistry(
            directory_ops=self.directory_ops, introspection_source=introspection_source)
        self.self_info = SelfInfoProcessor(
            self.capabilities, brain_info=brain_info, architecture_info=architecture_info)
        self.executor = DirectoryOperationExecutor(self.directory_ops)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def can_handle(self, user_input: str) -> bool:
        """Deterministic yes/no: can the Preprocessor Brain fully handle this?"""
        classification = self.classifier.classify(user_input)
        if classification.domain == Domain.SELF_INFO:
            return True
        if classification.domain == Domain.DIRECTORY and classification.confidence >= 0.5:
            intent = self.intent_detector.detect(user_input.strip())
            return intent.intent != Intent.UNKNOWN
        return False

    def process(self, user_input: Optional[str]) -> PreprocessorResult:
        """
        Main entry point. NEVER raises.

        Returns a PreprocessorResult. If ``requires_llm`` is True, the
        caller should fall through to the Local/Cloud Brain as normal;
        the Preprocessor Brain has deliberately declined to guess.
        """
        try:
            return self._process_inner(user_input)
        except Exception as exc:  # absolute safety net
            return PreprocessorResult(
                success=False, domain=Domain.UNKNOWN, intent=Intent.UNKNOWN,
                error="PREPROCESSOR_INTERNAL_ERROR", error_domain=ErrorDomain.PREPROCESSOR,
                requires_llm=True,
                metadata={"internal_message": str(exc)},
            )

    def process_self_info_intent(
        self,
        subtype: str,
        original_text: str = "",
    ) -> PreprocessorResult:
        """Answer a QUIPS-owned self-info subtype without reclassifying its wording."""
        intent_by_subtype = {
            "IDENTITY": Intent.GET_IDENTITY,
            "CAPABILITIES": Intent.GET_CAPABILITIES,
            "LIMITATIONS": Intent.GET_LIMITATIONS,
            "ARCHITECTURE": Intent.GET_ARCHITECTURE,
            "OPERATIONAL": Intent.GET_BRAIN_INFO,
            "BRAINS": Intent.GET_BRAIN_INFO,
        }
        intent = intent_by_subtype.get(str(subtype).upper())
        if intent is None:
            return PreprocessorResult(
                success=False,
                domain=Domain.SELF_INFO,
                intent=Intent.UNKNOWN,
                error=f"Unsupported self-info subtype: {subtype}",
                error_domain=ErrorDomain.SELF_INFO,
                requires_llm=False,
                metadata={"source": "QUIPS"},
            )
        return PreprocessorResult(
            success=True,
            domain=Domain.SELF_INFO,
            intent=intent,
            operation=intent,
            target=None,
            result=self.self_info.answer(intent),
            confidence=1.0,
            requires_llm=False,
            metadata={"source": "QUIPS", "request": str(original_text)},
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _process_inner(self, user_input: Optional[str]) -> PreprocessorResult:
        if not user_input or not isinstance(user_input, str) or not user_input.strip():
            return PreprocessorResult(
                success=False, domain=Domain.UNKNOWN, intent=Intent.UNKNOWN,
                error="EMPTY_INPUT", error_domain=ErrorDomain.PREPROCESSOR,
                confidence=0.0, requires_llm=False,
            )

        text = user_input.strip()
        classification = self.classifier.classify(text)

        if classification.domain == Domain.SELF_INFO:
            return self._handle_self_info(text, classification)

        if classification.domain == Domain.DIRECTORY:
            return self._handle_directory(text, classification)

        if classification.domain == Domain.GENERAL_LLM:
            return PreprocessorResult(
                success=False, domain=Domain.GENERAL_LLM, intent=Intent.UNKNOWN,
                confidence=classification.confidence, requires_llm=True,
                metadata={"reason": classification.reason},
            )

        # UNKNOWN — defer to the LLM rather than guessing, but flag low confidence.
        return PreprocessorResult(
            success=False, domain=Domain.UNKNOWN, intent=Intent.UNKNOWN,
            confidence=classification.confidence, requires_llm=True,
            metadata={"reason": classification.reason},
        )

    def _handle_self_info(self, text: str, classification: ClassificationResult) -> PreprocessorResult:
        intent_result = self.self_info.detect_intent(text.lower())
        answer = self.self_info.answer(intent_result.intent)
        return PreprocessorResult(
            success=True, domain=Domain.SELF_INFO, intent=intent_result.intent,
            operation=intent_result.intent, result=answer,
            confidence=min(classification.confidence, intent_result.confidence),
            requires_llm=False,
            metadata={"classification_reason": classification.reason,
                      "intent_reason": intent_result.reason},
        )

    def _handle_directory(self, text: str, classification: ClassificationResult) -> PreprocessorResult:
        intent_result = self.intent_detector.detect(text)

        if intent_result.intent == Intent.UNKNOWN or classification.confidence < 0.5:
            # Do NOT guess. Never perform a dangerous/irreversible
            # operation from a weak signal — surface a clarification.
            return PreprocessorResult(
                success=False, domain=Domain.DIRECTORY, intent=Intent.UNKNOWN,
                error=DirErrorCode.AMBIGUOUS_REQUEST, error_domain=ErrorDomain.PREPROCESSOR,
                confidence=classification.confidence, requires_llm=False,
                metadata={
                    "clarification_needed": True,
                    "message": "I couldn't determine a specific file/directory "
                               "operation from that request. Could you say what "
                               "you'd like me to do (e.g. read, create, write, "
                               "list, move, copy, rename, delete) and which "
                               "file/folder?",
                    "classification_reason": classification.reason,
                    "intent_reason": intent_result.reason,
                },
            )

        cmd = self.parser.parse(intent_result.intent, text)

        if not cmd.complete:
            return PreprocessorResult(
                success=False, domain=Domain.DIRECTORY, intent=cmd.intent,
                operation=cmd.intent,
                error=DirErrorCode.AMBIGUOUS_REQUEST, error_domain=ErrorDomain.PREPROCESSOR,
                confidence=intent_result.confidence, requires_llm=False,
                metadata={
                    "clarification_needed": True,
                    "missing_fields": cmd.missing_fields,
                    "parsed_so_far": cmd.to_dict(),
                    "message": f"I understood this as a {cmd.intent} request but "
                               f"couldn't determine: {', '.join(cmd.missing_fields)}.",
                },
            )

        capability = self.capabilities.directory_capability(cmd.intent)
        if capability.state in (CapabilityState.UNSUPPORTED, CapabilityState.UNAVAILABLE):
            return PreprocessorResult(
                success=False, domain=Domain.DIRECTORY, intent=cmd.intent,
                operation=cmd.intent, target=cmd.target or cmd.source,
                error=DirErrorCode.CAPABILITY_UNAVAILABLE, error_domain=ErrorDomain.DIRECTORY_CONTROL,
                confidence=intent_result.confidence, requires_llm=False,
                metadata={"capability_description": capability.description,
                          "command": cmd.to_dict()},
            )

        result = self.executor.execute(cmd)
        result.confidence = min(result.confidence, intent_result.confidence) or intent_result.confidence
        return result


# =============================================================================
# SECTION 9 — Standalone self-test / demo (Section 17 of the design doc)
# =============================================================================
#
# Runs entirely offline, with no dependency on the rest of the AG
# codebase. Exercises every test case listed in the design doc,
# including malformed input, nonexistent targets, and directory/file
# mismatches.
# =============================================================================

def _print_result(label: str, result: PreprocessorResult) -> None:
    print(f"\n>>> {label}")
    print(f"    success={result.success}  domain={result.domain}  intent={result.intent}")
    if result.target:
        print(f"    target={result.target!r}")
    if result.requires_llm:
        print(f"    requires_llm=True  reason={result.metadata.get('reason')}")
    if result.error:
        print(f"    error={result.error}  ({result.metadata.get('internal_message') or result.metadata.get('message')})")
    if result.result is not None:
        out = result.result
        if isinstance(out, str) and len(out) > 200:
            out = out[:200] + "... [truncated for display]"
        print(f"    result={out!r}")


def run_workspace_regression_tests() -> None:
    """
    Focused regression tests for SET_WORKSPACE (Phase 4 of the workspace
    fix). Asserts, rather than just prints, so a future regression fails
    loudly instead of needing to be eyeballed.
    """
    import tempfile

    with tempfile.TemporaryDirectory(prefix="workspace_regress_") as tmp:
        good_dir = Path(tmp) / "AG"
        good_dir.mkdir()
        other_dir = Path(tmp) / "Other"
        other_dir.mkdir()
        a_file = good_dir / "test.txt"
        a_file.write_text("not a directory", encoding="utf-8")
        nonexistent = Path(tmp) / "does-not-exist"

        def fresh_brain() -> PreprocessorBrain:
            return PreprocessorBrain()

        # --- Setting: every phrasing variant must resolve to SWITCH_DIRECTORY
        # and succeed with the resolved path as the result. ---
        setting_cases = [
            f"set workspace to {good_dir}",
            f"use {good_dir}",
            f"work in {good_dir}",
            f"switch workspace to {good_dir}",
            f"switch to {good_dir}",
            f"make {good_dir} my workspace",
        ]
        for text in setting_cases:
            b = fresh_brain()
            result = b.process(text)
            assert result.domain == Domain.DIRECTORY, f"[{text}] wrong domain: {result.domain}"
            assert result.intent == Intent.SWITCH_DIRECTORY, f"[{text}] wrong intent: {result.intent}"
            assert result.success, f"[{text}] did not succeed: {result.error} / {result.metadata}"
            assert Path(result.result) == good_dir.resolve(), \
                f"[{text}] resolved to {result.result!r}, expected {good_dir.resolve()}"

        # --- Queries: every phrasing variant must return the active workspace. ---
        query_cases = [
            "what is my current workspace?",
            "what's the current workspace?",
            "what workspace are you working in?",
            "show my active workspace",
            "where are you working?",
        ]
        b = fresh_brain()
        set_result = b.process(f"set workspace to {good_dir}")
        assert set_result.success, f"setup set_workspace failed: {set_result.error}"
        for text in query_cases:
            result = b.process(text)
            assert result.intent == Intent.CURRENT_DIRECTORY, \
                f"[{text}] wrong intent: {result.intent} (this used to mis-fire as LIST_DIRECTORY)"
            assert result.success, f"[{text}] did not succeed: {result.error}"
            assert Path(result.result) == good_dir.resolve(), \
                f"[{text}] returned {result.result!r}, expected {good_dir.resolve()}"

        # --- Validation: nonexistent path -> NOT_FOUND, file path -> NOT_A_DIRECTORY. ---
        b = fresh_brain()
        bad1 = b.process(f"set workspace to {nonexistent}")
        assert not bad1.success and bad1.error == DirErrorCode.NOT_FOUND, \
            f"nonexistent workspace path did not yield NOT_FOUND: {bad1.to_dict()}"

        bad2 = b.process(f"set workspace to {a_file}")
        assert not bad2.success and bad2.error == DirErrorCode.NOT_A_DIRECTORY, \
            f"file-as-workspace did not yield NOT_A_DIRECTORY: {bad2.to_dict()}"

        # --- State preservation: a failed switch must NOT clobber a
        # previously-valid active workspace. ---
        b = fresh_brain()
        first = b.process(f"set workspace to {good_dir}")
        assert first.success
        failed = b.process(f"set workspace to {nonexistent}")
        assert not failed.success
        still = b.process("what is my current workspace?")
        assert Path(still.result) == good_dir.resolve(), (
            f"active workspace was clobbered by a failed switch: {still.result!r} "
            f"(expected it to remain {good_dir.resolve()})"
        )

        # --- Switching again to a second valid workspace must actually move. ---
        second = b.process(f"set workspace to {other_dir}")
        assert second.success
        confirm = b.process("what is my current workspace?")
        assert Path(confirm.result) == other_dir.resolve()

    print("Workspace regression tests: ALL PASSED "
          f"({len(setting_cases)} setting phrasings, {len(query_cases)} query phrasings, "
          "validation, and state-preservation).")


def run_listing_regression_tests() -> None:
    """
    Focused regression tests for LIST_DIRECTORY / READ_FILE (Phase 7 of
    the workspace-listing fix). Asserts, rather than just prints.
    """
    import tempfile

    with tempfile.TemporaryDirectory(prefix="listing_regress_") as tmp:
        workspace = Path(tmp) / "AG"
        workspace.mkdir()
        (workspace / "test.txt").write_text("hello workspace", encoding="utf-8")
        (workspace / "brain.py").write_text("# brain\n", encoding="utf-8")
        (workspace / "brains").mkdir()

        def fresh_brain_in_workspace() -> PreprocessorBrain:
            b = PreprocessorBrain()
            setup = b.process(f"set workspace to {workspace}")
            assert setup.success, f"workspace setup failed: {setup.error}"
            return b

        # --- Workspace-referring listing phrasings: all must resolve to
        # LIST_DIRECTORY with target=ACTIVE_WORKSPACE at the parse level,
        # and must actually list the active workspace's real contents. ---
        workspace_listing_cases = [
            "list the current workspace",
            "list files in the current workspace",
            "read files in the current workspace",
            "show files in the current workspace",
            "what files are in the current workspace?",
            "show me what's in the current workspace",
            "what's in the current workspace?",
        ]
        expected_names = {"test.txt", "brain.py", "brains"}
        for text in workspace_listing_cases:
            b = fresh_brain_in_workspace()
            cmd = b.parser.parse(b.intent_detector.detect(text).intent, text)
            assert cmd.intent == Intent.LIST_DIRECTORY, f"[{text}] wrong intent: {cmd.intent}"
            assert cmd.target == ACTIVE_WORKSPACE, \
                f"[{text}] target was {cmd.target!r}, expected ACTIVE_WORKSPACE sentinel"

            result = b.process(text)
            assert result.success, f"[{text}] failed: {result.error} / {result.metadata}"
            assert result.intent == Intent.LIST_DIRECTORY
            assert Path(result.target) == workspace.resolve(), \
                f"[{text}] reported target {result.target!r}, expected {workspace.resolve()}"
            names = {e["name"] for e in result.result}
            assert names == expected_names, f"[{text}] listed {names}, expected {expected_names}"

        # --- Explicit directory targets must keep working and must NOT
        # swallow surrounding filler text into the target. ---
        explicit_cases = [
            f"list {workspace}",
            f"show files in {workspace}",
            f"list the directory {workspace}",
        ]
        for text in explicit_cases:
            b = fresh_brain_in_workspace()
            result = b.process(text)
            assert result.success, f"[{text}] failed: {result.error} / {result.metadata}"
            assert result.intent == Intent.LIST_DIRECTORY
            assert Path(result.target) == workspace.resolve(), \
                f"[{text}] target was {result.target!r}, expected {workspace.resolve()}"

        # --- File reading must resolve relative to the active workspace,
        # and explicit paths must keep their drive letter intact. ---
        b = fresh_brain_in_workspace()
        read_result = b.process("read test.txt")
        assert read_result.success and read_result.intent == Intent.READ_FILE
        assert read_result.result == "hello workspace", read_result.result

        explicit_read_cmd = b.parser.parse(Intent.READ_FILE, f"read {workspace / 'test.txt'}")
        assert str(workspace) in explicit_read_cmd.target, \
            f"drive/path prefix was lost: {explicit_read_cmd.target!r}"

        # --- Directory vs file must not be conflated: listing a directory
        # and reading a file must resolve to different intents. ---
        list_intent = b.intent_detector.detect(f"list {workspace}").intent
        read_intent = b.intent_detector.detect(f"read {workspace / 'test.txt'}").intent
        assert list_intent == Intent.LIST_DIRECTORY
        assert read_intent == Intent.READ_FILE
        assert list_intent != read_intent

    print("Listing/read regression tests: ALL PASSED "
          f"({len(workspace_listing_cases)} workspace-listing phrasings, "
          f"{len(explicit_cases)} explicit-directory phrasings, file reading, "
          "and directory-vs-file distinction).")


def run_self_tests() -> None:
    import tempfile

    with tempfile.TemporaryDirectory(prefix="preprocessor_brain_test_") as tmp:
        workspace = Path(tmp) / "AG"
        workspace.mkdir()
        (workspace / "brain.py").write_text("# brain module\n", encoding="utf-8")
        brains_dir = workspace / "brains" / "cloud_brain"
        brains_dir.mkdir(parents=True)
        (brains_dir / "adaptive_brain.py").write_text("# adaptive brain\n", encoding="utf-8")

        other_workspace = Path(tmp) / "AG" / "tests"
        other_workspace.mkdir(exist_ok=True)

        brain = PreprocessorBrain()
        brain.directory_ops.set_directory(str(workspace))

        cases = [
            "What are your capabilities?",
            "What can you do with files?",
            "What is my current directory?",
            "List everything in the current directory.",
            "Read brain.py.",
            "Read brains/cloud_brain/adaptive_brain.py.",
            "Open the brains folder.",
            f"Switch to {other_workspace}.",
            "What is the active directory now?",
            "Create a folder called AB_Test.",
            "Create a file called test.txt and write: Preprocessor test successful.",
            "Rename test.txt to renamed_test.txt.",
            "Delete renamed_test.txt.",
            "Do something with that file.",
            "Read does_not_exist.py.",
            "List not_a_real_subdir.",
            "Explain quantum computing.",
        ]

        for case in cases:
            result = brain.process(case)
            _print_result(case, result)


if __name__ == "__main__":
    run_self_tests()
    print("\n" + "=" * 70)
    run_workspace_regression_tests()
    print("=" * 70)
    run_listing_regression_tests()