"""QUIPS — deterministic request structure analysis for AG.

QUIPS identifies whether an input contains independent request units.  It
does not answer requests, perform operations, call an LLM, or replace AG's
existing intent routing.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Tuple

from target_resolution_models import ActionHint, ResolutionMode, TargetReference, TargetType
from target_resolver import looks_absolute
from self_info.query_router import (
    ROUTE_ARCHITECTURE,
    ROUTE_CAPABILITIES,
    ROUTE_IDENTITY,
    ROUTE_LIMITATIONS,
    ROUTE_OPERATIONAL,
    route_query,
)


class RequestCategory:
    SELF_INFO = "SELF_INFO"
    CONVERSATION = "CONVERSATION"
    FILE_OPERATION = "FILE_OPERATION"
    MEMORY = "MEMORY"
    TASK = "TASK"
    PROJECT = "PROJECT"
    WORKSPACE_QUERY = "WORKSPACE_QUERY"
    EXPLANATION = "EXPLANATION"
    GENERAL = "GENERAL"
    UNKNOWN = "UNKNOWN"


class RoutingDecision:
    """Downstream ownership selected by deterministic request analysis."""

    DETERMINISTIC_SELF_INFO = "DETERMINISTIC_SELF_INFO"
    LEGACY_INTENT_ROUTER = "LEGACY_INTENT_ROUTER"
    LLM_FALLBACK = "LLM_FALLBACK"


class DesiredOutcome:
    OPEN_TARGET = "OPEN_TARGET"
    OPEN_DIRECTORY = "OPEN_DIRECTORY"
    DIRECTORY_CONTENTS = "DIRECTORY_CONTENTS"
    FILE_CONTENT = "FILE_CONTENT"
    EXPLAIN_TARGET = "EXPLAIN_TARGET"
    EXPLAIN_FILE = "EXPLAIN_FILE"
    EXPLAIN_RESULT = "EXPLAIN_RESULT"
    CREATE_FILE = "CREATE_FILE"
    CREATE_DIRECTORY = "CREATE_DIRECTORY"
    WRITE_FILE = "WRITE_FILE"
    APPEND_FILE = "APPEND_FILE"
    SEARCH_FILES = "SEARCH_FILES"
    DELETE_TARGET = "DELETE_TARGET"
    RENAME_TARGET = "RENAME_TARGET"
    MOVE_TARGET = "MOVE_TARGET"
    COPY_TARGET = "COPY_TARGET"
    GENERAL_RESPONSE = "GENERAL_RESPONSE"
    SELF_INFORMATION = "SELF_INFORMATION"
    CONVERSATION_CONTEXT = "CONVERSATION_CONTEXT"
    WORKSPACE_INFORMATION = "WORKSPACE_INFORMATION"
    MEMORY_OPERATION = "MEMORY_OPERATION"


@dataclass(frozen=True)
class RequestUnderstanding:
    """Purposeful, deterministic interpretation of one request unit."""

    action: str
    target: Optional[str]
    target_type: str
    domain: str
    desired_outcome: str
    destination: Optional[str] = None
    references: Tuple[str, ...] = ()
    context_requirements: Tuple[str, ...] = ()
    missing_information: Tuple[str, ...] = ()
    is_complete: bool = True
    semantic_reasoning_required: bool = False
    content: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RequestUnit:
    """One independently routable request extracted from user input."""

    original_text: str
    text: str
    index: int
    is_compound: bool
    category: str
    intent: str
    confidence: str
    legacy_intent: Optional[str] = None
    routing_decision: str = RoutingDecision.LEGACY_INTENT_ROUTER
    context_references: Tuple[str, ...] = ()
    depends_on: Tuple[int, ...] = ()
    processing_status: str = "PENDING"
    result_state: Optional[str] = None
    target_reference: Optional[TargetReference] = None
    destination_reference: Optional[TargetReference] = None
    understanding: Optional[RequestUnderstanding] = None

    @property
    def route_intent(self) -> Optional[str]:
        """Backward-compatible name for the legacy single-request route."""
        return self.legacy_intent

    @property
    def is_target_request(self) -> bool:
        return self.target_reference is not None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class UnitProcessingResult:
    """Outcome recorded by the router after a request unit is processed."""

    unit_index: int
    handler: str
    status: str
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Optional[str]]:
        return asdict(self)


@dataclass(frozen=True)
class QUIPSResult:
    """Structured, side-effect-free result from :func:`analyze_request`."""

    original_text: str
    is_compound: bool
    requests: List[RequestUnit]

    @property
    def request_count(self) -> int:
        return len(self.requests)

    def to_dict(self) -> Dict[str, object]:
        return {
            "original_text": self.original_text,
            "is_compound": self.is_compound,
            "request_count": self.request_count,
            "requests": [request.to_dict() for request in self.requests],
        }


_BOUNDARY = re.compile(r"\s*,\s*(?:and\s+)?|\s+(?:and|also|then)\s+", re.IGNORECASE)

# The canonical set of verbs QUIPS recognizes as filesystem operations.
# Single source of truth, shared by the category-classification regex,
# the operation-prefix stripper, and the action-hint inference below —
# previously these three consumers each maintained an independent copy,
# which is how "access" ended up missing from the classification regex
# and the action-hint check while still being present in the prefix
# stripper: a request like "access test.txt in AG" (the canonical
# example from the Target Resolver design) could have its filename
# extracted correctly but would never reach that code, because the
# classifier itself didn't recognize "access" as a file-operation verb
# and routed the whole request to GENERAL/LLM_FALLBACK instead.
_FILE_OPERATION_CORE_VERBS = (
    "access", "create", "make", "write", "read", "open",
    "delete", "remove", "rename", "move", "copy", "list",
)
_FILE_OPERATION_TARGET_WORDS = r"\bfile\b|\bfolder\b|\bdirectory\b|\.[a-z0-9]{1,8}\b"
_FILE_OPERATION_RE = re.compile(
    r"\b(?:" + "|".join(_FILE_OPERATION_CORE_VERBS) + r")\b.*(?:" + _FILE_OPERATION_TARGET_WORDS + ")",
    re.IGNORECASE,
)
_CREATE_FOLDER_FILE_WORKFLOW = re.compile(
    r"^\s*(?:create|make)\s+(?:a\s+)?(?:folder|directory)\s+"
    r"(?:(?:called|named)\s+)?(?P<folder>\"[^\"]+\"|'[^']+'|.+?)\s+"
    r"and\s+(?:create|make)\s+(?:a\s+)?file\s+"
    r"(?:(?:called|named)\s+)?(?P<file>\"[^\"]+\"|'[^']+'|.+?)\s+"
    r"(?:inside|in|within)\s+(?:it|that\s+folder|the\s+folder)\s*[.!?]*$",
    re.IGNORECASE,
)
_WORKSPACE_LIST_RE = re.compile(
    r"^\s*(?:what\s+(?:files|folders)\s+are\s+here|what\s+is\s+here|"
    r"list\s+(?:the\s+)?(?:files|folders|this\s+directory|(?:the\s+)?current\s+directory|"
    r"current\s+workspace))\b",
    re.IGNORECASE,
)
_BARE_OPEN_TARGET_RE = re.compile(
    r"^\s*open\s+(?:the\s+)?[A-Za-z0-9_.-]+(?:\s+(?:folder|directory))?\b",
    re.IGNORECASE,
)
_DIRECTORY_CONTENTS_RE = re.compile(
    r"^\s*(?:(?:show|tell)\s+me\s+)?(?:what(?:'s|\s+is)\s+inside|"
    r"(?:show|list)\s+(?:me\s+)?(?:the\s+)?contents\s+of)\s+.+$",
    re.IGNORECASE,
)
_EXPLAIN_FILE_RE = re.compile(
    r"^\s*(?:tell\s+me\s+about|explain|describe)\s+.+\.[A-Za-z0-9]{1,8}\s*$",
    re.IGNORECASE,
)
_CONTEXTUAL_FILE_ACTION_RE = re.compile(
    r"^\s*(?:read|open|write|list|show|tell\s+me\s+what(?:'s|\s+is)\s+inside|"
    r"what(?:'s|\s+is)\s+inside)\s+(?:it|this|that|the\s+(?:file|folder|directory)|"
    r"the\s+(?:older|newer|first|second|previous|last)\s+(?:file|folder|directory)|"
    r"the\s+(?:file|folder|directory)\s+i\s+(?:created|made|opened|read)\s+earlier|"
    r"the\s+thing\s+i\s+(?:created|made)\s+earlier)\b",
    re.IGNORECASE,
)
_WRITE_FILE_RE = re.compile(
    r"""^\s*write\s+(?:"[^"]*"|'[^']*')\s+(?:into|to|in)\s+.+$""",
    re.IGNORECASE,
)
_APPEND_FILE_RE = re.compile(
    r"""^\s*(?:please\s+)?(?:append|add)\s+(?:"[^"]*"|'[^']*')\s+"""
    r"(?:into|to|in)\s+.+$",
    re.IGNORECASE,
)
_SEARCH_FILES_RE = re.compile(
    r"^\s*(?:please\s+)?find\s+(?:a\s+)?files?\b.*$",
    re.IGNORECASE,
)
_RESULT_EXPLANATION_RE = re.compile(
    r"^\s*(?:what\s+(?:did\s+you|have\s+you)\s+(?:change|create|make|write|do)|"
    r"what\s+(?:changed|was\s+changed|was\s+created|was\s+written)|"
    r"what\s+happened|what\s+did\s+you\s+actually\s+do|"
    r"(?:tell|explain|describe)\s+me\s+what\s+(?:it|that|this)\s+(?:does|did|contains)|"
    r"what\s+does\s+(?:it|that|this)\s+do)\b",
    re.IGNORECASE,
)
_CONVERSATION_RE = re.compile(
    r"^\s*(?:what\s+did\s+i\s+(?:just\s+)?ask(?:\s+you)?\b|"
    r"what\s+was\s+(?:my|the)\s+previous\s+request\b|"
    r"what\s+did\s+you\s+tell\s+me\b|"
    r"what\s+were\s+we\s+talking\s+about\b|"
    r"explain\s+what\s+you\s+(?:just\s+)?told\s+me\b|"
    r"tell\s+me\s+more\b|more\s+about\b|what\s+about\b|"
    r"go\s+on\b|continue\b|elaborate\b|"
    r"(?:simplify|expand\s+on)\s+(?:it|this|that)\b)",
    re.IGNORECASE,
)

_REQUEST_START = re.compile(
    r"^(?:what|who|where|when|why|how|tell|describe|explain|list|show|"
    r"create|make|write|read|open|delete|remove|rename|move|copy|find|"
    r"search|remember|recall|add|complete|finish|mark|set|switch|inspect|"
    r"access|check|navigate|go)\b",
    re.IGNORECASE,
)
_WORKFLOW = re.compile(
    r"\b(?:create|make)\s+(?:a\s+)?(?:folder|directory)\b.*\band\s+"
    r"(?:put|place|move|copy|create|write)\b.*\b(?:inside|into|there)\b",
    re.IGNORECASE,
)

# Workspace-SETTING phrasing (imperative) vs workspace-QUERYING phrasing
# (interrogative). Both map to RequestCategory.WORKSPACE_QUERY (preserving
# the existing category name/value for anyone already matching on it) but
# are distinguished by `intent` (WORKSPACE_SET vs WORKSPACE_QUERY), and —
# critically — both are routed to LEGACY_INTENT_ROUTER rather than falling
# through to LLM_FALLBACK, which is what "set workspace to D:/AG" did
# before this fix (see _classify_broad_category).
_WORKSPACE_SET_RE = re.compile(
    r"\bset\s+(?:the\s+)?workspace\s+to\b|\bswitch\s+workspace(?:s)?\s+to\b|"
    r"\bswitch\s+to\b|\bchange\s+(?:the\s+)?(?:directory|workspace)\s+to\b|"
    r"^\s*cd\s|\bwork\s+in\b|\bmake\b.*\bmy\s+workspace\b",
    re.IGNORECASE,
)
_WORKSPACE_QUERY_RE = re.compile(
    r"\b(?:current|active|my)\s+workspace\b|\bwhere am i\b|\bwhat workspace\b|"
    r"\bwhat\s+folder\s+am\s+i\s+in\b|\bworking in\b|\bwhere are you\b",
    re.IGNORECASE,
)

# "create a file called X" / "...named X" — the filler-text-as-filename
# bug this project has fixed at the DirectoryControl/Preprocessor-Brain
# layer previously, now fixed at its source in QUIPS's own extraction so
# it never produces a malformed TargetReference in the first place.
_CALLED_NAMED_RE = re.compile(
    r"\b(?:called|named|titled)\s+(\"[^\"]+\"|'[^']+'|[^\s,]+)", re.IGNORECASE,
)
_TRAILING_SENTENCE_PUNCT = "?!. "


def analyze_request(
    user_input: str,
    intent_classifier: Optional[Callable[[str], str]] = None,
) -> QUIPSResult:
    """Return a deterministic structural analysis of *user_input*.

    ``intent_classifier`` is normally AG's existing ``detect_intent``.  It is
    injected to keep QUIPS independent from ``Jarwis.py`` and to avoid a circular
    import.  QUIPS records that existing route in ``route_intent``; it does not
    duplicate the legacy intent system.
    """
    text = (user_input or "").strip()
    try:
        parts = _split_independent_requests(text)
    except Exception:
        # Decomposition itself failed unexpectedly — degrade to a single
        # whole-text unit rather than losing the request entirely.
        parts = [text] if text else []

    is_compound = len(parts) > 1
    units: List[RequestUnit] = []
    for index, part in enumerate(parts):
        try:
            units.append(_classify_unit(text, part, index, is_compound, intent_classifier))
        except Exception as exc:
            # One unit failing to classify must not destroy its siblings'
            # results (error isolation — see design spec section 16).
            # Falls back to the safest possible structural result: an
            # UNKNOWN unit deferred to the LLM, never a guessed category.
            units.append(RequestUnit(
                original_text=text,
                text=part,
                index=index,
                is_compound=is_compound,
                category=RequestCategory.UNKNOWN,
                intent="UNKNOWN",
                confidence="low",
                routing_decision=RoutingDecision.LLM_FALLBACK,
                processing_status="CLASSIFICATION_FAILED",
                result_state=f"error: {exc}",
            ))
    return QUIPSResult(original_text=text, is_compound=is_compound, requests=units)


def _split_independent_requests(text: str) -> List[str]:
    """Split only obvious coordinated requests; keep operational workflows whole."""
    workflow = _CREATE_FOLDER_FILE_WORKFLOW.fullmatch(text)
    if workflow:
        folder = workflow.group("folder").strip().strip("\"'")
        filename = workflow.group("file").strip().strip("\"'")
        return [f"create folder {folder}", f"create file {filename}"]
    if not text or _WORKFLOW.search(text):
        return [text] if text else []

    candidates = [part.strip(" ,") for part in _BOUNDARY.split(text) if part.strip(" ,")]
    if len(candidates) < 2 or not all(_REQUEST_START.search(part) for part in candidates):
        return [text]
    return candidates


def _classify_unit(
    original_text: str,
    text: str,
    index: int,
    is_compound: bool,
    intent_classifier: Optional[Callable[[str], str]],
) -> RequestUnit:
    normalized = text.lower().strip().rstrip(_TRAILING_SENTENCE_PUNCT)
    category, intent, routing_decision = _classify_broad_category(normalized)
    legacy_intent = intent_classifier(text) if intent_classifier is not None else None
    depends_on = (
        (index - 1,)
        if index > 0 and (
            intent == "EXPLAIN_RESULT"
            or _CREATE_FOLDER_FILE_WORKFLOW.search(original_text)
        )
        else ()
    )
    # Case-preserving (filenames are case-sensitive, e.g. "Brain.py"),
    # but with trailing sentence punctuation removed — computed once here
    # and reused by every extraction helper below, rather than each one
    # independently re-stripping (or, as before, failing to).
    extraction_text = text.strip().rstrip(_TRAILING_SENTENCE_PUNCT)
    target_reference = _build_target_reference(extraction_text, text, category, intent)
    destination_reference = _build_destination_reference(
        extraction_text, text, category, intent,
    )
    understanding = _build_understanding(
        extraction_text, category, intent, target_reference,
        destination_reference,
    )
    return RequestUnit(
        original_text=original_text,
        text=text,
        index=index,
        is_compound=is_compound,
        category=category,
        intent=intent,
        confidence="high" if category != RequestCategory.UNKNOWN else "low",
        legacy_intent=legacy_intent,
        routing_decision=routing_decision,
        context_references=_extract_context_references(normalized),
        depends_on=depends_on,
        target_reference=target_reference,
        destination_reference=destination_reference,
        understanding=understanding,
    )


def _classify_broad_category(text: str) -> tuple[str, str, str]:
    """Classify ownership before consulting the legacy flat intent router.

    Self-info ownership is delegated to the existing self-info query router,
    which remains the single source for self-info semantics.  Once ownership
    is known, it cannot be overwritten by an unrecognised legacy intent.
    """
    self_info = route_query(text)
    self_info_intents = {
        ROUTE_IDENTITY: "IDENTITY",
        ROUTE_CAPABILITIES: "CAPABILITIES",
        ROUTE_LIMITATIONS: "LIMITATIONS",
        ROUTE_ARCHITECTURE: "ARCHITECTURE",
        ROUTE_OPERATIONAL: "BRAINS" if self_info.sub_topic == "brains" else "OPERATIONAL",
    }
    if self_info.is_self_info and self_info.category in self_info_intents:
        return (
            RequestCategory.SELF_INFO,
            self_info_intents[self_info.category],
            RoutingDecision.DETERMINISTIC_SELF_INFO,
        )

    if _CONVERSATION_RE.search(text):
        return (
            RequestCategory.CONVERSATION,
            "CONVERSATION_CONTEXT",
            RoutingDecision.LLM_FALLBACK,
        )
    if _RESULT_EXPLANATION_RE.search(text):
        return RequestCategory.EXPLANATION, "EXPLAIN_RESULT", RoutingDecision.LLM_FALLBACK
    if _WORKSPACE_LIST_RE.search(text):
        return RequestCategory.WORKSPACE_QUERY, "WORKSPACE_LIST", RoutingDecision.LEGACY_INTENT_ROUTER
    if (
        _BARE_OPEN_TARGET_RE.search(text)
        or _FILE_OPERATION_RE.search(text)
        or _WRITE_FILE_RE.search(text)
        or _APPEND_FILE_RE.search(text)
        or _SEARCH_FILES_RE.search(text)
        or _DIRECTORY_CONTENTS_RE.search(text)
        or _EXPLAIN_FILE_RE.search(text)
        or _CONTEXTUAL_FILE_ACTION_RE.search(text)
    ):
        if re.search(r"\b(?:create|make)\b.*\b(?:folder|directory)\b", text):
            return RequestCategory.FILE_OPERATION, "CREATE_FOLDER", RoutingDecision.LEGACY_INTENT_ROUTER
        if _SEARCH_FILES_RE.search(text):
            return RequestCategory.FILE_OPERATION, "SEARCH_FILES", RoutingDecision.LEGACY_INTENT_ROUTER
        if re.search(r"\b(?:create|make)\b", text):
            return RequestCategory.FILE_OPERATION, "CREATE_FILE", RoutingDecision.LEGACY_INTENT_ROUTER
        if _DIRECTORY_CONTENTS_RE.search(text) or re.search(r"\b(?:list|show)\b.*\b(?:folder|directory|contents)\b", text):
            return RequestCategory.FILE_OPERATION, "LIST_DIRECTORY", RoutingDecision.LEGACY_INTENT_ROUTER
        if _EXPLAIN_FILE_RE.search(text):
            return RequestCategory.FILE_OPERATION, "EXPLAIN_FILE", RoutingDecision.LEGACY_INTENT_ROUTER
        if re.search(r"\b(?:delete|remove)\b", text):
            return RequestCategory.FILE_OPERATION, "DELETE_TARGET", RoutingDecision.LEGACY_INTENT_ROUTER
        if re.search(r"\brename\b", text):
            return RequestCategory.FILE_OPERATION, "RENAME_TARGET", RoutingDecision.LEGACY_INTENT_ROUTER
        if re.search(r"\bmove\b", text):
            return RequestCategory.FILE_OPERATION, "MOVE_TARGET", RoutingDecision.LEGACY_INTENT_ROUTER
        if re.search(r"\b(?:copy|duplicate)\b", text):
            return RequestCategory.FILE_OPERATION, "COPY_TARGET", RoutingDecision.LEGACY_INTENT_ROUTER
        if re.search(r"\b(?:read|access)\b", text):
            return RequestCategory.FILE_OPERATION, "READ_FILE", RoutingDecision.LEGACY_INTENT_ROUTER
        if _APPEND_FILE_RE.search(text) or re.search(r"\bappend\b", text):
            return RequestCategory.FILE_OPERATION, "APPEND_FILE", RoutingDecision.LEGACY_INTENT_ROUTER
        if re.search(r"\bwrite\b", text):
            return RequestCategory.FILE_OPERATION, "WRITE_FILE", RoutingDecision.LEGACY_INTENT_ROUTER
        if re.search(r"\bopen\b", text):
            return RequestCategory.FILE_OPERATION, "OPEN_TARGET", RoutingDecision.LEGACY_INTENT_ROUTER
        return RequestCategory.FILE_OPERATION, "FILE_OPERATION", RoutingDecision.LEGACY_INTENT_ROUTER
    if re.search(r"\binspect\b|\bshow\s+(?:information|info)\s+about\b", text):
        return RequestCategory.FILE_OPERATION, "INSPECT", RoutingDecision.LEGACY_INTENT_ROUTER
    if re.search(r"\b(?:remember|recall|memory|memories)\b", text):
        return RequestCategory.MEMORY, "MEMORY", RoutingDecision.LEGACY_INTENT_ROUTER
    if _WORKSPACE_SET_RE.search(text):
        return RequestCategory.WORKSPACE_QUERY, "WORKSPACE_SET", RoutingDecision.LEGACY_INTENT_ROUTER
    if _WORKSPACE_QUERY_RE.search(text):
        return RequestCategory.WORKSPACE_QUERY, "WORKSPACE_QUERY", RoutingDecision.LEGACY_INTENT_ROUTER
    if re.search(r"\b(?:tell|explain|describe)\b.*\b(?:what|which|the)\b.*"
                 r"\b(?:created|made|wrote|changed|result|results|it)\b", text):
        return RequestCategory.EXPLANATION, "EXPLAIN_RESULT", RoutingDecision.LLM_FALLBACK
    if re.search(r"\b(?:task|tasks|todo|to-do|complete|finish)\b", text):
        return RequestCategory.TASK, "TASK", RoutingDecision.LEGACY_INTENT_ROUTER
    if re.search(r"\b(?:project|milestone|version|phase|objective|progress)\b", text):
        return RequestCategory.PROJECT, "PROJECT", RoutingDecision.LEGACY_INTENT_ROUTER
    if text:
        return RequestCategory.GENERAL, "GENERAL_REASONING", RoutingDecision.LLM_FALLBACK
    return RequestCategory.UNKNOWN, "UNKNOWN", RoutingDecision.LLM_FALLBACK


def _extract_context_references(text: str) -> Tuple[str, ...]:
    """Preserve reference words for the conversation layer; never resolve them."""
    descriptors = (
        r"the\s+(?:file|folder|directory|thing|object)\s+i\s+"
        r"(?:created|made|opened|read|changed)\s+earlier",
        r"what\s+you\s+(?:created|made|wrote|changed)",
        r"the\s+(?:local|cloud|other)\s+one",
    )
    found = [
        match.group(0)
        for pattern in descriptors
        for match in re.finditer(pattern, text, re.IGNORECASE)
    ]
    found.extend(
        match.group(0)
        for match in re.finditer(r"\b(?:it|this|that|these|those)\b", text)
    )
    return tuple(found)


def _extract_scope(text: str) -> Optional[str]:
    """Extract a named scope like 'AG' or 'brains' when explicitly stated.
    The capture group's character class includes '.', so a trailing
    sentence period (already stripped from `text` upstream in the normal
    path, but kept here too for defense in depth) is trimmed explicitly
    rather than relying solely on the caller."""
    match = re.search(r"\bin\s+([A-Za-z0-9_.\-/\\]+)\s*$", text, re.IGNORECASE)
    if match:
        return match.group(1).strip(" \"'").rstrip(_TRAILING_SENTENCE_PUNCT)
    return None


def _infer_action_hint(text: str, intent: str) -> str:
    """Word-boundary matching throughout — the previous substring checks
    (e.g. `"make" in lower`) could false-positive inside unrelated words
    (e.g. "remake", "homemaker"); unlikely in short commands but cheap
    to make correct, consistent with the project's existing standard."""
    lower = text.lower()
    if re.search(r"\baccess\b", lower):
        return ActionHint.ACCESS
    if intent == "LIST_DIRECTORY":
        return ActionHint.LIST
    if intent in {"READ_FILE", "EXPLAIN_FILE", "OPEN_TARGET"} or re.search(r"\b(?:read|open|inspect|show)\b", lower):
        return ActionHint.READ
    if intent == "WRITE_FILE":
        return ActionHint.WRITE
    if intent == "APPEND_FILE":
        return ActionHint.APPEND
    if re.search(r"\b(?:list|inside|contents)\b", lower):
        return ActionHint.LIST
    if re.search(r"\b(?:create|make|write)\b", lower):
        if intent == "CREATE_FOLDER" or re.search(r"\b(?:folder|directory)\b", lower):
            return ActionHint.CREATE
        return ActionHint.WRITE if re.search(r"\bwrite\b", lower) else ActionHint.CREATE
    if re.search(r"\b(?:delete|remove)\b", lower):
        return ActionHint.DELETE
    if intent == "RENAME_TARGET":
        return ActionHint.RENAME_SOURCE
    if re.search(r"\b(?:copy|duplicate)\b", lower):
        return ActionHint.COPY_SOURCE
    if intent == "MOVE_TARGET" or re.search(r"\bmove\b", lower):
        return ActionHint.MOVE_SOURCE
    return ActionHint.UNSPECIFIED


def _infer_target_type(target: str, intent: str) -> str:
    """Use explicit operation semantics and unambiguous target syntax.

    Existing targets are probed by Target Resolver, which can override
    this hint with the actual filesystem type.
    """
    lowered = target.lower().strip()
    if intent in {"CREATE_FOLDER", "LIST_DIRECTORY"}:
        return TargetType.DIRECTORY
    if re.search(r"\bfile\b", lowered):
        return TargetType.FILE
    if intent in {"CREATE_FILE", "WRITE_FILE", "APPEND_FILE"}:
        return TargetType.FILE
    if re.search(r"\.[a-z0-9]{1,8}$", lowered):
        return TargetType.FILE
    if (re.search(r"\b(?:folder|directory)\b", lowered)
            or lowered.endswith(("/", "\\"))):
        return TargetType.DIRECTORY
    return TargetType.UNKNOWN


_OPERATION_PREFIXES = (
    "open the ", "read the ", "access the ", "append ",
) + tuple(f"{verb} " for verb in _FILE_OPERATION_CORE_VERBS) + (
    "show ", "inspect ", "find ", "search ", "list ", "navigate ", "go to ",
)


def _strip_operation_prefix(text: str) -> str:
    normalized = text.strip()
    for prefix in _OPERATION_PREFIXES:
        if normalized.lower().startswith(prefix):
            return normalized[len(prefix):].strip()
    return normalized


def _extract_target_text(text: str) -> str:
    clean = text.strip().strip('"\'')
    write_match = re.match(
        r"""^\s*(?:write|append|add)\s+(?:"([^"]*)"|'([^']*)')\s+(?:into|to|in)\s+(.+?)\s*$""",
        text,
        re.IGNORECASE,
    )
    if write_match:
        return write_match.group(3).strip().strip('"\'')
    generic_target = re.match(
        r"^\s*(?:read|open|access|write|list|show)\s+(?:the|a|an)\s+(file|folder|directory)\s*[?.!]*\s*$",
        text,
        re.IGNORECASE,
    )
    if generic_target:
        return "the " + generic_target.group(1).lower()
    content_match = re.match(
        r"^\s*(?:(?:show|tell)\s+me\s+)?(?:what(?:'s|\s+is)\s+inside|"
        r"(?:show|list)\s+(?:me\s+)?(?:the\s+)?contents\s+of)\s+(.+?)\s*$",
        clean, re.IGNORECASE,
    )
    if content_match:
        return content_match.group(1).strip().strip('"\'')

    explanation_match = re.match(
        r"^\s*(?:tell\s+me\s+about|explain|describe)\s+(.+\.[A-Za-z0-9]{1,8})\s*$",
        clean, re.IGNORECASE,
    )
    if explanation_match:
        return explanation_match.group(1).strip().strip('"\'')

    clean = _strip_operation_prefix(text)
    clean = re.sub(
        r"^\s*(?:tell\s+me\s+what(?:'s|\s+is)\s+inside|what(?:'s|\s+is)\s+inside)\s+",
        "", clean, flags=re.IGNORECASE,
    )
    scope_match = re.search(r"\s+in\s+([A-Za-z0-9_.\-/\\]+)\s*$", clean, re.IGNORECASE)
    if scope_match:
        clean = clean[:scope_match.start()].rstrip()
    if not clean:
        return clean

    if re.match(
        r"^(?:the\s+)?(?:file|folder|directory)\s+i\s+"
        r"(?:created|made|opened|read|changed)\s+earlier\b",
        clean, re.IGNORECASE,
    ):
        return clean if clean.lower().startswith("the ") else "the " + clean

    # "create a file called test.txt" / "...named test.txt" — without
    # this, the remainder after stripping just the leading verb is "a
    # file called test.txt", which would become the (wrong) target.
    # This mirrors the same fix already applied at the DirectoryControl/
    # Preprocessor-Brain layer, now applied at its source.
    called_match = _CALLED_NAMED_RE.search(clean)
    if called_match:
        name = called_match.group(1)
        if len(name) >= 2 and name[0] == name[-1] and name[0] in "\"'":
            name = name[1:-1]
        return name.strip()

    clean = re.sub(
        r"^(?:a\s+)?(?:file|folder|directory)\s+",
        "", clean, flags=re.IGNORECASE,
    )
    if re.fullmatch(r"(?:a\s+)?(?:file|folder|directory)", clean, re.IGNORECASE):
        return ""

    if re.search(r"\b(?:this|that|it|the file|the folder|that file|this file|the directory|that directory)\b", clean, re.IGNORECASE):
        return clean.strip()
    return clean.strip().strip('"\'')


def _build_understanding(
    text: str,
    category: str,
    intent: str,
    target_reference: Optional[TargetReference],
    destination_reference: Optional[TargetReference],
) -> RequestUnderstanding:
    target = target_reference.original_target if target_reference else None
    target_type = target_reference.target_type if target_reference else TargetType.UNKNOWN
    references = _extract_context_references(text)
    context_requirements: List[str] = []
    missing: List[str] = []
    action = intent
    domain = "GENERAL"
    outcome = DesiredOutcome.GENERAL_RESPONSE
    destination = (
        destination_reference.original_target
        if destination_reference is not None else None
    )
    content = None
    write_match = re.match(
        r"""^\s*(?:write|append|add)\s+(?:"([^"]*)"|'([^']*)')\s+(?:into|to|in)\s+.+?\s*$""",
        text,
        re.IGNORECASE,
    )
    if write_match:
        content = write_match.group(1) if write_match.group(1) is not None else write_match.group(2)

    if category == RequestCategory.SELF_INFO:
        domain, outcome = "SELF_INFO", DesiredOutcome.SELF_INFORMATION
    elif category == RequestCategory.CONVERSATION:
        domain, outcome = "CONVERSATION", DesiredOutcome.CONVERSATION_CONTEXT
    elif category == RequestCategory.MEMORY:
        domain, outcome = "MEMORY", DesiredOutcome.MEMORY_OPERATION
    elif category == RequestCategory.EXPLANATION:
        domain, outcome = "CONVERSATION", DesiredOutcome.EXPLAIN_RESULT
        if references or re.search(r"\b(?:created|made|wrote|changed|change|write|create)\b", text):
            references = ("previous request result",)
            context_requirements.append("previous request result")
    elif category == RequestCategory.WORKSPACE_QUERY:
        domain, outcome = "FILESYSTEM", DesiredOutcome.WORKSPACE_INFORMATION
    elif category == RequestCategory.FILE_OPERATION:
        domain = "FILESYSTEM"
        outcome_by_intent = {
            "CREATE_FILE": DesiredOutcome.CREATE_FILE,
            "CREATE_FOLDER": DesiredOutcome.CREATE_DIRECTORY,
            "WRITE_FILE": DesiredOutcome.WRITE_FILE,
            "APPEND_FILE": DesiredOutcome.APPEND_FILE,
            "SEARCH_FILES": DesiredOutcome.SEARCH_FILES,
            "LIST_DIRECTORY": DesiredOutcome.DIRECTORY_CONTENTS,
            "READ_FILE": DesiredOutcome.FILE_CONTENT,
            "EXPLAIN_FILE": DesiredOutcome.EXPLAIN_FILE,
            "DELETE_TARGET": DesiredOutcome.DELETE_TARGET,
            "RENAME_TARGET": DesiredOutcome.RENAME_TARGET,
            "MOVE_TARGET": DesiredOutcome.MOVE_TARGET,
            "COPY_TARGET": DesiredOutcome.COPY_TARGET,
            "OPEN_TARGET": (
                DesiredOutcome.OPEN_DIRECTORY
                if target_type == TargetType.DIRECTORY
                else DesiredOutcome.OPEN_TARGET
            ),
        }
        outcome = outcome_by_intent.get(intent, DesiredOutcome.OPEN_TARGET)
        action_by_intent = {
            "CREATE_FILE": "CREATE",
            "CREATE_FOLDER": "CREATE",
            "WRITE_FILE": "WRITE",
            "APPEND_FILE": "APPEND",
            "SEARCH_FILES": "SEARCH",
            "LIST_DIRECTORY": "LIST",
            "READ_FILE": "READ",
            "EXPLAIN_FILE": "EXPLAIN",
            "DELETE_TARGET": "DELETE",
            "RENAME_TARGET": "RENAME",
            "MOVE_TARGET": "MOVE",
            "COPY_TARGET": "COPY",
            "OPEN_TARGET": "OPEN",
        }
        action = action_by_intent.get(intent, intent)

        if (
            target_type == TargetType.UNKNOWN
            and intent == "OPEN_TARGET"
            and target_reference.resolution_mode != ResolutionMode.CONTEXT_REFERENCE
        ):
            explicit_file = bool(target and re.search(r"\.[A-Za-z0-9]{1,8}$", target))
            if not explicit_file:
                target_type = TargetType.DIRECTORY
                outcome = DesiredOutcome.OPEN_DIRECTORY

        if references:
            context_requirements.append("previously-mentioned filesystem entity")
        if target_reference is None or not target:
            missing.append("filesystem target")
        if intent in {"RENAME_TARGET", "MOVE_TARGET", "COPY_TARGET"} and not destination:
            missing.append("operation destination")
        elif references and target_reference.resolution_mode == ResolutionMode.CONTEXT_REFERENCE:
            context_requirements.append("resolve target from working memory")
    elif category in (RequestCategory.GENERAL, RequestCategory.UNKNOWN):
        outcome = DesiredOutcome.GENERAL_RESPONSE

    vague_reference = bool(re.search(
        r"\b(?:it|this|that|these|those|thing|one|something|object|item|stuff)\b",
        text,
        re.IGNORECASE,
    ))
    first_word = text.split(maxsplit=1)[0].lower() if text.split() else ""
    recognized_question_starts = {
        "what", "who", "where", "when", "why", "how", "tell", "describe",
        "explain", "show", "list", "create", "make", "write", "read", "open",
        "delete", "remove", "rename", "move", "copy", "find", "search",
        "remember", "recall", "add", "complete", "finish", "mark", "set",
        "switch", "inspect", "access", "check", "navigate", "go",
    }
    unclassified_imperative = bool(first_word and first_word not in recognized_question_starts)
    semantic_required = (
        category in (RequestCategory.GENERAL, RequestCategory.UNKNOWN)
        and vague_reference
        and (unclassified_imperative or intent == "UNKNOWN" or first_word in {
            "open", "read", "move", "copy", "rename", "create", "make",
        })
    )
    is_complete = not missing and not (
        target_reference is not None
        and target_reference.resolution_mode == ResolutionMode.CONTEXT_REFERENCE
    )
    return RequestUnderstanding(
        action=action,
        target=target,
        target_type=target_type,
        domain=domain,
        desired_outcome=outcome,
        destination=destination,
        references=references,
        context_requirements=tuple(dict.fromkeys(context_requirements)),
        missing_information=tuple(missing),
        is_complete=is_complete,
        semantic_reasoning_required=semantic_required,
        content=content,
    )


def _detect_resolution_mode(target: str, scope: Optional[str], text: str) -> str:
    """Absolute-path detection reuses Target Resolver's canonical
    `looks_absolute()` rather than an independent copy. The previous
    independent copy here additionally treated ANY occurrence of a
    backslash anywhere in the string as proof of an absolute path
    (`"\\\\" in target`) — which misclassified an ordinary relative
    Windows-style nested path like "backup\\test.txt" as ABSOLUTE_PATH.
    Since Target Resolver trusts resolution_mode rather than
    re-deriving it, that bug would have sent a plainly relative target
    through absolute resolution with no safety net downstream."""
    if not target.strip():
        return ResolutionMode.UNSPECIFIED
    if looks_absolute(target):
        return ResolutionMode.ABSOLUTE_PATH
    if scope:
        return ResolutionMode.NAMED_SCOPE
    if re.fullmatch(
        r"the\s+(?:file|folder|directory|thing|object)|"
        r"the\s+(?:older|newer|first|second|previous|last)\s+(?:file|folder|directory)",
        target.strip(),
        re.IGNORECASE,
    ):
        return ResolutionMode.CONTEXT_REFERENCE
    if re.search(
        r"\b(?:that|this|it|the\s+(?:file|folder|directory|thing|object)\s+i\s+"
        r"(?:created|made|opened|read|changed)\s+earlier|"
        r"the\s+(?:older|newer|first|second|previous|last)\s+(?:file|folder|directory))\b",
        text, re.IGNORECASE,
    ):
        return ResolutionMode.CONTEXT_REFERENCE
    return ResolutionMode.WORKSPACE_RELATIVE


def _build_target_reference(
    extraction_text: str, original_text: str, category: str, intent: str
) -> Optional[TargetReference]:
    """Convert a QUIPS filesystem request into the structured TargetReference
    contract. `extraction_text` is case-preserved (filenames are
    case-sensitive) but has trailing sentence punctuation already
    stripped; `original_text` is kept verbatim for `raw_text` only, so no
    information is lost even though it isn't used for extraction."""
    if category != RequestCategory.FILE_OPERATION:
        return None

    rename_parts = (
        _extract_rename_targets(extraction_text)
        if intent in {"RENAME_TARGET", "MOVE_TARGET", "COPY_TARGET"} else None
    )
    target = rename_parts[0] if rename_parts else _extract_target_text(extraction_text)
    if not target:
        return None

    scope = _extract_scope(extraction_text)
    resolution_mode = _detect_resolution_mode(target, scope, extraction_text)
    action = _infer_action_hint(extraction_text, intent)
    context: Dict[str, object] = {}

    if resolution_mode == ResolutionMode.CONTEXT_REFERENCE:
        context["reference_path"] = target
        context["reference_type"] = "contextual"

    target_type = _infer_target_type(target, intent)
    return TargetReference(
        original_target=target,
        target_type=target_type,
        scope=scope,
        resolution_mode=resolution_mode,
        action=action,
        context=context,
        raw_text=original_text,
    )


def _build_destination_reference(
    extraction_text: str,
    original_text: str,
    category: str,
    intent: str,
) -> Optional[TargetReference]:
    if category != RequestCategory.FILE_OPERATION or intent not in {
        "RENAME_TARGET", "MOVE_TARGET", "COPY_TARGET",
    }:
        return None
    parts = _extract_rename_targets(extraction_text)
    if parts is None or not parts[1]:
        return None
    destination = parts[1]
    scope = _extract_scope(extraction_text)
    return TargetReference(
        original_target=destination,
        target_type=_infer_target_type(destination, "UNKNOWN"),
        scope=scope,
        resolution_mode=_detect_resolution_mode(destination, scope, extraction_text),
        action={
            "RENAME_TARGET": ActionHint.RENAME_DESTINATION,
            "MOVE_TARGET": ActionHint.MOVE_DESTINATION,
            "COPY_TARGET": ActionHint.COPY_DESTINATION,
        }[intent],
        raw_text=original_text,
    )


def _extract_rename_targets(text: str) -> Optional[Tuple[str, str]]:
    clean = _strip_operation_prefix(text)
    scope_match = re.search(r"\s+in\s+([A-Za-z0-9_.\-/\\]+)\s*$", clean, re.IGNORECASE)
    if scope_match:
        clean = clean[:scope_match.start()].rstrip()
    match = re.match(r"^(.+?)\s+(?:to|as)\s+(.+?)\s*$", clean, re.IGNORECASE)
    if not match:
        return None

    source, destination = match.groups()
    source = re.sub(r"^(?:the\s+)?(?:file|folder|directory)\s+", "", source, flags=re.IGNORECASE)
    destination = re.sub(r"^(?:the\s+)?(?:file|folder|directory)\s+", "", destination, flags=re.IGNORECASE)
    source = source.strip().strip("\"'").rstrip(_TRAILING_SENTENCE_PUNCT)
    destination = destination.strip().strip("\"'").rstrip(_TRAILING_SENTENCE_PUNCT)
    return (source, destination) if source and destination else None


def resolve_target_reference(
    request: RequestUnit,
    *,
    workspace: Optional[str] = None,
    scope_registry: Optional[Dict[str, str]] = None,
    resolver: Optional[object] = None,
):
    """Resolve a file-operation RequestUnit through the Target Resolver contract."""
    if request.category != RequestCategory.FILE_OPERATION or request.target_reference is None:
        return None
    if resolver is not None:
        return resolver.resolve(request.target_reference)

    from target_resolver import TargetResolver

    target_resolver = TargetResolver(workspace_source=workspace, scope_registry=scope_registry)
    return target_resolver.resolve(request.target_reference)