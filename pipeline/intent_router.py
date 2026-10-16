"""
pipeline/intent_router.py
==========================
AG Intent Router — deterministic, rule-based intent classification.

Classifies a raw user input string into exactly one of:
    LOCAL_STATE_QUERY, DIRECTORY_OP, FILE_OP, SEARCH_OP,
    CAPABILITY_QUERY, GENERAL_QUERY

Also extracts the primary target (filename/path/foldername),
secondary target (for move/copy/rename), and content (for write/append)
using the NL entity extractor.

Design rules
------------
- Fully standalone: no dependency on Jarwis.py.
- No LLM calls, no network calls.
- Stdlib only.
- Python 3.12 compatible.
- All classification is done before calling the entity extractor to
  avoid redundant work.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from pipeline.nl_entity_extractor import extract_target


# ---------------------------------------------------------------------------
# Intent enum
# ---------------------------------------------------------------------------

class Intent(str, Enum):
    LOCAL_STATE_QUERY = "LOCAL_STATE_QUERY"
    DIRECTORY_OP      = "DIRECTORY_OP"
    FILE_OP           = "FILE_OP"
    SEARCH_OP         = "SEARCH_OP"
    CAPABILITY_QUERY  = "CAPABILITY_QUERY"
    GENERAL_QUERY     = "GENERAL_QUERY"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class IntentResult:
    """The output of IntentRouter.route()."""

    intent: str
    """One of the Intent enum values."""

    sub_op: Optional[str] = None
    """Sub-operation within DIRECTORY_OP or FILE_OP (e.g. 'switch', 'read')."""

    target: Optional[str] = None
    """Primary target — filename, folder name, or path."""

    secondary_target: Optional[str] = None
    """Secondary target — destination for move/copy/rename."""

    content: Optional[str] = None
    """Content payload — used for write/append/create with initial content."""

    raw_input: str = ""
    """The original, unmodified input string."""


# ---------------------------------------------------------------------------
# Static lookup tables
# ---------------------------------------------------------------------------

# Exact-match phrases for LOCAL_STATE_QUERY
_LOCAL_STATE_EXACT: frozenset[str] = frozenset({
    "what is my current active directory",
    "what is the active directory",
    "what is the active directory now",
    "what active directory",
    "what directory am i in",
    "what folder am i in",
    "show active directory",
    "show current directory",
    "current workspace",
    "active workspace",
    "where am i",
    "show workspace",
    "show directory",
    "what is the current directory",
    "what is my active directory",
    "what is my current directory",
    # Phase 3 additions — natural location queries
    "where are you working",
    "what is the current workspace",
    "what is my current workspace",
    "what workspace am i in",
    "what is my active workspace",
    "what is my current active directory",
    "current directory",
    "where are we",
    "what directory are we in",
    "what folder are we in",
})

# Exact-match phrases for CAPABILITY_QUERY
_CAPABILITY_EXACT: frozenset[str] = frozenset({
    "what are your capabilities",
    "what can you do",
    "what are you capable of",
    "who are you",
    "what are you",
    "what is ag",
    "how do you work",
    "how does ag work",
    "what are your limitations",
    "what cant you do",
    "what can't you do",
    "what systems do you have",
    "what modules are loaded",
    "what modules do you have",
    "what version are you",
    "what version is ag",
    "ag version",
    "what brains are available",
    "which brains do you have",
    "what brains do you have",
    "tell me about yourself",
    "describe yourself",
    "what are you made of",
    "what are your components",
    "show me your systems",
    "show your systems",
    "runtime status",
    "system status",
    "what is your architecture",
    "explain your architecture",
    "how are you built",
    "what is your version",
    "what is your name",
    "what is your purpose",
    "how are you structured",
    "what is ambient guidance",
    "what are you missing",
    "what is not implemented",
    "what is incomplete",
})

# Exact-match phrases for DIRECTORY_OP sub_op='clear'
_DIR_CLEAR_EXACT: frozenset[str] = frozenset({
    "clear active directory",
    "clear the active directory",
    "clear workspace",
    "clear the workspace",
    "remove workspace",
    "remove the workspace",
    "unset workspace",
    "forget workspace",
    "clear active workspace",
    "clear the active workspace",
    "remove active directory",
    "remove the active directory",
})

# Exact-match phrases for DIRECTORY_OP sub_op='list'
_DIR_LIST_EXACT: frozenset[str] = frozenset({
    "what is here",
    "list everything",
    "list everything in current directory",
    "list everything in the current directory",
})

# Prefix tables: (prefix, intent, sub_op)
# Checked in order; first match wins.
_PREFIX_RULES: list[tuple[str, str, str]] = [
    # Navigation verbs must precede the broader "move " file-operation rule.
    ("go to ",                 Intent.DIRECTORY_OP, "navigate"),
    ("navigate to ",           Intent.DIRECTORY_OP, "navigate"),
    ("move into ",             Intent.DIRECTORY_OP, "navigate"),
    # DIRECTORY_OP — create folder (must appear BEFORE generic create_file rules)
    ("create a folder called ",    Intent.DIRECTORY_OP, "create_folder"),
    ("create a folder named ",     Intent.DIRECTORY_OP, "create_folder"),
    ("create folder ",             Intent.DIRECTORY_OP, "create_folder"),
    ("create a directory called ", Intent.DIRECTORY_OP, "create_folder"),
    ("create a directory named ",  Intent.DIRECTORY_OP, "create_folder"),
    ("create directory ",          Intent.DIRECTORY_OP, "create_folder"),
    ("make a folder called ",      Intent.DIRECTORY_OP, "create_folder"),
    ("make a folder named ",       Intent.DIRECTORY_OP, "create_folder"),
    ("make folder ",               Intent.DIRECTORY_OP, "create_folder"),
    ("mkdir ",                     Intent.DIRECTORY_OP, "create_folder"),
    # FILE_OP
    ("read file ",            Intent.FILE_OP,      "read"),
    ("open file ",            Intent.FILE_OP,      "read"),
    ("read ",                 Intent.FILE_OP,      "read"),
    ("create a file called ", Intent.FILE_OP,      "create_file"),
    ("create a file named ",  Intent.FILE_OP,      "create_file"),
    ("create file ",          Intent.FILE_OP,      "create_file"),
    ("make a file called ",   Intent.FILE_OP,      "create_file"),
    ("make a file named ",    Intent.FILE_OP,      "create_file"),
    ("make file ",            Intent.FILE_OP,      "create_file"),
    ("write to ",             Intent.FILE_OP,      "write"),
    ("write this to ",        Intent.FILE_OP,      "write"),
    ("append to ",            Intent.FILE_OP,      "append"),
    ("append this to ",       Intent.FILE_OP,      "append"),
    ("update ",               Intent.FILE_OP,      "update"),
    ("move ",                 Intent.FILE_OP,      "move"),
    ("copy ",                 Intent.FILE_OP,      "copy"),
    ("rename ",               Intent.FILE_OP,      "rename"),
    ("rename file ",          Intent.FILE_OP,      "rename"),
    # DIRECTORY_OP — switch
    ("switch to ",            Intent.DIRECTORY_OP, "switch"),
    ("cd ",                   Intent.DIRECTORY_OP, "switch"),
    ("change directory to ",  Intent.DIRECTORY_OP, "switch"),
    # DIRECTORY_OP — set
    ("work in ",              Intent.DIRECTORY_OP, "set"),
    ("use ",                  Intent.DIRECTORY_OP, "set"),
    ("set workspace to ",     Intent.DIRECTORY_OP, "set"),
    # DIRECTORY_OP — list
    ("list files",            Intent.DIRECTORY_OP, "list"),
    ("list folders",          Intent.DIRECTORY_OP, "list"),
    ("show files",            Intent.DIRECTORY_OP, "list"),
    ("show folders",          Intent.DIRECTORY_OP, "list"),
    # SEARCH_OP
    ("search for ",           Intent.SEARCH_OP,    None),
    ("find ",                 Intent.SEARCH_OP,    None),
    ("search ",               Intent.SEARCH_OP,    None),
]

# Two-target prefix rules: (prefix, " to ", sub_op) — for move/copy/rename
_TWO_TARGET_OPS: tuple[str, ...] = ("move", "copy", "rename", "rename file")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class IntentRouter:
    """
    Deterministic rule-based intent classifier and entity extractor.

    Usage
    -----
    ::

        router = IntentRouter()
        result = router.route("create a file called notes.txt")
        # result.intent  → 'FILE_OP'
        # result.sub_op  → 'create_file'
        # result.target  → 'notes.txt'
    """

    def route(
        self,
        user_input: str,
        base_path: Optional[Path] = None,
    ) -> IntentResult:
        """
        Classify ``user_input`` and extract entities.

        Parameters
        ----------
        user_input : str
            The raw user message.
        base_path : Path, optional
            Active workspace path; passed through to the entity extractor
            for relative-path resolution.

        Returns
        -------
        IntentResult
        """
        raw = user_input
        normalized = user_input.strip().lower().rstrip()
        normalized = re.sub(r"[?!]+$", "", normalized).rstrip()
        if normalized.endswith(".") and not normalized.endswith(".."):
            normalized = normalized[:-1].rstrip()

        # ----------------------------------------------------------------
        # 1. LOCAL_STATE_QUERY (known phrases and normalized interrogatives)
        # ----------------------------------------------------------------
        if normalized in _LOCAL_STATE_EXACT or re.fullmatch(
            r"what\s+is\s+(?:my|your|the)\s+(?:current|active)\s+"
            r"(?:workspace|directory)",
            normalized,
        ):
            return IntentResult(
                intent=Intent.LOCAL_STATE_QUERY,
                raw_input=raw,
            )

        _PARENT_EXACT: frozenset[str] = frozenset({
            "go back", "go up", "cd ..", "back", "..",
            "go to parent", "go to ..",
            "navigate to parent", "navigate to ..",
            "move to parent", "move to ..",
        })
        if normalized in _PARENT_EXACT:
            return IntentResult(
                intent=Intent.DIRECTORY_OP,
                sub_op="go_back",
                raw_input=raw,
            )

        # ----------------------------------------------------------------
        # 2. CAPABILITY_QUERY (exact match)
        # ----------------------------------------------------------------
        if normalized in _CAPABILITY_EXACT:
            return IntentResult(
                intent=Intent.CAPABILITY_QUERY,
                raw_input=raw,
            )

        # ----------------------------------------------------------------
        # 3. DIRECTORY_OP — clear (exact match)
        # ----------------------------------------------------------------
        if normalized in _DIR_CLEAR_EXACT:
            return IntentResult(
                intent=Intent.DIRECTORY_OP,
                sub_op="clear",
                raw_input=raw,
            )

        # ----------------------------------------------------------------
        # 4. DIRECTORY_OP — list (exact match)
        # ----------------------------------------------------------------
        if normalized in _DIR_LIST_EXACT:
            return IntentResult(
                intent=Intent.DIRECTORY_OP,
                sub_op="list",
                raw_input=raw,
            )

        # ----------------------------------------------------------------
        # 5. Prefix-based rules
        # ----------------------------------------------------------------
        for prefix, intent, sub_op in _PREFIX_RULES:
            if normalized.startswith(prefix):
                remainder = normalized[len(prefix):]
                original_remainder = raw.strip()[len(prefix):] if len(raw.strip()) >= len(prefix) else raw[len(prefix):]

                # For two-target ops (move, copy, rename), split on " to "
                if sub_op in ("move", "copy", "rename") and " to " in remainder:
                    parts = remainder.split(" to ", 1)
                    target_raw = parts[0].strip()
                    secondary_raw = parts[1].strip()
                    extracted = extract_target(target_raw, base_path)
                    sec_extracted = extract_target(secondary_raw, base_path)
                    return IntentResult(
                        intent=str(intent.value),
                        sub_op=sub_op,
                        target=extracted.get("target") or target_raw or None,
                        secondary_target=sec_extracted.get("target") or secondary_raw or None,
                        raw_input=raw,
                    )

                # For DIRECTORY_OP 'set', only match path-like remainders
                if sub_op == "set":
                    if not _looks_like_path(remainder):
                        # "use" or "work in" without a path → fall through to GENERAL_QUERY
                        continue

                # Standard single-target extraction
                # Use the original-casing remainder for entity extraction
                original_remainder = _slice_original(raw, prefix)
                extracted = extract_target(original_remainder, base_path)

                # For list sub_op, target may be empty (list current dir)
                target = extracted.get("target") or None
                content = extracted.get("content") or None

                return IntentResult(
                    intent=str(intent.value),
                    sub_op=sub_op,
                    target=target,
                    content=content,
                    raw_input=raw,
                )

        # ----------------------------------------------------------------
        # 6. go_back / parent detection (all phrasings that weren't
        #    captured by prefix rules above, because "go to .." doesn't
        #    start with "go to " without a space-separated target)
        # ----------------------------------------------------------------
        _PARENT_EXACT: frozenset[str] = frozenset({
            "go back", "go up", "cd ..", "back", "..",
            "go to parent", "go to ..",
            "navigate to parent", "navigate to ..",
            "move to parent", "move to ..",
        })
        if normalized in _PARENT_EXACT:
            return IntentResult(
                intent=Intent.DIRECTORY_OP,
                sub_op="go_back",
                raw_input=raw,
            )

        # ----------------------------------------------------------------
        # 7. GENERAL_QUERY fallback
        # ----------------------------------------------------------------
        return IntentResult(
            intent=Intent.GENERAL_QUERY,
            raw_input=raw,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _looks_like_path(text: str) -> bool:
    """
    Return True if text appears to be a filesystem path.

    Accepts: absolute paths (e.g. C:\\, /), relative paths starting with
    ./ or ../, UNC paths, or strings containing path separators.
    """
    t = text.strip()
    if not t:
        return False
    return (
        (":" in t and len(t) > 2)          # Windows drive letter (C:\...)
        or t.startswith("\\")              # UNC or absolute backslash
        or t.startswith("/")               # Unix absolute
        or t.startswith("./")              # Explicit relative
        or t.startswith(".\\")
        or t.startswith("../")             # Parent-relative
        or t.startswith("..\\")
        or "/" in t                        # Any forward slash
        or "\\" in t                       # Any backslash
    )


def _slice_original(raw: str, prefix_lower: str) -> str:
    """
    Slice the original-casing ``raw`` string after the lowercase ``prefix_lower``.

    Falls back to the raw string if the length check fails.
    """
    stripped = raw.strip()
    if len(stripped) >= len(prefix_lower):
        return stripped[len(prefix_lower):]
    return stripped


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_default_router: Optional[IntentRouter] = None


def get_intent_router() -> IntentRouter:
    """Return the shared IntentRouter instance (created on first call)."""
    global _default_router
    if _default_router is None:
        _default_router = IntentRouter()
    return _default_router


def route_intent(
    user_input: str,
    base_path: Optional[Path] = None,
) -> IntentResult:
    """
    Module-level convenience function.

    Equivalent to ``get_intent_router().route(user_input, base_path)``.
    """
    return get_intent_router().route(user_input, base_path)


def classify(
    user_input: str,
    base_path: Optional[Path] = None,
) -> IntentResult:
    """
    Module-level convenience alias for route_intent().

    Equivalent to ``get_intent_router().route(user_input, base_path)``.
    Preferred name for external callers and tests.
    """
    return get_intent_router().route(user_input, base_path)
