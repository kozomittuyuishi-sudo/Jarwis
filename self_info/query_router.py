"""
self_info/query_router.py
==========================
Self-Info Query Router

Classifies an incoming user query as a self-info query and determines
which part of the self-model is most relevant to answer it.

Design rules:
- No LLM calls. Pure string matching + keyword heuristics.
- Returns a QueryRoute with a category and optionally a sub-topic.
- is_self_info_query() is the gate — returns True only for queries
  that the Self-Info system should own.
- Stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import List, Optional


# ---------------------------------------------------------------------------
# Route categories (map 1-to-1 with self-model sections)
# ---------------------------------------------------------------------------

ROUTE_IDENTITY        = "identity"
ROUTE_ARCHITECTURE    = "architecture"
ROUTE_CAPABILITIES    = "capabilities"
ROUTE_LIMITATIONS     = "limitations"
ROUTE_OPERATIONAL     = "operational"
ROUTE_TASK_ASSESSMENT = "task_assessment"
ROUTE_GENERAL         = "general"     # catch-all self-info


# ---------------------------------------------------------------------------
# Query Route dataclass
# ---------------------------------------------------------------------------

@dataclass
class QueryRoute:
    """
    Result of routing a user query through the Self-Info router.

    Attributes
    ----------
    is_self_info : bool
        True when this query belongs to the Self-Info system.
    category : str
        Which section of the self-model to consult.
        One of: identity, architecture, capabilities, limitations,
                operational, task_assessment, general.
    original_query : str
        The normalised (lowercased, stripped) query text.
    sub_topic : str, optional
        A more specific hint within the category (e.g. "brains" inside
        operational, or "voice" inside limitations).
    assessment_target : str, optional
        For task_assessment routes — what the user wants to know if
        AB can do (the extracted task description).
    """
    is_self_info: bool
    category: str
    original_query: str
    sub_topic: Optional[str] = None
    assessment_target: Optional[str] = None


# ---------------------------------------------------------------------------
# Trigger sets — exact matches
# ---------------------------------------------------------------------------

_IDENTITY_EXACT: frozenset = frozenset({
    "who are you",
    "what are you",
    "what is ag",
    "what is ambient guidance",
    "tell me about yourself",
    "describe yourself",
    "introduce yourself",
    "what is your name",
    "what's your name",
    "what is your short name",
    "what's your short name",
    "what is your nickname",
    "what's your nickname",
    "what are you called",
    "what is your purpose",
    "what is your version",
    "what version are you",
    "what version is ag",
    "ag version",
    "current version",
    "what version are we on",
    "version",
    "what project is this",
})

_ARCHITECTURE_EXACT: frozenset = frozenset({
    "what are you made of",
    "what are your components",
    "show me your systems",
    "show your systems",
    "what is your architecture",
    "explain your architecture",
    "how are you built",
    "how do you work",
    "how does ag work",
    "how are you structured",
    "what systems do you have",
    "what modules are loaded",
    "what modules do you have",
    "what subsystems do you have",
    "runtime status",
    "system status",
    "how is ag structured",
})

_CAPABILITIES_EXACT: frozenset = frozenset({
    "what can you do",
    "what are your capabilities",
    "what are you capable of",
    "what features do you have",
    "what do you support",
    "what is implemented",
    "what functionality exists",
    "what can ag do",
    "list your capabilities",
    "show capabilities",
    "show your capabilities",
    "what can you currently do",
})

_LIMITATIONS_EXACT: frozenset = frozenset({
    "what are your limitations",
    "what can't you do",
    "what cant you do",
    "what are you missing",
    "what is not implemented",
    "what is missing",
    "what features are missing",
    "what is broken",
    "what doesnt work",
    "what doesn't work",
    "what are your weaknesses",
    "what can you not do",
    "what is not working",
    "what is incomplete",
})

_OPERATIONAL_EXACT: frozenset = frozenset({
    "what brains are available",
    "which brains do you have",
    "what brains do you have",
    "what brains are loaded",
    "show brains",
    "list brains",
    "which brain is available",
    "what ai models do you have",
    "what models do you have",
    "brain status",
    "current brain",
    "what brain are you using",
    "which brain are you using",
    "which brain is active",
    "what mode are you in",
    "current mode",
    "who is answering",
    "who is responding",
    "project status",
    "ag status",
    "current project",
    "what are we building",
    "what is the current phase",
    "what phase are you in",
})


# ---------------------------------------------------------------------------
# Keyword triggers — prefix / contains matches
# ---------------------------------------------------------------------------

_TASK_ASSESSMENT_PREFIXES: tuple = (
    "can you ",
    "could you ",
    "are you able to ",
    "do you support ",
    "is it possible for you to ",
    "can ag ",
    "could ag ",
    "would you be able to ",
    "are you capable of ",
    # Qualitative capability assessment patterns
    "how good are you with ",
    "how good are you at ",
    "how capable are you with ",
    "how capable are you at ",
    "how well can you ",
    "how well do you ",
    "how well are you ",
    "what are your limitations with ",
    "what are your limitations on ",
    "what are your limitations for ",
    "how proficient are you ",
    "how proficient are you at ",
    "how proficient are you with ",
    "how skilled are you ",
    "how skilled are you at ",
    "how skilled are you with ",
    "how effective are you ",
    "how effective are you at ",
    "how effective are you with ",
    "can you handle ",
    "can you perform ",
    "can you manage ",
    "can you do ",
    "are you good at ",
    "are you good with ",
    "how do you handle ",
    "how do you perform ",
    "what can you do with ",
)

_LIMITATIONS_KEYWORDS: tuple = (
    "limitation",
    "limitations",
    "can't do",
    "cannot do",
    "not able",
    "unable to",
    "not supported",
    "missing feature",
    "blocked feature",
    "not implemented",
    "what's missing",
    "whats missing",
    "weakness",
    "weaknesses",
    "not capable",
    "incapable",
)

_CAPABILITIES_KEYWORDS: tuple = (
    "capability",
    "capabilities",
    "feature",
    "what do you do",
    "what can you",
    "what does ag do",
)

_ARCHITECTURE_KEYWORDS: tuple = (
    "architecture",
    "structure",
    "module",
    "subsystem",
    "component",
    "pipeline",
    "how does it work",
    "how do you work",
    "built with",
    "how is ag",
)

_IDENTITY_KEYWORDS: tuple = (
    "who are you",
    "what are you",
    "your name",
    "short name",
    "nickname",
    "your version",
    "your purpose",
    "about yourself",
    "about you",
    "your identity",
)

# Semantic phrasing families.  These remain in the self-info subsystem rather
# than being duplicated by QUIPS or the legacy intent router.
_SEMANTIC_PATTERNS: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"\b(?:tell me )?who you are\b"), ROUTE_IDENTITY),
    (re.compile(r"\bwhat (?:are|were) you capable of\b|"
                r"\bwhat you(?:'| a)re capable of\b|"
                r"\bwhat you can do\b|"
                r"\bwhat functionality do you have\b|"
                r"\bwhat functions? do you support\b|"
                r"\bwhat can you help me with\b"), ROUTE_CAPABILITIES),
    (re.compile(r"\bhow you work(?: internally)?\b|"
                r"\bhow do you work(?: internally)?\b|"
                r"\bhow do you process requests\b|"
                r"\bwhat happens internally\b|"
                r"\bhow does your system work\b"), ROUTE_ARCHITECTURE),
    (re.compile(r"\bhow many brains do you have\b|\bwhich brains are available\b|"
                r"\bwhat brains do you have\b"), ROUTE_OPERATIONAL),
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_self_info_query(user_input: str) -> bool:
    """
    Return True if this query should be answered by the Self-Info system.

    Fast path — avoids the full routing overhead.
    """
    norm = _normalise(user_input)
    if any(pattern.search(norm) for pattern, _ in _SEMANTIC_PATTERNS):
        return True
    if norm in (
        _IDENTITY_EXACT
        | _ARCHITECTURE_EXACT
        | _CAPABILITIES_EXACT
        | _LIMITATIONS_EXACT
        | _OPERATIONAL_EXACT
    ):
        return True
    for prefix in _TASK_ASSESSMENT_PREFIXES:
        if norm.startswith(prefix):
            return True
    for kw in (_LIMITATIONS_KEYWORDS + _CAPABILITIES_KEYWORDS + _ARCHITECTURE_KEYWORDS + _IDENTITY_KEYWORDS):
        if kw in norm:
            return True
    return False


def route_query(user_input: str) -> QueryRoute:
    """
    Classify a user query and return a QueryRoute.

    If is_self_info is False, the caller should fall through to the
    normal pipeline.  When True, category tells the engine which
    section of the self-model to consult.
    """
    norm = _normalise(user_input)

    for pattern, category in _SEMANTIC_PATTERNS:
        if pattern.search(norm):
            return QueryRoute(
                is_self_info=True,
                category=category,
                original_query=norm,
                sub_topic="brains" if category == ROUTE_OPERATIONAL else None,
            )

    # --- Task assessment (highest priority — catches "can you X?") ---
    for prefix in _TASK_ASSESSMENT_PREFIXES:
        if norm.startswith(prefix):
            target = norm[len(prefix):].strip().rstrip("?").strip()
            return QueryRoute(
                is_self_info=True,
                category=ROUTE_TASK_ASSESSMENT,
                original_query=norm,
                assessment_target=target or norm,
            )

    # --- Exact match blocks ---
    if norm in _IDENTITY_EXACT:
        return QueryRoute(is_self_info=True, category=ROUTE_IDENTITY, original_query=norm)

    if norm in _ARCHITECTURE_EXACT:
        return QueryRoute(is_self_info=True, category=ROUTE_ARCHITECTURE, original_query=norm)

    if norm in _CAPABILITIES_EXACT:
        return QueryRoute(is_self_info=True, category=ROUTE_CAPABILITIES, original_query=norm)

    if norm in _LIMITATIONS_EXACT:
        return QueryRoute(is_self_info=True, category=ROUTE_LIMITATIONS, original_query=norm)

    if norm in _OPERATIONAL_EXACT:
        # Determine sub-topic
        sub = _extract_operational_subtopic(norm)
        return QueryRoute(is_self_info=True, category=ROUTE_OPERATIONAL, original_query=norm, sub_topic=sub)

    # --- Keyword / contains matching ---
    for kw in _LIMITATIONS_KEYWORDS:
        if kw in norm:
            return QueryRoute(is_self_info=True, category=ROUTE_LIMITATIONS, original_query=norm)

    for kw in _CAPABILITIES_KEYWORDS:
        if kw in norm:
            return QueryRoute(is_self_info=True, category=ROUTE_CAPABILITIES, original_query=norm)

    for kw in _ARCHITECTURE_KEYWORDS:
        if kw in norm:
            return QueryRoute(is_self_info=True, category=ROUTE_ARCHITECTURE, original_query=norm)

    for kw in _IDENTITY_KEYWORDS:
        if kw in norm:
            return QueryRoute(is_self_info=True, category=ROUTE_IDENTITY, original_query=norm)

    # Not a self-info query
    return QueryRoute(is_self_info=False, category=ROUTE_GENERAL, original_query=norm)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Lowercase, strip, remove trailing punctuation."""
    return re.sub(r"^(?:and|also|then)\s+", "", text.strip().lower()).rstrip("?!." )


def _extract_operational_subtopic(norm: str) -> Optional[str]:
    """Best-effort sub-topic extraction for operational queries."""
    if "brain" in norm or "model" in norm or "ai" in norm:
        return "brains"
    if "phase" in norm or "status" in norm or "version" in norm:
        return "project_status"
    if "mode" in norm:
        return "brain_mode"
    return None
