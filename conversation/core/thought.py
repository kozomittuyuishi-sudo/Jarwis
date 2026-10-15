"""
core/thought.py
===============
AG Phase 2 — Thought Object

The Thought is the single object that flows through the entire cognitive
pipeline from the moment the user's message is received until the final
response is returned.

Every stage of the pipeline reads from and writes to a Thought.
Nothing is passed directly to the Brain except ``thought.resolved_input``.

Pipeline stages and their contract:
    1. Created by the main loop from raw user input.
    2. CognitiveEvaluator  — populates intent, confidence, missing_information,
                             requires_context, ask_for_clarification.
    3. ContextResolver     — populates resolved_input, candidate_contexts,
                             detected_entities.
    4. Brain               — receives only resolved_input; populates raw_response.
    5. ResponseVerifier    — validates raw_response; populates final_response,
                             verification_passed, verification_notes.
    6. WorkingMemory       — consumes the completed Thought to build a
                             ConversationFrame and update entity/context state.

Design principles:
    - All fields have safe defaults; no stage may crash if a prior stage
      left a field at its default.
    - The reasoning_log is append-only. Use ``thought.log(...)`` to add entries.
    - Mutation methods return ``self`` to allow chaining where convenient.
    - Deterministic, in-memory, stdlib only. No LLM calls.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid

from quips import RequestUnderstanding


class ResolvedEntity:
    """An entity reference resolved from structured conversational memory."""

    def __init__(
        self,
        name: str,
        entity_type: str,
        source_context: str,
        resolved_location: str,
        relationship: Optional[str] = None,
        source_action_id: Optional[str] = None,
    ) -> None:
        self.name = str(name)
        self.entity_type = str(entity_type)
        self.source_context = str(source_context)
        self.resolved_location = str(resolved_location)
        self.relationship = relationship
        self.source_action_id = source_action_id

    def to_dict(self) -> Dict[str, Optional[str]]:
        return {
            "name": self.name,
            "entity_type": self.entity_type,
            "source_context": self.source_context,
            "resolved_location": self.resolved_location,
            "relationship": self.relationship,
            "source_action_id": self.source_action_id,
        }


class Thought:
    """
    Central pipeline carrier object for one conversational turn.

    Attributes
    ----------
    thought_id : str
        Unique identifier for this thought (auto-generated).
    created_at : str
        ISO-8601 UTC timestamp when the thought was created.

    raw_input : str
        The exact string the user typed, unmodified.

    intent : Optional[str]
        Classified intent label (e.g. ``"unknown"``, ``"recall"``, ``"greeting"``).
        Set by CognitiveEvaluator.

    requires_context : bool
        True if the input cannot be answered without resolving contextual
        references (pronouns, ellipsis, implicit subjects, etc.).
        Set by CognitiveEvaluator.

    missing_information : List[str]
        Human-readable list of what is missing to answer the question.
        Example: ["subject of the question (who is 'he'?)"]
        Set by CognitiveEvaluator.

    detected_entities : List[Dict[str, Any]]
        Lightweight entity mentions extracted from the input.
        Each dict has at minimum: {"name": str, "type": str}.
        Set by CognitiveEvaluator and/or ContextResolver.

    candidate_contexts : List[Dict[str, Any]]
        Possible context interpretations produced by ContextResolver.
        Each dict has: {"label": str, "confidence": float, "source": str}.

    resolved_input : str
        The rewritten, context-enriched version of raw_input.
        This is the ONLY thing sent to the Brain.
        Set by ContextResolver; defaults to raw_input if no resolution needed.

    confidence : float
        Overall confidence in the resolved context (0.0–1.0).
        Set by ContextResolver / ConfidenceEngine.

    ask_for_clarification : bool
        True if confidence is too low to proceed and the user should be
        asked to clarify. When True the Brain is NOT invoked.
        Set by ContextResolver / ConfidenceEngine.

    clarification_question : Optional[str]
        The question AG should ask the user when ask_for_clarification is True.

    raw_response : Optional[str]
        The Brain's unmodified response text.
        Set after Brain execution.

    final_response : Optional[str]
        The verified, sanitized response ready for the user.
        Set by ResponseVerifier.

    verification_passed : bool
        True if ResponseVerifier approved the response.
        Set by ResponseVerifier.

    verification_notes : List[str]
        Any notes or warnings produced by ResponseVerifier.

    reasoning_log : List[Dict[str, str]]
        Internal step-by-step trace of the pipeline's reasoning.
        Each entry: {"step": str, "note": str, "timestamp": str}.
        Never shown to users unless DEBUG_MODE is active.
    """

    def __init__(self, raw_input: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.thought_id: str = f"thought_{uuid.uuid4().hex[:12]}"
        self.created_at: str = now

        # Input
        self.raw_input: str = str(raw_input).strip()

        # Evaluation (CognitiveEvaluator fills these)
        self.intent: Optional[str] = None
        self.requires_context: bool = False
        self.missing_information: List[str] = []
        self.detected_entities: List[Dict[str, Any]] = []
        self.request_understanding: Optional[RequestUnderstanding] = None
        self.resolved_entity: Optional[ResolvedEntity] = None
        self.is_complete: bool = True
        self.semantic_reasoning_required: bool = False

        # Context resolution (ContextResolver fills these)
        self.candidate_contexts: List[Dict[str, Any]] = []
        self.resolved_input: str = self.raw_input  # default: pass-through
        self.confidence: float = 1.0

        # Clarification gate (ConfidenceEngine / ContextResolver fill these)
        self.ask_for_clarification: bool = False
        self.clarification_question: Optional[str] = None

        # Brain output (Brain execution fills this)
        self.raw_response: Optional[str] = None

        # Verification output (ResponseVerifier fills these)
        self.final_response: Optional[str] = None
        self.verification_passed: bool = False
        self.verification_notes: List[str] = []

        # Internal trace — append-only, never shown to users
        self.reasoning_log: List[Dict[str, str]] = []

    # ------------------------------------------------------------------
    # Reasoning log
    # ------------------------------------------------------------------

    def log(self, step: str, note: str) -> "Thought":
        """
        Append a reasoning step to the internal trace.

        :param step: Short label for the pipeline stage, e.g. "evaluator".
        :param note: Human-readable description of what was determined.
        :returns: self (allows chaining).
        """
        entry: Dict[str, str] = {
            "step": str(step),
            "note": str(note),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.reasoning_log.append(entry)
        return self

    def get_log(self) -> List[Dict[str, str]]:
        """Return a copy of the reasoning log."""
        return deepcopy(self.reasoning_log)

    def format_log(self) -> str:
        """Return a human-readable multi-line trace of the reasoning log."""
        if not self.reasoning_log:
            return "[Reasoning log is empty]"
        lines = [f"[{e['step']}] {e['note']}" for e in self.reasoning_log]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Evaluation helpers (used by CognitiveEvaluator)
    # ------------------------------------------------------------------

    def set_intent(self, intent: str) -> "Thought":
        self.intent = str(intent)
        return self

    def flag_missing(self, description: str) -> "Thought":
        """Record a specific piece of missing information."""
        if description and description not in self.missing_information:
            self.missing_information.append(str(description))
        return self

    def add_entity(self, name: str, entity_type: str = "unknown", **kwargs: Any) -> "Thought":
        """Add a lightweight entity mention. kwargs forwarded as extra fields."""
        entry: Dict[str, Any] = {"name": str(name), "type": str(entity_type)}
        entry.update(kwargs)
        self.detected_entities.append(entry)
        return self

    # ------------------------------------------------------------------
    # Context resolution helpers (used by ContextResolver)
    # ------------------------------------------------------------------

    def add_candidate_context(self, label: str, confidence: float, source: str) -> "Thought":
        """Record a candidate context interpretation."""
        self.candidate_contexts.append({
            "label": str(label),
            "confidence": float(confidence),
            "source": str(source),
        })
        return self

    def resolve(self, resolved_input: str, confidence: float) -> "Thought":
        """
        Set the resolved prompt and overall confidence.
        Always prefer explicit calls to this method over direct attribute mutation.
        """
        self.resolved_input = str(resolved_input).strip() or self.raw_input
        self.confidence = max(0.0, min(1.0, float(confidence)))
        return self

    def request_clarification(self, question: str) -> "Thought":
        """Signal that the user should be asked for clarification."""
        self.ask_for_clarification = True
        self.clarification_question = str(question)
        return self

    # ------------------------------------------------------------------
    # Response helpers (used by Brain execution and ResponseVerifier)
    # ------------------------------------------------------------------

    def set_raw_response(self, text: str) -> "Thought":
        self.raw_response = str(text)
        return self

    def set_final_response(self, text: str, passed: bool, notes: Optional[List[str]] = None) -> "Thought":
        self.final_response = str(text)
        self.verification_passed = bool(passed)
        self.verification_notes = list(notes or [])
        return self

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Full snapshot of the Thought, safe for logging or storage."""
        return {
            "thought_id": self.thought_id,
            "created_at": self.created_at,
            "raw_input": self.raw_input,
            "intent": self.intent,
            "requires_context": self.requires_context,
            "missing_information": list(self.missing_information),
            "detected_entities": deepcopy(self.detected_entities),
            "request_understanding": (
                self.request_understanding.to_dict()
                if self.request_understanding is not None else None
            ),
            "resolved_entity": (
                self.resolved_entity.to_dict()
                if self.resolved_entity is not None else None
            ),
            "is_complete": self.is_complete,
            "semantic_reasoning_required": self.semantic_reasoning_required,
            "candidate_contexts": deepcopy(self.candidate_contexts),
            "resolved_input": self.resolved_input,
            "confidence": self.confidence,
            "ask_for_clarification": self.ask_for_clarification,
            "clarification_question": self.clarification_question,
            "raw_response": self.raw_response,
            "final_response": self.final_response,
            "verification_passed": self.verification_passed,
            "verification_notes": list(self.verification_notes),
            "reasoning_log": self.get_log(),
        }

    def __repr__(self) -> str:
        return (
            f"Thought(id={self.thought_id!r}, intent={self.intent!r}, "
            f"confidence={self.confidence:.2f}, "
            f"ask_clarification={self.ask_for_clarification})"
        )
