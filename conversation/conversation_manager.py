"""
AG Beta — Conversation Manager, Phase A.

One cohesive subsystem, one runtime file (same principle as
schema_processor.py / control_layer.py). Contains:

  1. Action Pattern Analyzer (pre-existing - brain-switch detection)
  2. Conversation state contract + ConversationManager class
  3. Reference resolution
  4. Topic tracking / topic shift detection
  5. Context building (for Working Memory / Executive to consume)
  6. Schema Processor compatibility (to_document)
  7. Module-level singleton + free-function public API

Deterministic. No LLM calls. No action execution. No brain switching.
Does not replace working_memory.py or executive.py - it produces a
context object they can consume. Stays entirely in-memory; nothing
here writes to memory.json or calls Storage Manager automatically.
"""

import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# ======================================================================
# 1. Action Pattern Analyzer (pre-existing, preserved as-is)
#
# Jarwis.py imports analyze_action_pattern() directly - kept unchanged so
# the existing brain-switch integration in Jarwis.py keeps working.
# ======================================================================

PERMANENT_PATTERNS = [
    r"^switch to (cloud|local)( brain)?$",
    r"^go (online|offline)$",
    r"^set (default )?brain to (cloud|local)$",
]

TEMPORARY_PATTERNS = [
    r"\buse (cloud|local) brain\b",
    r"\banswer this using (cloud|local) brain\b",
    r"\brespond with (cloud|local) brain\b",
]


def _new_action_id() -> str:
    return uuid.uuid4().hex[:8]


def _empty_result() -> Dict[str, Any]:
    return {
        "is_action_request": False,
        "primary_intent": "none",
        "actions": [],
        "confidence": 0.0,
        "missing_data": [],
        "requires_confirmation": False
    }


def _base_action(action_type: str, operation: str, scope: str, target, resolved_query: str) -> Dict[str, Any]:
    return {
        "id": _new_action_id(),
        "action_type": action_type,
        "operation": operation,
        "scope": scope,
        "target": target,
        "mode_change": action_type == "brain_mode_change",
        "topic": None,
        "resolved_query": resolved_query,
        "parameters": {},
        "constraints": {},
        "context_refs": [],
        "depends_on": [],
        "execution_policy": {"requires_confirmation": False},
        "confidence": 0.85
    }


def analyze_action_pattern(user_input: str, working_memory=None) -> Dict[str, Any]:
    text = str(user_input).strip().lower()

    # --- Permanent brain mode change ---
    for pattern in PERMANENT_PATTERNS:
        match = re.search(pattern, text)
        if match:
            groups = [g for g in match.groups() if g]
            target = "cloud" if ("cloud" in groups or "online" in groups) else "local"

            action = _base_action("brain_mode_change", "set_default_brain", "global", target, user_input)
            action["parameters"] = {"persistent": True}
            action["confidence"] = 0.95

            return {
                "is_action_request": True,
                "primary_intent": "brain_mode_change",
                "actions": [action],
                "confidence": 0.95,
                "missing_data": [],
                "requires_confirmation": False
            }

    # --- Temporary brain use (current thread/request only) ---
    for pattern in TEMPORARY_PATTERNS:
        match = re.search(pattern, text)
        if match:
            target = match.group(1)

            topic_match = re.search(r"\bfor\s+(.+)$", text)
            topic = topic_match.group(1).strip() if topic_match else None
            resolved_query = topic if topic else user_input

            if working_memory is not None:
                from memory.working_memory import resolve_thread_reference
                resolved_query = resolve_thread_reference(resolved_query, working_memory)

            action = _base_action("brain_routing", "temporary_brain_use", "current_thread", target, resolved_query)
            action["topic"] = topic
            action["parameters"] = {
                "brain": target,
                "temporary": True,
                "restore_previous_mode": True
            }
            action["execution_policy"] = {"requires_confirmation": True}
            action["confidence"] = 0.9

            return {
                "is_action_request": True,
                "primary_intent": "brain_routing",
                "actions": [action],
                "confidence": 0.9,
                "missing_data": [],
                "requires_confirmation": True
            }

    return _empty_result()


# ======================================================================
# 2. Conversation state contract + ConversationManager class
# ======================================================================

MAX_RECENT_TURNS = 12
VALID_STATES = {"active", "awaiting_confirmation", "awaiting_information", "idle"}

YES_WORDS = {"yes", "y", "yeah", "yep", "sure", "ok", "okay", "go ahead", "confirm", "confirmed"}
NO_WORDS = {"no", "n", "nope", "nah", "cancel", "stop", "don't", "dont"}
PENDING_ACTION_WORDS = {"do it", "do that", "proceed", "go for it", "run it", "execute it"}
PRONOUNS = {"it", "that", "this"}

SHIFT_MARKERS = ["new topic", "separately", "forget that", "different question"]

BRAIN_ROUTING_RE = re.compile(r"\buse (cloud|local) brain\b")
BROAD_REFERENCE_RE = re.compile(r"\ball this\b|\bthe above\b|\bthis discussion\b|\bthat discussion\b")

REFERENCE_TEMPLATES = [
    (re.compile(r"^give (me )?examples?$"), lambda topic: f"Give examples about {topic}."),
    (re.compile(r"^simplify (that|it|this)$"), lambda topic: f"Simplify the current explanation about {topic}."),
    (re.compile(r"^explain (that|it|this)$"), lambda topic: f"Explain the current explanation about {topic}."),
    (re.compile(r"^(more|tell me more)( about it)?$"), lambda topic: f"Tell me more about {topic}."),
    (re.compile(r"^why(\s+though)?$"), lambda topic: f"Why does {topic} happen?"),
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _truncate(text: str, max_len: int = 60) -> str:
    text = str(text).strip()
    return text if len(text) <= max_len else text[:max_len].rstrip() + "..."


class ConversationManager:
    """
    Tracks the active discussion: topic, thread, recent turns, pending
    action/confirmation state, and resolves contextual references before
    normal intent routing. Deterministic, in-memory, stdlib only.
    """

    def __init__(self):
        self._new_state()

    # -- lifecycle -----------------------------------------------------

    def _new_state(self) -> None:
        now = _now_iso()
        conversation_id = f"conv_{uuid.uuid4().hex[:12]}"

        self.conversation_id = conversation_id
        self.thread_id = f"thread_{uuid.uuid4().hex[:12]}"
        self.current_topic: Optional[str] = None
        self.previous_topic: Optional[str] = None
        self.topic_history: List[str] = []
        self.thread_summary: str = ""
        self.recent_turns: List[Dict[str, Any]] = []
        self.recent_entities: List[str] = []
        self.resolved_references: List[Dict[str, Any]] = []
        self.pending_action: Optional[Dict[str, Any]] = None
        self.pending_confirmation: Optional[Dict[str, Any]] = None
        self.state: str = "active"
        self.turn_count: int = 0
        self.created_at: str = now
        self.updated_at: str = now
        self._last_needs_clarification: bool = False

    def create_conversation(self) -> Dict[str, Any]:
        self._new_state()
        return self.get_state()

    def reset_conversation(self) -> Dict[str, Any]:
        return self.create_conversation()

    def get_state(self) -> Dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "thread_id": self.thread_id,
            "current_topic": self.current_topic,
            "previous_topic": self.previous_topic,
            "topic_history": list(self.topic_history),
            "thread_summary": self.thread_summary,
            "recent_turns": [dict(turn) for turn in self.recent_turns],
            "recent_entities": list(self.recent_entities),
            "resolved_references": [dict(ref) for ref in self.resolved_references],
            "pending_action": dict(self.pending_action) if self.pending_action else None,
            "pending_confirmation": dict(self.pending_confirmation) if self.pending_confirmation else None,
            "state": self.state,
            "turn_count": self.turn_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def _touch(self) -> None:
        self.updated_at = _now_iso()

    # -- turns -----------------------------------------------------------

    def add_turn(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        turn = {
            "role": role,
            "content": content,
            "timestamp": _now_iso(),
            "metadata": metadata or {},
        }
        self.recent_turns.append(turn)
        self.turn_count += 1

        while len(self.recent_turns) > MAX_RECENT_TURNS:
            self.recent_turns.pop(0)

        self._recompute_thread_summary()
        self._touch()
        return self.get_state()

    # -- topic tracking ----------------------------------------------------

    def set_topic(self, topic: str, confidence: float = 1.0) -> Dict[str, Any]:
        if topic and topic != self.current_topic:
            self.previous_topic = self.current_topic
            self.current_topic = topic
            self.topic_history.append(topic)

            if topic not in self.recent_entities:
                self.recent_entities.append(topic)
                self.recent_entities = self.recent_entities[-10:]

        self._recompute_thread_summary()
        self._touch()
        return self.get_state()

    def detect_topic_shift(self, user_input: str) -> Dict[str, Any]:
        text = str(user_input).strip().lower()

        back_to_match = re.search(r"\bback to\s+(.+)$", text)
        if back_to_match:
            candidate = back_to_match.group(1).strip().rstrip("?.! ")
            return {"shift_detected": True, "new_topic": candidate or None, "marker": "back to"}

        if "about ag" in text:
            return {"shift_detected": True, "new_topic": "ag", "marker": "about ag"}

        for marker in SHIFT_MARKERS:
            if marker in text:
                return {"shift_detected": True, "new_topic": None, "marker": marker}

        return {"shift_detected": False, "new_topic": None, "marker": None}

    # -- reference resolution ----------------------------------------------

    def _reference_result(self, resolved, original_input, resolved_input, reference_type,
                           topic, confidence, needs_clarification, reason) -> Dict[str, Any]:
        result = {
            "resolved": resolved,
            "original_input": original_input,
            "resolved_input": resolved_input,
            "reference_type": reference_type,
            "topic": topic,
            "confidence": confidence,
            "needs_clarification": needs_clarification,
            "reason": reason,
        }
        self._last_needs_clarification = needs_clarification
        self.resolved_references.append(dict(result))
        self.resolved_references = self.resolved_references[-20:]
        return result

    def resolve_reference(self, user_input: str) -> Dict[str, Any]:
        text = str(user_input).strip()
        lowered = text.lower().rstrip("?.! ")

        # Layer 1: pending confirmation / pending action state
        if self.pending_confirmation is not None and lowered in (YES_WORDS | NO_WORDS):
            decision = "yes" if lowered in YES_WORDS else "no"
            return self._reference_result(
                True, text, text, "confirmation_response", self.current_topic, 1.0, False,
                f"Pending confirmation active; interpreted as '{decision}'."
            )

        if self.pending_action is not None and lowered in PENDING_ACTION_WORDS:
            resolved_input = (
                self.pending_action.get("resolved_query")
                or self.pending_action.get("description")
                or str(self.pending_action)
            )
            return self._reference_result(
                True, text, resolved_input, "pending_action", self.current_topic, 1.0, False,
                "Resolved to the pending action."
            )

        # Layer 2: pronoun/reference detection via templates + topic
        if self.current_topic:
            for pattern, builder in REFERENCE_TEMPLATES:
                if pattern.match(lowered):
                    resolved_input = builder(self.current_topic)
                    return self._reference_result(
                        True, text, resolved_input, "topic_reference", self.current_topic, 0.9, False,
                        "Matched a known reference pattern."
                    )

            if BRAIN_ROUTING_RE.search(lowered) and BROAD_REFERENCE_RE.search(lowered):
                brain = "cloud" if "cloud" in lowered else "local"
                resolved_input = f"Create a briefing of the current {self.current_topic} discussion using {brain} brain."
                return self._reference_result(
                    True, text, resolved_input, "brain_routing_reference", self.current_topic, 0.85, False,
                    "Matched brain-routing combined with a broad thread reference."
                )

            # Layer 3: continuation-pattern detection (standalone pronoun substitution)
            words = lowered.split()
            if len(words) <= 6 and any(word in PRONOUNS for word in words):
                replaced = [self.current_topic if word in PRONOUNS else word for word in words]
                resolved_input = " ".join(replaced)
                return self._reference_result(
                    True, text, resolved_input, "pronoun_substitution", self.current_topic, 0.75, False,
                    "Substituted a standalone pronoun with the current topic."
                )

            # Layer 5: recent-turn / short-input fallback
            if len(words) <= 4:
                resolved_input = f"{text} (continuing the {self.current_topic} discussion)"
                return self._reference_result(
                    True, text, resolved_input, "recent_turn_fallback", self.current_topic, 0.5, False,
                    "Short input treated as a loose continuation of the current topic."
                )

        # Layer 6: unresolved
        needs_clarification = self.current_topic is None
        reason = (
            "No topic or pending state available to resolve this reference."
            if needs_clarification else
            "Input does not match a known reference pattern; treated as standalone."
        )
        return self._reference_result(False, text, text, "none", self.current_topic, 0.0, needs_clarification, reason)

    # -- pending action / confirmation --------------------------------------

    def set_pending_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        self.pending_action = dict(action) if action else None
        self._recompute_thread_summary()
        self._touch()
        return self.get_state()

    def clear_pending_action(self) -> None:
        self.pending_action = None
        if self.state == "awaiting_information":
            self.state = "active"
        self._recompute_thread_summary()
        self._touch()

    def set_pending_confirmation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.pending_confirmation = dict(data) if data else None
        self.state = "awaiting_confirmation"
        self._touch()
        return self.get_state()

    def clear_pending_confirmation(self) -> None:
        self.pending_confirmation = None
        self.state = "active"
        self._touch()

    # -- context building ----------------------------------------------------

    def build_context(self, user_input: str) -> Dict[str, Any]:
        shift = self.detect_topic_shift(user_input)
        if shift["shift_detected"] and shift["new_topic"]:
            self.set_topic(shift["new_topic"])

        reference_resolution = self.resolve_reference(user_input)
        resolved_input = reference_resolution["resolved_input"] if reference_resolution["resolved"] else user_input

        return {
            "conversation_id": self.conversation_id,
            "thread_id": self.thread_id,
            "current_topic": self.current_topic,
            "thread_summary": self.thread_summary,
            "recent_turns": [dict(turn) for turn in self.recent_turns],
            "recent_entities": list(self.recent_entities),
            "pending_action": dict(self.pending_action) if self.pending_action else None,
            "pending_confirmation": dict(self.pending_confirmation) if self.pending_confirmation else None,
            "resolved_input": resolved_input,
            "reference_resolution": reference_resolution,
            "conversation_state": self.state,
        }

    # -- internal helpers -------------------------------------------------

    def _recompute_thread_summary(self) -> None:
        parts = []

        if self.current_topic:
            parts.append(f"Topic: {self.current_topic}")

        last_user = next((t["content"] for t in reversed(self.recent_turns) if t["role"] == "user"), None)
        if last_user:
            parts.append(f"Last request: {_truncate(last_user)}")

        last_assistant = next((t["content"] for t in reversed(self.recent_turns) if t["role"] == "assistant"), None)
        if last_assistant:
            parts.append(f"Last conclusion: {_truncate(last_assistant)}")

        if self.pending_action:
            parts.append("Pending action awaiting confirmation." if self.pending_confirmation else "Pending action queued.")

        self.thread_summary = " | ".join(parts)

    # -- Schema Processor compatibility --------------------------------------

    def to_document(self) -> Dict[str, Any]:
        """
        Returns a dict compatible with schema_processor's "conversation"
        schema. NOT written to Storage Manager automatically - the caller
        decides whether/when to persist it.
        """
        return {
            "current_topic": self.current_topic or "unspecified",
            "turns": [dict(turn) for turn in self.recent_turns],
            "thread_summary": self.thread_summary,
            "entities": list(self.recent_entities),
            "references": [
                ref["resolved_input"] for ref in self.resolved_references if ref.get("resolved")
            ][-10:],
            "pending_clarification": bool(self._last_needs_clarification),
        }


# ======================================================================
# 3. Module-level singleton + free-function public API
#
# Matches the exact signatures requested (no explicit manager argument),
# operating on one shared default ConversationManager instance. Code that
# wants an isolated instance (e.g. tests) should instantiate
# ConversationManager() directly instead of using these functions.
# ======================================================================

_default_manager = ConversationManager()


def create_conversation() -> Dict[str, Any]:
    return _default_manager.create_conversation()


def get_state() -> Dict[str, Any]:
    return _default_manager.get_state()


def add_turn(role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return _default_manager.add_turn(role, content, metadata)


def set_topic(topic: str, confidence: float = 1.0) -> Dict[str, Any]:
    return _default_manager.set_topic(topic, confidence)


def detect_topic_shift(user_input: str) -> Dict[str, Any]:
    return _default_manager.detect_topic_shift(user_input)


def resolve_reference(user_input: str) -> Dict[str, Any]:
    return _default_manager.resolve_reference(user_input)


def build_context(user_input: str) -> Dict[str, Any]:
    return _default_manager.build_context(user_input)


def set_pending_action(action: Dict[str, Any]) -> Dict[str, Any]:
    return _default_manager.set_pending_action(action)


def clear_pending_action() -> None:
    _default_manager.clear_pending_action()


def set_pending_confirmation(data: Dict[str, Any]) -> Dict[str, Any]:
    return _default_manager.set_pending_confirmation(data)


def clear_pending_confirmation() -> None:
    _default_manager.clear_pending_confirmation()


def reset_conversation() -> Dict[str, Any]:
    return _default_manager.reset_conversation()


def to_document() -> Dict[str, Any]:
    return _default_manager.to_document()
