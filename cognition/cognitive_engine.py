"""
cognitive_engine.py
===================
AG Cognitive Engine (Phase A)

Role:
- Serves as AG's prefrontal reasoning pipeline sitting between Working Memory
  and Brain execution.
- Consumes Working Memory context (`build_context()`) to deterministically classify
  intent, assess cognitive modes, frame brain demands, evaluate scratchpad candidates,
  and assemble execution payloads.

Boundaries:
- MUST NOT invoke external LLM APIs directly.
- MUST NOT write to persistent storage files (memory.json, tasks.json, etc.).
- MUST NOT perform direct system actions or override Control Layer security rules.
"""

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


class CognitiveMode:
    QUERY = "QUERY"            # Informational / conversational requests
    ACTION = "ACTION"          # Operations, commands, or system changes
    PLANNING = "PLANNING"      # Multi-step breakdowns, strategy formation
    CLARIFICATION = "CLARIFICATION"  # Unresolved references or missing parameters


class CognitiveEngine:
    """Prefrontal reasoning and orchestration engine for AG."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        # Keywords suggesting an explicit system or file action
        self._action_keywords = {
            "delete", "remove", "execute", "run", "create", "write",
            "modify", "update", "clear", "reset", "git", "push", "commit"
        }
        # Keywords suggesting multi-step planning
        self._planning_keywords = {
            "plan", "roadmap", "steps", "milestones", "breakdown",
            "architecture", "design", "strategy", "phase"
        }

    # ------------------------------------------------------------------
    # 1. Context Ingestion
    # ------------------------------------------------------------------

    def ingest_context(self, context_snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ingests context built by `WorkingMemory.build_context()`.
        Validates structure and normalizes into standard Cognitive State.
        """
        if not isinstance(context_snapshot, dict):
            context_snapshot = {}

        return {
            "session_id": context_snapshot.get("session_id"),
            "objective": context_snapshot.get("active_objective"),
            "task": context_snapshot.get("active_task"),
            "topic": context_snapshot.get("active_topic"),
            "facts": deepcopy(context_snapshot.get("temporary_facts", {})),
            "memories": deepcopy(context_snapshot.get("cached_memories", [])),
            "scratchpad": deepcopy(context_snapshot.get("scratchpad", [])),
            "decisions": deepcopy(context_snapshot.get("decisions", {})),
            "pending_confirmation": deepcopy(context_snapshot.get("pending_confirmation")),
            "pending_action": deepcopy(context_snapshot.get("pending_action")),
            "pending_questions": deepcopy(context_snapshot.get("pending_questions", [])),
            "conversation_reference": deepcopy(context_snapshot.get("conversation_reference", {})),
            "action_events": deepcopy(context_snapshot.get("action_events", [])),
            "ingested_at": datetime.now(timezone.utc).isoformat()
        }

    # ------------------------------------------------------------------
    # 2. Deterministic Intent & Mode Classification
    # ------------------------------------------------------------------

    def classify_intent(self, user_input: str, cog_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classifies operational mode (QUERY, ACTION, PLANNING, CLARIFICATION)
        and determines if probe simulation or clarification is required.
        """
        text_lower = user_input.lower().strip() if user_input else ""

        # 1. Check for pending confirmations or unresolved questions
        if cog_state.get("pending_confirmation") or cog_state.get("pending_questions"):
            return {
                "mode": CognitiveMode.CLARIFICATION,
                "requires_simulation": False,
                "confidence": 0.95,
                "reason": "Active pending confirmation or questions require resolution."
            }

        # 2. Action Detection (check substrings or pending action)
        matched_actions = [kw for kw in self._action_keywords if kw in text_lower]
        if matched_actions or cog_state.get("pending_action"):
            return {
                "mode": CognitiveMode.ACTION,
                "requires_simulation": True,
                "action_triggers": matched_actions,
                "confidence": 0.90,
                "reason": f"Action keywords detected: {matched_actions}"
            }

        # 3. Planning Detection (check substrings or active objective)
        matched_planning = [kw for kw in self._planning_keywords if kw in text_lower]
        if matched_planning or cog_state.get("objective"):
            if matched_planning:
                return {
                    "mode": CognitiveMode.PLANNING,
                    "requires_simulation": False,
                    "planning_triggers": matched_planning,
                    "confidence": 0.85,
                    "reason": f"Planning keywords detected: {matched_planning}"
                }

        # 4. Default to Query Mode
        return {
            "mode": CognitiveMode.QUERY,
            "requires_simulation": False,
            "confidence": 0.80,
            "reason": "Standard informational query."
        }
    # ------------------------------------------------------------------
    # 3. Brain Demand Specification
    # ------------------------------------------------------------------

    def determine_brain_demand(self, intent_analysis: Dict[str, Any], cog_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Frames requirements for the Control Layer's Brain Registry.
        Does NOT execute model calls.
        """
        mode = intent_analysis.get("mode", CognitiveMode.QUERY)

        if mode == CognitiveMode.ACTION:
            return {
                "required_tier": "cloud_high_reasoning",
                "local_preferred": False,
                "min_context_window": 8192,
                "temperature": 0.1,  # Low temperature for deterministic actions
                "reasoning_depth": "strict"
            }
        elif mode == CognitiveMode.PLANNING:
            return {
                "required_tier": "cloud_deep_thinking",
                "local_preferred": False,
                "min_context_window": 16384,
                "temperature": 0.3,
                "reasoning_depth": "analytical"
            }
        else:
            # Query / Conversational
            return {
                "required_tier": "fast_local_or_cloud",
                "local_preferred": True,  # Prefer local fast brain for light queries
                "min_context_window": 4096,
                "temperature": 0.7,
                "reasoning_depth": "standard"
            }

    # ------------------------------------------------------------------
    # 4. Scratchpad & Candidate Evaluator
    # ------------------------------------------------------------------

    def evaluate_scratchpad(self, scratchpad_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Ranks and surfaces relevant candidate notes from Working Memory's scratchpad.
        Filters out redundant entries.
        """
        if not scratchpad_entries:
            return []

        ranked_notes = []
        seen_text = set()

        # Reverse iterate to prioritize recent notes
        for entry in reversed(scratchpad_entries):
            note_text = entry.get("note", "").strip()
            if note_text and note_text not in seen_text:
                seen_text.add(note_text)
                ranked_notes.append({
                    "timestamp": entry.get("timestamp"),
                    "note": note_text,
                    "priority": "high" if len(ranked_notes) < 3 else "normal"
                })

        return ranked_notes

    # ------------------------------------------------------------------
    # 5. Execution Payload Assembler
    # ------------------------------------------------------------------

    def assemble_payload(
        self,
        user_input: str,
        cog_state: Dict[str, Any],
        intent_analysis: Dict[str, Any],
        brain_demand: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Synthesizes active state, intent, and scratchpad notes into a structured
        execution payload ready for Brain consumption.
        """
        ranked_scratchpad = self.evaluate_scratchpad(cog_state.get("scratchpad", []))

        system_context = {
            "active_objective": cog_state.get("objective"),
            "active_task": cog_state.get("task"),
            "active_topic": cog_state.get("topic"),
            "relevant_facts": cog_state.get("facts"),
            "recent_actions": [
                {
                    key: event.get(key)
                    for key in (
                        "action_id", "action", "intended_action", "request_unit_text",
                        "resolved_path", "target_type", "status", "verification_status",
                        "failure_stage", "error", "dependencies",
                    )
                }
                for event in cog_state.get("action_events", [])[-8:]
            ],
            "evaluated_scratchpad": [item["note"] for item in ranked_scratchpad if item["priority"] == "high"]
        }

        return {
            "mode": intent_analysis.get("mode"),
            "brain_demand": brain_demand,
            "user_prompt": user_input,
            "system_context": system_context,
            "requires_simulation": intent_analysis.get("requires_simulation", False),
            "payload_created_at": datetime.now(timezone.utc).isoformat()
        }

    # ------------------------------------------------------------------
    # 6. Single-Turn Orchestrator (pipeline entry point)
    # ------------------------------------------------------------------

    def process_turn(self, user_input: str, memory_snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the full Cognitive Engine pipeline for a single user turn.

        This is the primary entry point called by ``AGPipeline``. It runs
        all five internal stages in sequence and returns a fully assembled
        Brain execution payload.

        Stages:
            1. ``ingest_context``      — normalize Working Memory snapshot
            2. ``classify_intent``     — determine cognitive mode
            3. ``determine_brain_demand`` — frame model requirements
            4. (implicit) scratchpad evaluated inside ``assemble_payload``
            5. ``assemble_payload``    — build the final dispatch payload

        :param user_input:      The cleaned user utterance for this turn.
        :param memory_snapshot: The dict returned by
                                ``WorkingMemory.build_context()`` or
                                ``WorkingMemory.get_snapshot()``.
        :returns: A payload dict suitable for ``BrainDispatcher.dispatch()``.
        """
        if not isinstance(memory_snapshot, dict):
            memory_snapshot = {}

        # Stage 1 — normalise context
        cog_state = self.ingest_context(memory_snapshot)

        # Stage 2 — classify intent / cognitive mode
        intent_analysis = self.classify_intent(user_input, cog_state)

        # Stage 3 — determine brain requirements
        brain_demand = self.determine_brain_demand(intent_analysis, cog_state)

        # Stage 4+5 — evaluate scratchpad and assemble final payload
        return self.assemble_payload(user_input, cog_state, intent_analysis, brain_demand)