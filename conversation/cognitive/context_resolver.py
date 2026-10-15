"""
cognitive/context_resolver.py
==============================
AG Phase 2 — ContextResolver

The ContextResolver is the second pipeline stage.  It runs after the
CognitiveEvaluator has populated the Thought and before the Brain is called.

Its sole responsibility is to produce a complete, unambiguous
``thought.resolved_input`` — the prompt the Brain will actually receive.

The Brain should NEVER receive an ambiguous question.

Responsibilities
----------------
1. Inspect WorkingMemory for active entities, topics, and history.
2. Identify the most likely referent for any reference words detected by
   the CognitiveEvaluator.
3. Rewrite the input to replace references with explicit content.
4. Score the resolution using the ConfidenceEngine.
5. If confidence is high  → rewrite automatically.
   If confidence is low   → mark the Thought to ask the user.
6. Log every decision in the Thought's reasoning_log.

Design principles
-----------------
- NEVER generates responses. Output is always a rewritten prompt string.
- NEVER uses hardcoded pronoun lists for matching. It detects that a
  reference slot needs filling (from the Thought's ``missing_information``),
  then looks for a candidate in context.
- Works entirely from the Thought + ConversationContext / EntityRegistry
  passed in by WorkingMemory.  No direct imports of Jarwis.py.

Example
-------
    Input:  "Where did he work?"
    Context entities: [Albert Einstein (confidence=0.95, last mentioned 1 turn ago)]
    Output: "Where did Albert Einstein work?"
    Confidence: 0.88
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from conversation.core.thought import Thought

from conversation.cognitive.confidence_engine import ConfidenceEngine, ResolutionScore
from conversation.core.conversation_context import ConversationContext
from conversation.core.entity import EntityRegistry, ENTITY_TYPE_PERSON


class ContextResolver:
    """
    Resolves incomplete questions into complete, unambiguous prompts.

    Usage
    -----
    ::

        resolver = ContextResolver()
        thought = resolver.resolve(
            thought,
            context=working_memory.get_conversation_context(),
            entity_registry=working_memory.get_entity_registry(),
        )
        # thought.resolved_input is now safe to send to the Brain
    """

    def __init__(self, confidence_engine: Optional[ConfidenceEngine] = None) -> None:
        self._confidence = confidence_engine or ConfidenceEngine()

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def resolve(
        self,
        thought: "Thought",
        context: Optional[ConversationContext] = None,
        entity_registry: Optional[EntityRegistry] = None,
    ) -> "Thought":
        """
        Attempt to resolve contextual references in the Thought.

        After this call:
            - ``thought.resolved_input`` contains the (possibly rewritten) prompt.
            - ``thought.confidence`` reflects the resolution confidence.
            - ``thought.ask_for_clarification`` is True if we cannot resolve.
            - ``thought.candidate_contexts`` lists what was considered.
            - The reasoning_log captures every decision.

        :param thought: The Thought to resolve (mutated in-place).
        :param context: The active ConversationContext from WorkingMemory.
        :param entity_registry: The active EntityRegistry from WorkingMemory.
        :returns: The mutated Thought.
        """
        thought.log("resolver", f"Resolving input: {thought.raw_input!r}")

        # Fast path: if the evaluator marked this as self-contained, skip
        if not thought.requires_context:
            thought.log("resolver", "No context required — passing input through unchanged.")
            thought.resolve(thought.raw_input, 1.0)
            return thought

        # Fast path: if no context objects are available
        if context is None and entity_registry is None:
            thought.log("resolver", "No context available — cannot resolve references.")
            score = self._confidence.score_no_context_available()
            self._confidence.annotate_thought(thought, score)
            if score.should_ask_user:
                thought.request_clarification(
                    self._build_clarification_question(thought)
                )
            return thought

        thought.log("resolver", "Context available — attempting resolution.")

        # Collect resolution candidates
        candidates = self._collect_candidates(thought, context, entity_registry)

        if not candidates:
            thought.log("resolver", "No resolution candidates found.")
            score = self._confidence.score_no_context_available()
            self._confidence.annotate_thought(thought, score)
            if score.should_ask_user:
                thought.request_clarification(
                    self._build_clarification_question(thought)
                )
            return thought

        # Select the best candidate
        best = max(candidates, key=lambda c: c["confidence"])
        alternative_names = [c["label"] for c in candidates if c["label"] != best["label"]]

        # Score the resolution
        score = self._confidence.score_entity_resolution(
            referent_text=thought.raw_input,
            candidate_entity=best["label"],
            candidate_confidence=best["confidence"],
            turns_since_mention=best.get("turns_since_mention", 0),
            alternative_candidates=alternative_names if len(candidates) > 1 else None,
        )
        self._confidence.annotate_thought(thought, score)

        # Record candidates in the Thought
        for c in candidates:
            thought.add_candidate_context(
                label=c["label"],
                confidence=c["confidence"],
                source=c["source"],
            )

        if score.should_auto_resolve:
            rewritten = self._rewrite(thought.raw_input, best["label"])
            thought.resolve(rewritten, score.score)
            thought.log(
                "resolver",
                f"Auto-resolved: '{thought.raw_input}' → '{rewritten}' "
                f"(referent: {best['label']!r}, confidence {score.score:.2f})",
            )
        elif score.should_ask_user:
            thought.resolve(thought.raw_input, score.score)
            thought.request_clarification(
                self._build_clarification_question(thought, candidates)
            )
            thought.log(
                "resolver",
                f"Confidence too low ({score.score:.2f}) — requesting clarification.",
            )
        else:
            # Tentative resolution: rewrite but note it is uncertain
            rewritten = self._rewrite(thought.raw_input, best["label"])
            thought.resolve(rewritten, score.score)
            thought.log(
                "resolver",
                f"Tentative resolution ({score.score:.2f}): '{rewritten}' — proceeding with caution.",
            )

        thought.log("resolver", "Resolution complete.")
        return thought

    # ------------------------------------------------------------------
    # Candidate collection
    # ------------------------------------------------------------------

    def _collect_candidates(
        self,
        thought: "Thought",
        context: Optional[ConversationContext],
        entity_registry: Optional[EntityRegistry],
    ) -> List[Dict[str, Any]]:
        """
        Build a list of resolution candidates from all available context sources.

        Each candidate: {"label": str, "confidence": float, "source": str,
                         "turns_since_mention": int}
        """
        candidates: List[Dict[str, Any]] = []
        seen_labels: set = set()

        def _add(label: str, confidence: float, source: str, turns: int = 0) -> None:
            if label and label.lower() not in seen_labels:
                seen_labels.add(label.lower())
                candidates.append({
                    "label": label,
                    "confidence": confidence,
                    "source": source,
                    "turns_since_mention": turns,
                })

        # Source 1: EntityRegistry — most authoritative
        if entity_registry is not None:
            recent_person = entity_registry.most_recent_person()
            if recent_person:
                _add(recent_person.name, recent_person.confidence, "entity_registry.person")

            most_recent = entity_registry.most_recent()
            if most_recent and most_recent != recent_person:
                _add(most_recent.name, most_recent.confidence, "entity_registry.recent")

            # Also consider entities mentioned in the current Thought
            for ent in thought.detected_entities:
                name = ent.get("name", "")
                reg_ent = entity_registry.find(name) if name else None
                conf = reg_ent.confidence if reg_ent else 0.75
                _add(name, conf, "entity_registry.input_mention")

        # Source 2: ConversationContext active entities
        if context is not None:
            recent_entity = context.most_recent_entity()
            if recent_entity:
                # Try to get a richer confidence from the registry
                reg_conf = 0.80
                if entity_registry:
                    reg_ent = entity_registry.find(recent_entity)
                    if reg_ent:
                        reg_conf = reg_ent.confidence
                _add(recent_entity, reg_conf, "conversation_context.entity")

            # Source 3: Current topic as fallback
            topic = context.current_topic()
            if topic:
                _add(topic, 0.60, "conversation_context.topic")

            # Source 4: Recent frames — look for entities in recent history
            for frame in reversed(context.recent_frames(3)):
                turns_ago = 1  # approximate; we don't track exact distance here
                for ent in frame.get("entities", []):
                    name = ent.get("name", "") if isinstance(ent, dict) else str(ent)
                    if name:
                        _add(name, 0.70 - (turns_ago * 0.10), "history.entity", turns_ago)
                        turns_ago += 1

        return candidates

    # ------------------------------------------------------------------
    # Rewriting
    # ------------------------------------------------------------------

    def _rewrite(self, raw_input: str, referent: str) -> str:
        """
        Substitute the reference in ``raw_input`` with ``referent``.

        Strategy:
          1. If a reference word (he/she/it/they/this/that etc.) appears
             early in the sentence, replace the first occurrence.
          2. Otherwise append "(about <referent>)" as context.
          3. Never produce a worse question than the original.
        """
        text = raw_input.strip()

        # Pattern: replace a reference word that occupies the subject position
        # (beginning of question, or after a wh-word)
        _SUBJECT_REF_RE = re.compile(
            r"\b(he|she|it|they|him|her|them|this|that|these|those"
            r"|the person|the one|the thing|the place)\b",
            re.IGNORECASE,
        )

        match = _SUBJECT_REF_RE.search(text)
        if match:
            start, end = match.span()
            # Only substitute if the reference is in the first half of the sentence
            if start < len(text) // 2 + 5:
                rewritten = text[:start] + referent + text[end:]
                # Clean up double spaces
                rewritten = re.sub(r" +", " ", rewritten).strip()
                return rewritten

        # Fallback: append context
        if referent.lower() not in text.lower():
            return f"{text} (about {referent})"

        return text

    # ------------------------------------------------------------------
    # Clarification question builder
    # ------------------------------------------------------------------

    def _build_clarification_question(
        self,
        thought: "Thought",
        candidates: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Build a natural clarification question to ask the user.

        If there are multiple candidates, list them.
        Otherwise ask a general "who / what are you referring to?" question.
        """
        base = thought.raw_input.rstrip("?.!")

        if candidates and len(candidates) >= 2:
            names = [c["label"] for c in candidates[:3]]
            if len(names) == 2:
                listed = f"{names[0]} or {names[1]}"
            else:
                listed = f"{', '.join(names[:-1])}, or {names[-1]}"
            return f"Are you asking about {listed}? (referring to: '{base}')"

        # Check if there's a wh-word in the question to frame the ask
        text_lower = thought.raw_input.lower()
        if "where" in text_lower:
            return f"Where are you asking about? Could you clarify who or what '{base}' refers to?"
        if "when" in text_lower:
            return f"When are you asking about? Could you clarify the subject of '{base}'?"
        if "why" in text_lower:
            return f"Could you clarify what '{base}' refers to? I want to make sure I answer the right question."
        if "who" in text_lower or "he" in text_lower or "she" in text_lower:
            return f"Who are you asking about? ('{base}')"
        if "what" in text_lower or "it" in text_lower or "this" in text_lower or "that" in text_lower:
            return f"What are you asking about? ('{base}')"

        return f"Could you clarify what '{base}' refers to? I want to make sure I answer correctly."
