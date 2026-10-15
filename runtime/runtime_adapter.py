"""
runtime_adapter.py
==================
Runtime Integration Adapters — Project AB

Thin IntrospectableSubsystem shims that wrap every existing AG subsystem
so they can be registered with the IntrospectionEngine and appear in a
RuntimeSnapshot.

Design rules:
- Each adapter wraps one existing subsystem instance.
- No subsystem code is modified.
- No new logic is introduced — adapters only surface what already exists.
- All five interface methods (status, health, capabilities, limitations,
  diagnostics) are safe to call at any time and never raise.
- Stdlib only inside adapter methods.

Usage (from Jarwis.py main()):
::

    from runtime_adapter import build_registry

    engine = IntrospectionEngine(registry=build_registry(...))
    snapshot = engine.build_snapshot()
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

from runtime.introspection.subsystem_interface import (
    IntrospectableSubsystem,
    SubsystemStatus,
    HealthStatus,
    SubsystemHealth,
)
from runtime.introspection.subsystem_registry import SubsystemRegistry


# ---------------------------------------------------------------------------
# Helper — safe string list
# ---------------------------------------------------------------------------

def _safe_list(value: Any) -> List[str]:
    """Return value as a list of strings, or [] on any error."""
    try:
        if isinstance(value, (list, tuple)):
            return [str(v) for v in value]
        return []
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Adapter: ConversationManager
# ---------------------------------------------------------------------------

class ConversationManagerAdapter(IntrospectableSubsystem):
    """Wraps the ConversationManager instance."""

    _subsystem_type = "context"

    def __init__(self, manager: Any) -> None:
        self._manager = manager

    def subsystem_name(self) -> str:
        return "conversation_manager"

    def status(self) -> SubsystemStatus:
        try:
            state = self._manager.get_state()
            if state.get("conversation_id"):
                return SubsystemStatus.ONLINE
            return SubsystemStatus.DEGRADED
        except Exception:
            return SubsystemStatus.UNKNOWN

    def health(self) -> SubsystemHealth:
        try:
            state = self._manager.get_state()
            issues = []
            if not state.get("conversation_id"):
                issues.append("No active conversation ID.")
            return SubsystemHealth(
                status=HealthStatus.HEALTHY if not issues else HealthStatus.WARNING,
                issues=issues,
                checks_passed=1 if not issues else 0,
                checks_total=1,
                notes=["ConversationManager: in-memory, no persistence required."],
            )
        except Exception as exc:
            return SubsystemHealth(
                status=HealthStatus.CRITICAL,
                issues=[f"ConversationManager health check failed: {exc}"],
                checks_passed=0,
                checks_total=1,
            )

    def capabilities(self) -> List[str]:
        return [
            "Topic tracking and topic shift detection",
            "Contextual reference resolution",
            "Pending action and confirmation state management",
            "Thread summary generation",
            "Context building for downstream subsystems",
            "Brain-switch action pattern detection",
        ]

    def limitations(self) -> List[str]:
        return [
            "In-memory only — state lost on process restart",
            "Topic inference is heuristic, not LLM-based",
        ]

    def diagnostics(self) -> Dict[str, Any]:
        try:
            state = self._manager.get_state()
            return {
                "conversation_id": state.get("conversation_id"),
                "current_topic": state.get("current_topic"),
                "turn_count": state.get("turn_count", 0),
                "recent_entity_count": len(state.get("recent_entities", [])),
                "conversation_state": state.get("state"),
                "has_pending_action": state.get("pending_action") is not None,
                "has_pending_confirmation": state.get("pending_confirmation") is not None,
            }
        except Exception as exc:
            return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Adapter: WorkingMemory
# ---------------------------------------------------------------------------

class WorkingMemoryAdapter(IntrospectableSubsystem):
    """Wraps the WorkingMemory instance."""

    _subsystem_type = "memory"

    def __init__(self, memory: Any) -> None:
        self._memory = memory

    def subsystem_name(self) -> str:
        return "working_memory"

    def status(self) -> SubsystemStatus:
        try:
            if self._memory.is_expired():
                return SubsystemStatus.OFFLINE
            return SubsystemStatus.ONLINE
        except Exception:
            return SubsystemStatus.UNKNOWN

    def health(self) -> SubsystemHealth:
        try:
            expired = self._memory.is_expired()
            issues = ["Session has expired." ] if expired else []
            return SubsystemHealth(
                status=HealthStatus.WARNING if expired else HealthStatus.HEALTHY,
                issues=issues,
                checks_passed=0 if expired else 1,
                checks_total=1,
                notes=["WorkingMemory: ephemeral session RAM — no disk writes."],
            )
        except Exception as exc:
            return SubsystemHealth(
                status=HealthStatus.CRITICAL,
                issues=[f"WorkingMemory health check failed: {exc}"],
                checks_passed=0,
                checks_total=1,
            )

    def capabilities(self) -> List[str]:
        return [
            "Session-scoped ephemeral fact storage",
            "Active objective and task tracking",
            "Current topic tracking",
            "Conversation reference storage",
            "Discussion buffer for optional long-term persistence",
            "Decision cache for brain mode and routing decisions",
            "Reasoning scratchpad notes",
            "Thread reference resolution",
        ]

    def limitations(self) -> List[str]:
        return [
            "Ephemeral — cleared on process restart",
            "Does not write to persistent storage (memory.json)",
            "Session expires after configured TTL (default 1 hour)",
        ]

    def diagnostics(self) -> Dict[str, Any]:
        try:
            doc = self._memory.to_document()
            return {
                "session_id": doc.get("session_id"),
                "current_topic": doc.get("current_topic"),
                "current_objective": doc.get("current_objective"),
                "fact_count": len(doc.get("temporary_facts", {})),
                "cached_memory_count": len(doc.get("retrieved_memories", [])),
                "discussion_entry_count": len(doc.get("discussion_buffer", [])),
                "has_pending_action": doc.get("pending_action") is not None,
                "has_pending_confirmation": doc.get("pending_confirmation") is not None,
                "expired": self._memory.is_expired(),
            }
        except Exception as exc:
            return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Adapter: EntityTracker
# ---------------------------------------------------------------------------

class EntityTrackerAdapter(IntrospectableSubsystem):
    """Wraps the EntityTracker instance."""

    _subsystem_type = "context"

    def __init__(self, tracker: Any) -> None:
        self._tracker = tracker

    def subsystem_name(self) -> str:
        return "entity_tracker"

    def status(self) -> SubsystemStatus:
        try:
            # Always online as long as it's accessible
            _ = self._tracker.current_turn()
            return SubsystemStatus.ONLINE
        except Exception:
            return SubsystemStatus.UNKNOWN

    def health(self) -> SubsystemHealth:
        try:
            return SubsystemHealth(
                status=HealthStatus.HEALTHY,
                issues=[],
                checks_passed=1,
                checks_total=1,
                notes=[
                    f"Tracking {self._tracker.count()} entities, "
                    f"{self._tracker.count_active()} active, "
                    f"turn {self._tracker.current_turn()}."
                ],
            )
        except Exception as exc:
            return SubsystemHealth(
                status=HealthStatus.CRITICAL,
                issues=[f"EntityTracker health check failed: {exc}"],
                checks_passed=0,
                checks_total=1,
            )

    def capabilities(self) -> List[str]:
        return [
            "Conversational entity registration and tracking",
            "Turn-aware entity recency and frequency scoring",
            "Entity type classification (person, organization, place, object, event, concept)",
            "Conversation focus detection (most recently mentioned entity)",
            "Candidate ranking for reference resolution",
            "Active entity windowing (configurable recency window)",
        ]

    def limitations(self) -> List[str]:
        return [
            "Entity extraction is caller-provided — tracker does not parse text autonomously",
            "In-memory only — entities lost on session reset",
            "No named-entity recognition (NER) built in",
        ]

    def diagnostics(self) -> Dict[str, Any]:
        try:
            focus = self._tracker.get_focus()
            return {
                "current_turn": self._tracker.current_turn(),
                "total_entities": self._tracker.count(),
                "active_entities": self._tracker.count_active(),
                "conversation_focus": focus.name if focus else None,
                "recency_window": self._tracker._recency_window,
            }
        except Exception as exc:
            return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Adapter: ReferenceResolver
# ---------------------------------------------------------------------------

class ReferenceResolverAdapter(IntrospectableSubsystem):
    """Wraps the ReferenceResolver instance."""

    _subsystem_type = "cognitive"

    def __init__(self, resolver: Any) -> None:
        self._resolver = resolver

    def subsystem_name(self) -> str:
        return "reference_resolver"

    def status(self) -> SubsystemStatus:
        try:
            # Resolver is stateless — if it exists it's online
            _ = self._resolver._auto_threshold
            return SubsystemStatus.ONLINE
        except Exception:
            return SubsystemStatus.UNKNOWN

    def health(self) -> SubsystemHealth:
        try:
            return SubsystemHealth(
                status=HealthStatus.HEALTHY,
                issues=[],
                checks_passed=1,
                checks_total=1,
                notes=["ReferenceResolver: stateless, deterministic."],
            )
        except Exception as exc:
            return SubsystemHealth(
                status=HealthStatus.CRITICAL,
                issues=[f"ReferenceResolver health check failed: {exc}"],
                checks_passed=0,
                checks_total=1,
            )

    def capabilities(self) -> List[str]:
        return [
            "Pronoun resolution (he, she, they, it, this, that, etc.)",
            "Natural-language reference resolution (the company, the person, the city, etc.)",
            "Multi-candidate disambiguation via EntityTracker ranking",
            "Clarification question generation when confidence is low",
            "Confidence-scored resolution results",
        ]

    def limitations(self) -> List[str]:
        return [
            "Requires a populated EntityTracker to resolve references",
            "Resolution is heuristic — may misfire on ambiguous input",
            "Does not resolve references across session boundaries",
        ]

    def diagnostics(self) -> Dict[str, Any]:
        try:
            return {
                "auto_resolve_threshold": self._resolver._auto_threshold,
                "clarification_threshold": self._resolver._clarification_threshold,
            }
        except Exception as exc:
            return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Adapter: ExecutiveLayer
# ---------------------------------------------------------------------------

class ExecutiveLayerAdapter(IntrospectableSubsystem):
    """Wraps the ExecutiveLayer instance."""

    _subsystem_type = "decision"

    def __init__(self, layer: Any) -> None:
        self._layer = layer

    def subsystem_name(self) -> str:
        return "executive_layer"

    def status(self) -> SubsystemStatus:
        try:
            _ = self._layer.STRUCTURED_CATEGORIES
            return SubsystemStatus.ONLINE
        except Exception:
            return SubsystemStatus.UNKNOWN

    def health(self) -> SubsystemHealth:
        try:
            return SubsystemHealth(
                status=HealthStatus.HEALTHY,
                issues=[],
                checks_passed=1,
                checks_total=1,
                notes=["ExecutiveLayer: deterministic routing, no external dependencies."],
            )
        except Exception as exc:
            return SubsystemHealth(
                status=HealthStatus.CRITICAL,
                issues=[f"ExecutiveLayer health check failed: {exc}"],
                checks_passed=0,
                checks_total=1,
            )

    def capabilities(self) -> List[str]:
        return [
            "Storage routing decisions (long-term vs session vs temporary)",
            "Memory category routing (structured vs dynamic categories)",
            "Task and objective persistence trigger evaluation",
            "Execution cycle orchestration (input → cognitive → decision)",
        ]

    def limitations(self) -> List[str]:
        return [
            "Does not execute storage writes — issues decisions only",
            "Category detection is keyword-based, not semantic",
        ]

    def diagnostics(self) -> Dict[str, Any]:
        try:
            return {
                "structured_categories": sorted(self._layer.STRUCTURED_CATEGORIES),
                "config": dict(self._layer.config),
            }
        except Exception as exc:
            return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Adapter: BrainDispatcher
# ---------------------------------------------------------------------------

class BrainDispatcherAdapter(IntrospectableSubsystem):
    """Wraps the BrainDispatcher instance."""

    _subsystem_type = "brain"

    def __init__(self, dispatcher: Any) -> None:
        self._dispatcher = dispatcher

    def subsystem_name(self) -> str:
        return "brain_dispatcher"

    def status(self) -> SubsystemStatus:
        try:
            _ = self._dispatcher._adapters
            return SubsystemStatus.ONLINE
        except Exception:
            return SubsystemStatus.UNKNOWN

    def health(self) -> SubsystemHealth:
        try:
            adapters = self._dispatcher._adapters
            adapter_count = len(adapters)
            issues = []
            if adapter_count == 0:
                issues.append("No brain adapters registered.")
            return SubsystemHealth(
                status=HealthStatus.HEALTHY if not issues else HealthStatus.WARNING,
                issues=issues,
                checks_passed=1 if not issues else 0,
                checks_total=1,
                notes=[f"{adapter_count} brain adapter(s) registered."],
            )
        except Exception as exc:
            return SubsystemHealth(
                status=HealthStatus.CRITICAL,
                issues=[f"BrainDispatcher health check failed: {exc}"],
                checks_passed=0,
                checks_total=1,
            )

    def capabilities(self) -> List[str]:
        return [
            "Brain adapter registration and dynamic dispatch",
            "Control Layer authorization integration",
            "Standardized BrainResponse contract across all providers",
            "Mock brain adapter for offline and test mode",
            "Local and cloud brain routing",
        ]

    def limitations(self) -> List[str]:
        return [
            "Default adapters are mock implementations — requires real adapters for production dispatch",
            "Does not construct prompts — receives pre-built payloads",
        ]

    def diagnostics(self) -> Dict[str, Any]:
        try:
            adapters = self._dispatcher._adapters
            return {
                "registered_adapters": list(adapters.keys()),
                "has_control_layer": self._dispatcher.control_layer is not None,
            }
        except Exception as exc:
            return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Adapter: ResponseProcessor
# ---------------------------------------------------------------------------

class ResponseProcessorAdapter(IntrospectableSubsystem):
    """Wraps the ResponseProcessor instance."""

    _subsystem_type = "output"

    def __init__(self, processor: Any) -> None:
        self._processor = processor

    def subsystem_name(self) -> str:
        return "response_processor"

    def status(self) -> SubsystemStatus:
        try:
            _ = self._processor._critical_action_patterns
            return SubsystemStatus.ONLINE
        except Exception:
            return SubsystemStatus.UNKNOWN

    def health(self) -> SubsystemHealth:
        try:
            return SubsystemHealth(
                status=HealthStatus.HEALTHY,
                issues=[],
                checks_passed=1,
                checks_total=1,
                notes=["ResponseProcessor: deterministic, no external dependencies."],
            )
        except Exception as exc:
            return SubsystemHealth(
                status=HealthStatus.CRITICAL,
                issues=[f"ResponseProcessor health check failed: {exc}"],
                checks_passed=0,
                checks_total=1,
            )

    def capabilities(self) -> List[str]:
        return [
            "Response text sanitization (removes prompt artifact leakage)",
            "Output safety evaluation",
            "Pending action and confirmation request extraction",
            "Standardized ProcessedResponse output contract",
            "Fallback response on empty or unsafe brain output",
        ]

    def limitations(self) -> List[str]:
        return [
            "Safety checks are pattern-based, not semantic",
            "Does not invoke LLM calls for response evaluation",
        ]

    def diagnostics(self) -> Dict[str, Any]:
        try:
            return {
                "critical_action_pattern_count": len(self._processor._critical_action_patterns),
                "config": dict(self._processor.config),
            }
        except Exception as exc:
            return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Adapter: AdaptiveCloudBrain
# ---------------------------------------------------------------------------

class AdaptiveCloudBrainAdapter(IntrospectableSubsystem):
    """Wraps the AdaptiveCloudBrain instance."""

    _subsystem_type = "brain"

    def __init__(self, brain: Any) -> None:
        self._brain = brain

    def subsystem_name(self) -> str:
        return "adaptive_cloud_brain"

    def status(self) -> SubsystemStatus:
        try:
            _ = self._brain.provider_registry
            _ = self._brain.token_estimator
            return SubsystemStatus.ONLINE
        except Exception:
            return SubsystemStatus.UNKNOWN

    def health(self) -> SubsystemHealth:
        try:
            issues = []
            # Check all five components are present
            components = [
                "provider_registry",
                "token_estimator",
                "request_validator",
                "response_validator",
                "retry_manager",
            ]
            for comp in components:
                if not hasattr(self._brain, comp):
                    issues.append(f"Missing component: {comp}")

            checks_passed = len(components) - len(issues)
            return SubsystemHealth(
                status=HealthStatus.HEALTHY if not issues else HealthStatus.CRITICAL,
                issues=issues,
                checks_passed=checks_passed,
                checks_total=len(components),
                notes=["AdaptiveCloudBrain: full pipeline — prepare, validate, log, retry."],
            )
        except Exception as exc:
            return SubsystemHealth(
                status=HealthStatus.CRITICAL,
                issues=[f"AdaptiveCloudBrain health check failed: {exc}"],
                checks_passed=0,
                checks_total=5,
            )

    def capabilities(self) -> List[str]:
        return [
            "Token budget estimation before every cloud request",
            "Provider capability validation",
            "Pre-dispatch request validation",
            "Post-dispatch response validation",
            "Automatic retry with reduced budget on provider rejection",
            "Structured request/response logging",
            "BrainResult status reporting (PASSED, RETRYING, GAVE_UP, etc.)",
        ]

    def limitations(self) -> List[str]:
        return [
            "Does not make HTTP calls — orchestrates around the caller's dispatch",
            "Requires network access for cloud provider validation",
            "Retry budget reduction may reduce response quality on fallback",
        ]

    def diagnostics(self) -> Dict[str, Any]:
        try:
            stats = self._brain.logger_stats()
            registry = self._brain.provider_registry
            known_providers = []
            for pid in ("openrouter", "openai", "anthropic", "google"):
                profile = registry.get(pid)
                if profile:
                    known_providers.append(pid)
            return {
                "known_providers": known_providers,
                "logger_stats": stats,
            }
        except Exception as exc:
            return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Adapter: IntrospectionEngine itself
# ---------------------------------------------------------------------------

class IntrospectionEngineAdapter(IntrospectableSubsystem):
    """Wraps the IntrospectionEngine so it appears in its own snapshot."""

    _subsystem_type = "introspection"

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    def subsystem_name(self) -> str:
        return "introspection_engine"

    def status(self) -> SubsystemStatus:
        try:
            _ = self._engine.registry
            return SubsystemStatus.ONLINE
        except Exception:
            return SubsystemStatus.UNKNOWN

    def health(self) -> SubsystemHealth:
        try:
            count = self._engine.subsystem_count()
            return SubsystemHealth(
                status=HealthStatus.HEALTHY,
                issues=[],
                checks_passed=1,
                checks_total=1,
                notes=[f"IntrospectionEngine: {count} subsystem(s) registered."],
            )
        except Exception as exc:
            return SubsystemHealth(
                status=HealthStatus.CRITICAL,
                issues=[f"IntrospectionEngine health check failed: {exc}"],
                checks_passed=0,
                checks_total=1,
            )

    def capabilities(self) -> List[str]:
        return [
            "Runtime subsystem discovery via SubsystemRegistry",
            "Immutable RuntimeSnapshot generation",
            "Aggregated health status across all subsystems",
            "Capability and limitation union across all subsystems",
            "Per-subsystem diagnostic data collection",
            "Loaded module discovery via sys.modules",
        ]

    def limitations(self) -> List[str]:
        return [
            "Only subsystems registered with the registry are visible",
            "Snapshot reflects state at call time — not a live view",
        ]

    def diagnostics(self) -> Dict[str, Any]:
        try:
            summary = self._engine.snapshot_summary()
            return {
                "registered_subsystems": self._engine.registered_subsystem_names(),
                "last_snapshot_id": summary.get("snapshot_id"),
                "last_snapshot_taken_at": summary.get("taken_at"),
                "overall_health": summary.get("overall_health"),
            }
        except Exception as exc:
            return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Registry builder
# ---------------------------------------------------------------------------

def build_registry(
    conversation_manager: Any,
    working_memory: Any,
    entity_tracker: Any,
    reference_resolver: Any,
    executive_layer: Any,
    brain_dispatcher: Any,
    response_processor: Any,
    adaptive_cloud_brain: Any,
    introspection_engine: Optional[Any] = None,
) -> SubsystemRegistry:
    """
    Build and return a SubsystemRegistry populated with adapter shims
    for every AG subsystem.

    Pass ``introspection_engine`` only after the engine is constructed
    (it registers itself so it appears in its own snapshot).

    :returns: A populated SubsystemRegistry ready for IntrospectionEngine.
    """
    registry = SubsystemRegistry()

    registry.register("conversation_manager",  ConversationManagerAdapter(conversation_manager))
    registry.register("working_memory",         WorkingMemoryAdapter(working_memory))
    registry.register("entity_tracker",         EntityTrackerAdapter(entity_tracker))
    registry.register("reference_resolver",     ReferenceResolverAdapter(reference_resolver))
    registry.register("executive_layer",        ExecutiveLayerAdapter(executive_layer))
    registry.register("brain_dispatcher",       BrainDispatcherAdapter(brain_dispatcher))
    registry.register("response_processor",     ResponseProcessorAdapter(response_processor))
    registry.register("adaptive_cloud_brain",   AdaptiveCloudBrainAdapter(adaptive_cloud_brain))

    if introspection_engine is not None:
        registry.register(
            "introspection_engine",
            IntrospectionEngineAdapter(introspection_engine),
        )

    return registry
