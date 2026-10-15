"""
self_info/ab_self_model.py
===========================
Jarwis Self-Model — Structured Factual Data

Provides verified, structured information about Jarwis, the assistant identity
implemented by the AG project.
covering identity, architecture, capabilities, limitations, and operational
state.

Design rules:
- No long natural-language strings stored here.
- Structured dicts and lists only.
- If a fact cannot be verified from the project, it is marked UNKNOWN.
- No LLM calls. No I/O at module import.
- The project_context.json file is read lazily when get_identity() is
  called for the first time so import never blocks.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_PROJECT_CONTEXT_PATH = os.path.join(_ROOT, "configuration", "project_context.json")
_BRAIN_REGISTRY_PATH  = os.path.join(_ROOT, "configuration", "brain_registry.json")


def _load_json(path: str) -> Dict[str, Any]:
    """Load a JSON file, return empty dict on any error."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# 1. Identity
# ---------------------------------------------------------------------------

def get_identity() -> Dict[str, Any]:
    """
    Return Jarwis's verified identity facts.

    Sources:
    - configuration/project_context.json  (project metadata)
    - Hardcoded structural facts that don't change run-to-run.
    """
    ctx = _load_json(_PROJECT_CONTEXT_PATH)
    from runtime.persona_controller import get_persona_controller

    identity = {
        "name":          "Jarwis",
        "short_name":    "J",
        "full_name":     "Jarwis AI Assistant",
        "version":       ctx.get("current_version", "UNKNOWN"),
        "phase":         ctx.get("current_phase", "UNKNOWN"),
        "purpose":       "A personal AI assistant that provides intelligent responses, memory management, task tracking, and project awareness through a terminal interface.",
        "project_type":  "Python terminal-based AI assistant",
        "active_persona": get_persona_controller().mode.title(),
    }
    creator = ctx.get("creator") or ctx.get("creator_name")
    if isinstance(creator, str) and creator.strip():
        identity["creator"] = creator.strip()
    return identity


# ---------------------------------------------------------------------------
# 2. Architecture
# ---------------------------------------------------------------------------

# Static architecture description — these are verified from the codebase.
ARCHITECTURE: Dict[str, Any] = {
    "major_modules": {
        "Jarwis.py": "Main entry point. Owns the REPL, intent routing, and pipeline orchestration.",
        "cognition/": "ExecutiveLayer (storage/routing decisions), CognitiveEngine (turn processing), executive.py (plan building).",
        "conversation/": "ConversationManager, EntityTracker, ReferenceResolver, ResponseProcessor.",
        "memory/": "WorkingMemory (ephemeral session state), clean_memory utility.",
        "brains/": "brain.py (public API), BrainDispatcher, local_brain, TokenBudget, cloud_brain/ (AdaptiveCloudBrain + provider registry).",
        "pipeline/": "AGPipeline — orchestrates full cognitive pipeline for unknown-intent inputs.",
        "storage/": "StorageManager, IndexManager, SchemaProcessor — persistent long-term storage.",
        "runtime/": "RuntimeAdapter, IntrospectionEngine, SnapshotBuilder, SubsystemRegistry — runtime introspection.",
        "diagnostics/": "AnalyticsLogger, ProbeSimulator — event logging and health probing.",
        "configuration/": "JSON configuration files (project_context, brain_registry, memory, tasks).",
        "maintenance/": "Repository integrity, import integrity, pytest validation, dependency checks.",
        "self_info/": "Self-Info system — structured self-model, query routing, task assessment.",
    },
    "runtime_flow": [
        "User types input → Jarwis.py main() loop",
        "detect_intent() classifies the intent",
        "Self-info queries → SelfInfoEngine (self_info/)",
        "Introspection queries → IntrospectionEngine + _handle_introspection()",
        "Known intents (memory, tasks, project) → direct handlers",
        "Unknown intent → _pipeline_response() → ConversationManager → ReferenceResolver → ExecutiveLayer → BrainDispatcher → ResponseProcessor",
        "Cloud-direct prefix → ask_cloud_direct()",
        "All brain calls fall back to ask_brain() which respects active brain mode",
    ],
    "available_brains": "See get_operational_info()['brains']",
    "subsystems": [
        "ConversationManager",
        "WorkingMemory",
        "EntityTracker",
        "ReferenceResolver",
        "ExecutiveLayer",
        "BrainDispatcher",
        "ResponseProcessor",
        "AdaptiveCloudBrain",
        "IntrospectionEngine",
        "SelfInfoEngine",
        "StorageManager",
        "AnalyticsLogger",
    ],
}


def get_architecture() -> Dict[str, Any]:
    """Return the static architecture description."""
    return ARCHITECTURE


# ---------------------------------------------------------------------------
# 3. Capabilities
# ---------------------------------------------------------------------------

# Each entry is (capability_name, description, implemented, requires_external).
# implemented=True means the code exists and runs.
# requires_external=True means it needs something outside the process (network, Ollama, etc.).

CAPABILITIES: List[Dict[str, Any]] = [
    {
        "name": "Natural-language conversation",
        "description": "Multi-turn conversation with topic continuity and reference resolution.",
        "implemented": True,
        "requires_external": True,   # needs a brain (local or cloud)
    },
    {
        "name": "Persistent memory",
        "description": "Store and recall named facts across sessions via memory.json.",
        "implemented": True,
        "requires_external": False,
    },
    {
        "name": "Task management",
        "description": "Add, list, complete, and track active/completed tasks in tasks.json.",
        "implemented": True,
        "requires_external": False,
    },
    {
        "name": "Project context awareness",
        "description": "Reports project name, version, phase, milestones, and next step from project_context.json.",
        "implemented": True,
        "requires_external": False,
    },
    {
        "name": "Local brain (Ollama)",
        "description": "Offline LLM inference via Ollama (qwen2.5:3b by default).",
        "implemented": True,
        "requires_external": True,   # Ollama must be running
    },
    {
        "name": "Cloud brain (OpenRouter)",
        "description": "Cloud LLM inference via OpenRouter API.",
        "implemented": True,
        "requires_external": True,   # network + API key
    },
    {
        "name": "Adaptive brain selection",
        "description": "Auto mode selects local or cloud brain based on request characteristics.",
        "implemented": True,
        "requires_external": False,
    },
    {
        "name": "Runtime introspection",
        "description": "Build live snapshots of all registered subsystems (health, status, capabilities).",
        "implemented": True,
        "requires_external": False,
    },
    {
        "name": "Self-info system",
        "description": "Answer self-referential questions from verified structured data, not LLM invention.",
        "implemented": True,
        "requires_external": False,
    },
    {
        "name": "Entity tracking",
        "description": "Track named entities across conversation turns for reference resolution.",
        "implemented": True,
        "requires_external": False,
    },
    {
        "name": "Reference resolution",
        "description": "Resolve pronouns and implicit references to previously mentioned entities.",
        "implemented": True,
        "requires_external": False,
    },
    {
        "name": "Token budget estimation",
        "description": "Estimate token usage before dispatching to a brain.",
        "implemented": True,
        "requires_external": False,
    },
    {
        "name": "Response safety processing",
        "description": "Sanitize and safety-check brain responses before returning to user.",
        "implemented": True,
        "requires_external": False,
    },
    {
        "name": "Workspace and filesystem control",
        "description": (
            "Set, switch, navigate, and clear an active workspace; inspect, list, "
            "read, create, write, append, rename, copy, move, and search files; "
            "copy directories within workspace boundaries."
        ),
        "implemented": True,
        "requires_external": False,
    },
    {
        "name": "Directory content reading",
        "description": (
            "Index readable text files in a selected directory and answer questions "
            "using relevant file contents."
        ),
        "implemented": True,
        "requires_external": False,
    },
    {
        "name": "Analytics event logging",
        "description": "Log events to data/warehouse/events.jsonl for later analysis.",
        "implemented": True,
        "requires_external": False,
    },
    {
        "name": "Memory file viewer",
        "description": "Open File Explorer to the memory.json file.",
        "implemented": True,
        "requires_external": True,   # Windows File Explorer
    },
]


def get_capabilities() -> List[Dict[str, Any]]:
    """Return all capability records."""
    return CAPABILITIES


def get_implemented_capabilities() -> List[str]:
    """Return names of capabilities that are currently implemented."""
    return [c["name"] for c in CAPABILITIES if c["implemented"]]


def get_external_capabilities() -> List[str]:
    """Return names of capabilities that require external systems."""
    return [c["name"] for c in CAPABILITIES if c["requires_external"]]


# ---------------------------------------------------------------------------
# 4. Limitations
# ---------------------------------------------------------------------------

LIMITATIONS: List[Dict[str, Any]] = [
    {
        "name": "Voice recognition",
        "description": "No voice input or output. Terminal text only.",
        "category": "missing_capability",
        "blocked_by": "development_rule",
    },
    {
        "name": "Camera awareness",
        "description": "No image or video processing.",
        "category": "missing_capability",
        "blocked_by": "development_rule",
    },
    {
        "name": "Desktop automation",
        "description": "No ability to click, type into, or control desktop applications.",
        "category": "missing_capability",
        "blocked_by": "development_rule",
    },
    {
        "name": "Background mode",
        "description": "Runs only while the terminal is active. No background daemon.",
        "category": "missing_capability",
        "blocked_by": "development_rule",
    },
    {
        "name": "Snap detection",
        "description": "No ability to detect finger snaps or physical gestures.",
        "category": "missing_capability",
        "blocked_by": "development_rule",
    },
    {
        "name": "Persistent storage integration",
        "description": "StorageManager/IndexManager exist but are not yet fully wired into the live conversation pipeline.",
        "category": "incomplete_system",
        "blocked_by": "integration_pending",
    },
    {
        "name": "Working memory method gaps",
        "description": "Several WorkingMemory methods referenced in Jarwis.py are noted as migration placeholders (discussion_topic, get_discussion_entries, has_unsaved_discussion, resolve_thread_reference, update_working_memory).",
        "category": "incomplete_system",
        "blocked_by": "migration_pending",
    },
    {
        "name": "BrainDispatcher uses mock adapters",
        "description": "Default BrainDispatcher adapters are mocks. Real dispatch falls back to direct ask_brain() calls.",
        "category": "runtime_limitation",
        "blocked_by": "adapter_integration_pending",
    },
    {
        "name": "Ephemeral session state",
        "description": "WorkingMemory is cleared on process restart. No cross-session memory continuity beyond memory.json.",
        "category": "runtime_limitation",
        "blocked_by": "by_design",
    },
    {
        "name": "Cloud brain requires network and API key",
        "description": "OpenRouter integration requires OPENROUTER_API_KEY env var and internet access.",
        "category": "external_dependency",
        "blocked_by": "environment",
    },
    {
        "name": "Local brain requires Ollama",
        "description": "Local inference requires Ollama running locally with the configured model (qwen2.5:3b).",
        "category": "external_dependency",
        "blocked_by": "environment",
    },
    {
        "name": "Windows-only file explorer integration",
        "description": "Memory file viewer uses Windows explorer command.",
        "category": "runtime_limitation",
        "blocked_by": "platform",
    },
]


def get_limitations() -> List[Dict[str, Any]]:
    """Return all limitation records."""
    return LIMITATIONS


def get_limitations_by_category(category: str) -> List[Dict[str, Any]]:
    """Return limitations filtered by category."""
    return [lim for lim in LIMITATIONS if lim.get("category") == category]


# ---------------------------------------------------------------------------
# 5. Operational Information
# ---------------------------------------------------------------------------

def get_operational_info() -> Dict[str, Any]:
    """
    Return current operational state.

    Reads brain_registry.json for available brains.
    project_context.json for current status.
    Falls back to UNKNOWN if files are missing.
    """
    ctx      = _load_json(_PROJECT_CONTEXT_PATH)
    registry = _load_json(_BRAIN_REGISTRY_PATH)

    brains = []
    for brain in registry.get("brains", []):
        brains.append({
            "id":               brain.get("id", "UNKNOWN"),
            "provider":         brain.get("provider", "UNKNOWN"),
            "model":            brain.get("model", "UNKNOWN"),
            "requires_network": brain.get("requires_network", True),
            "enabled":          brain.get("enabled", False),
            "capabilities":     brain.get("capabilities", []),
        })

    from runtime.persona_controller import get_persona_controller

    return {
        "brains":           brains,
        "current_version":  ctx.get("current_version", "UNKNOWN"),
        "current_phase":    ctx.get("current_phase", "UNKNOWN"),
        "current_objective": ctx.get("current_objective", "UNKNOWN"),
        "next_milestone":   ctx.get("next_milestone", "UNKNOWN"),
        "blocked_features": ctx.get("blocked_features", []),
        "completed_milestones": ctx.get("completed_milestones", []),
        "active_persona": get_persona_controller().mode.title(),
        "brain_mode": _load_json(
            os.path.join(_ROOT, "configuration", "brain_config.json")
        ).get("brain_mode", "UNKNOWN"),
    }
