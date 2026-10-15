"""
introspection/runtime_snapshot.py
====================================
Runtime Snapshot

An immutable dataclass that represents AG's complete runtime state at
the moment the snapshot was taken.

The RuntimeSnapshot is the single source of truth describing:

    Identity        — name, version, instance ID
    Loaded Modules  — modules successfully imported at snapshot time
    Connected       — subsystems currently online
    Disconnected    — subsystems offline or degraded
    Brains          — brain subsystems and their statuses
    Memory          — memory subsystems and their statuses
    Runtime Health  — aggregated health across all subsystems
    Capabilities    — union of all subsystem capabilities
    Limitations     — union of all subsystem limitations
    Configuration   — active configuration summary
    Diagnostics     — per-subsystem diagnostic data

Immutability
------------
The snapshot uses Python's ``__slots__`` and raises ``AttributeError`` on
any attempt to mutate a field after construction.

The snapshot is NOT connected to Cloud Brain, NOT connected to Jarwis.py,
and NOT exposed through chat.  It is a standalone data object.

Stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, FrozenSet, List, Optional
import uuid


# ---------------------------------------------------------------------------
# Aggregated health summary
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AggregatedHealth:
    """
    Top-level health summary computed from all registered subsystems.

    Attributes
    ----------
    overall : str
        "healthy" | "warning" | "critical" | "unknown"
    healthy_count : int
    warning_count : int
    critical_count : int
    unknown_count : int
    total_subsystems : int
    issues : tuple of str
        All issues collected from every subsystem.
    """
    overall: str
    healthy_count: int
    warning_count: int
    critical_count: int
    unknown_count: int
    total_subsystems: int
    issues: tuple = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall": self.overall,
            "healthy_count": self.healthy_count,
            "warning_count": self.warning_count,
            "critical_count": self.critical_count,
            "unknown_count": self.unknown_count,
            "total_subsystems": self.total_subsystems,
            "issues": list(self.issues),
        }


# ---------------------------------------------------------------------------
# Runtime Snapshot
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RuntimeSnapshot:
    """
    Immutable snapshot of AG's complete runtime state.

    Constructed by SnapshotBuilder; never mutated after creation.

    Fields
    ------
    snapshot_id : str
        UUID for this snapshot instance.
    taken_at : str
        ISO-8601 UTC timestamp when the snapshot was taken.
    identity_name : str
        AG's registered identity name (e.g. "AG" / "Ambient Guidance").
    identity_version : str
        AG's version string.
    loaded_modules : tuple of str
        Python module names that were successfully imported at snapshot time.
    connected_subsystems : tuple of str
        Names of subsystems with status ONLINE.
    disconnected_subsystems : tuple of str
        Names of subsystems with status OFFLINE or DEGRADED.
    brains : dict
        Mapping of brain name → status string.
    memory : dict
        Mapping of memory subsystem name → status string.
    runtime_health : AggregatedHealth
        Aggregated health across all registered subsystems.
    capabilities : tuple of str
        Union of all subsystem capabilities (deduplicated, sorted).
    limitations : tuple of str
        Union of all subsystem limitations (deduplicated, sorted).
    configuration : dict
        Active configuration summary (key→value).
    diagnostics : dict
        Per-subsystem diagnostic data (subsystem name → diagnostic dict).
    subsystem_count : int
        Total number of subsystems registered at snapshot time.
    """

    snapshot_id: str
    taken_at: str
    identity_name: str
    identity_version: str
    loaded_modules: tuple
    connected_subsystems: tuple
    disconnected_subsystems: tuple
    brains: Dict[str, str]
    memory: Dict[str, str]
    runtime_health: AggregatedHealth
    capabilities: tuple
    limitations: tuple
    configuration: Dict[str, Any]
    diagnostics: Dict[str, Any]
    subsystem_count: int

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "taken_at": self.taken_at,
            "identity": {
                "name": self.identity_name,
                "version": self.identity_version,
            },
            "loaded_modules": list(self.loaded_modules),
            "connected_subsystems": list(self.connected_subsystems),
            "disconnected_subsystems": list(self.disconnected_subsystems),
            "brains": dict(self.brains),
            "memory": dict(self.memory),
            "runtime_health": self.runtime_health.to_dict(),
            "capabilities": list(self.capabilities),
            "limitations": list(self.limitations),
            "configuration": dict(self.configuration),
            "diagnostics": dict(self.diagnostics),
            "subsystem_count": self.subsystem_count,
        }

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def is_healthy(self) -> bool:
        """True when overall health is 'healthy'."""
        return self.runtime_health.overall == "healthy"

    @property
    def has_disconnected(self) -> bool:
        """True when at least one subsystem is disconnected."""
        return len(self.disconnected_subsystems) > 0

    @property
    def online_count(self) -> int:
        return len(self.connected_subsystems)

    @property
    def offline_count(self) -> int:
        return len(self.disconnected_subsystems)


# ---------------------------------------------------------------------------
# Snapshot factory helper
# ---------------------------------------------------------------------------

def make_empty_snapshot(
    identity_name: str = "AG",
    identity_version: str = "0.0.0",
) -> RuntimeSnapshot:
    """
    Create a minimal empty snapshot.  Useful as a default or in tests.
    """
    return RuntimeSnapshot(
        snapshot_id=uuid.uuid4().hex,
        taken_at=datetime.now(timezone.utc).isoformat(),
        identity_name=identity_name,
        identity_version=identity_version,
        loaded_modules=(),
        connected_subsystems=(),
        disconnected_subsystems=(),
        brains={},
        memory={},
        runtime_health=AggregatedHealth(
            overall="unknown",
            healthy_count=0,
            warning_count=0,
            critical_count=0,
            unknown_count=0,
            total_subsystems=0,
            issues=(),
        ),
        capabilities=(),
        limitations=(),
        configuration={},
        diagnostics={},
        subsystem_count=0,
    )
