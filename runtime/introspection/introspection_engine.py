"""
introspection/introspection_engine.py
=======================================
Introspection Engine

The top-level class for Part B.  It owns a SubsystemRegistry and a
SnapshotBuilder, exposes ``build_snapshot()`` as its primary public
method, and optionally caches the last snapshot for repeat access.

Responsibilities
----------------
- Discover every registered subsystem (via SubsystemRegistry).
- Collect runtime information (via each subsystem's interface methods).
- Aggregate runtime state (via SnapshotBuilder).
- Produce one immutable RuntimeSnapshot.

Rules
-----
- NOT connected to Cloud Brain.
- NOT connected to Jarwis.py.
- NOT exposed through chat.
- Stdlib only. No I/O. No LLM calls.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from runtime.introspection.subsystem_interface import IntrospectableSubsystem
from runtime.introspection.subsystem_registry import SubsystemRegistry
from runtime.introspection.runtime_snapshot import RuntimeSnapshot, make_empty_snapshot
from runtime.introspection.snapshot_builder import SnapshotBuilder, SnapshotBuilderConfig

logger = logging.getLogger(__name__)


class IntrospectionEngine:
    """
    Runtime Introspection Engine.

    Discovers every registered subsystem, aggregates their runtime state,
    and produces a single immutable RuntimeSnapshot on demand.

    Parameters
    ----------
    registry : SubsystemRegistry, optional
        The registry of introspectable subsystems.  If not provided, an
        empty registry is created (useful when all subsystems will be
        registered after construction).
    builder_config : SnapshotBuilderConfig, optional
        Configuration for the snapshot builder (identity name, version, etc.).
    cache_last : bool
        When True, the last snapshot is cached and accessible via
        ``last_snapshot``.  Default True.

    Usage
    -----
    ::

        engine   = IntrospectionEngine()
        engine.register("cloud_brain", MyCloudBrainAdapter())
        snapshot = engine.build_snapshot()

        print(snapshot.runtime_health.overall)
        print(snapshot.connected_subsystems)
    """

    def __init__(
        self,
        registry: Optional[SubsystemRegistry] = None,
        builder_config: Optional[SnapshotBuilderConfig] = None,
        cache_last: bool = True,
    ) -> None:
        self._registry: SubsystemRegistry = registry or SubsystemRegistry()
        self._builder: SnapshotBuilder    = SnapshotBuilder(config=builder_config)
        self._cache_last: bool            = cache_last
        self._last_snapshot: Optional[RuntimeSnapshot] = None

    # ------------------------------------------------------------------
    # Primary public method
    # ------------------------------------------------------------------

    def build_snapshot(self) -> RuntimeSnapshot:
        """
        Discover all registered subsystems, collect their runtime state,
        and produce an immutable RuntimeSnapshot.

        Returns the new snapshot and, when cache_last=True, stores it
        for access via ``last_snapshot``.

        This method never raises — if the builder encounters an error
        it returns a partial snapshot with error information in diagnostics.
        """
        try:
            snapshot = self._builder.build(self._registry)
        except Exception as exc:
            logger.error("IntrospectionEngine.build_snapshot() failed: %s", exc)
            snapshot = make_empty_snapshot(
                identity_name=self._builder._config.identity_name,
                identity_version=self._builder._config.identity_version,
            )

        if self._cache_last:
            self._last_snapshot = snapshot

        logger.info(
            "IntrospectionEngine: snapshot %s built — "
            "%d subsystems, %d online, health=%s",
            snapshot.snapshot_id[:8],
            snapshot.subsystem_count,
            snapshot.online_count,
            snapshot.runtime_health.overall,
        )

        return snapshot

    # ------------------------------------------------------------------
    # Subsystem registration convenience methods
    # ------------------------------------------------------------------

    def register(self, name: str, subsystem: IntrospectableSubsystem) -> None:
        """Register a subsystem under ``name``."""
        self._registry.register(name, subsystem)

    def unregister(self, name: str) -> bool:
        """Unregister a subsystem by name.  Returns True if it existed."""
        return self._registry.unregister(name)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def registry(self) -> SubsystemRegistry:
        """The underlying SubsystemRegistry."""
        return self._registry

    @property
    def last_snapshot(self) -> Optional[RuntimeSnapshot]:
        """
        The most recently built snapshot, or None if ``build_snapshot()``
        has not been called yet.
        """
        return self._last_snapshot

    def registered_subsystem_names(self) -> List[str]:
        """Return the names of all registered subsystems."""
        return self._registry.names()

    def subsystem_count(self) -> int:
        """Return the number of registered subsystems."""
        return len(self._registry)

    def snapshot_summary(self) -> Dict[str, Any]:
        """
        Return a lightweight summary of the last snapshot.

        Returns an empty dict if no snapshot has been built yet.
        """
        snap = self._last_snapshot
        if snap is None:
            return {}
        return {
            "snapshot_id": snap.snapshot_id,
            "taken_at": snap.taken_at,
            "identity_name": snap.identity_name,
            "identity_version": snap.identity_version,
            "subsystem_count": snap.subsystem_count,
            "online_count": snap.online_count,
            "offline_count": snap.offline_count,
            "overall_health": snap.runtime_health.overall,
            "capability_count": len(snap.capabilities),
            "limitation_count": len(snap.limitations),
        }
