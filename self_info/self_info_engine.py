"""
self_info/self_info_engine.py
==============================
Self-Info Engine

Top-level entry point for the Self-Info system.

Responsibility:
    Given a user query, determine if it is a self-info query, route it
    to the correct section of the self-model, assemble a structured
    context block, and return it — ready for the caller to either use
    directly or pass to a brain for natural-language formatting.

Architecture:
    SelfInfoEngine.answer(query) → SelfInfoResult
        .is_self_info     — whether this engine handled the query
        .context_block    — verified structured facts as formatted text
        .route            — the QueryRoute used
        .assessment       — TaskAssessment (if applicable)

The Cloud Brain MAY receive context_block to generate a natural-language
response, but it must NOT invent facts — all facts are pre-assembled here.

Design rules:
- No LLM calls inside this engine.
- Optionally accepts a RuntimeSnapshot from the IntrospectionEngine to
  enrich the context with live runtime state.
- Never raises — returns a degraded SelfInfoResult on any error.
- Stdlib only.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from self_info.query_router import (
    QueryRoute,
    route_query,
    ROUTE_IDENTITY,
    ROUTE_ARCHITECTURE,
    ROUTE_CAPABILITIES,
    ROUTE_LIMITATIONS,
    ROUTE_OPERATIONAL,
    ROUTE_TASK_ASSESSMENT,
    ROUTE_GENERAL,
)
from self_info.task_assessor import assess_task, TaskAssessment
from self_info.ab_self_model import (
    get_identity,
    get_architecture,
    get_capabilities,
    get_limitations,
    get_operational_info,
    get_implemented_capabilities,
    get_external_capabilities,
    get_limitations_by_category,
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SelfInfoResult:
    """
    Result of a SelfInfoEngine.answer() call.

    Attributes
    ----------
    is_self_info : bool
        True when the engine handled this query.
    context_block : str
        Verified structured context, formatted as plain text.
        Pass this to a brain prompt as the authoritative source of facts.
    route : QueryRoute
        The routing decision that was made.
    assessment : TaskAssessment, optional
        Only populated for task-assessment queries.
    error : str, optional
        Error message if something went wrong internally.
    """
    is_self_info: bool
    context_block: str
    route: QueryRoute
    assessment: Optional[TaskAssessment] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Self-Info Engine
# ---------------------------------------------------------------------------

class SelfInfoEngine:
    """
    Self-Info Engine for AB.

    Answers self-referential queries from verified structured data.
    Optionally accepts a RuntimeSnapshot for live enrichment.

    Usage
    -----
    ::

        engine = SelfInfoEngine()
        result = engine.answer("Who are you?")

        if result.is_self_info:
            # Use result.context_block as authoritative context
            prompt = f"{result.context_block}\\n\\nUser question: {user_input}"
            reply = ask_brain(prompt)
    """

    def answer(
        self,
        user_input: str,
        snapshot: Optional[Any] = None,
    ) -> SelfInfoResult:
        """
        Route the query and assemble a verified context block.

        Parameters
        ----------
        user_input : str
            The user's question.
        snapshot : RuntimeSnapshot, optional
            Live snapshot from IntrospectionEngine for runtime enrichment.

        Returns
        -------
        SelfInfoResult
            Always returns a result, never raises.
        """
        try:
            route = route_query(user_input)

            if not route.is_self_info:
                return SelfInfoResult(
                    is_self_info=False,
                    context_block="",
                    route=route,
                )

            if route.category == ROUTE_TASK_ASSESSMENT:
                return self._handle_task_assessment(user_input, route, snapshot)

            if route.category == ROUTE_IDENTITY:
                return self._handle_identity(route, snapshot)

            if route.category == ROUTE_ARCHITECTURE:
                return self._handle_architecture(route, snapshot)

            if route.category == ROUTE_CAPABILITIES:
                return self._handle_capabilities(route, snapshot)

            if route.category == ROUTE_LIMITATIONS:
                return self._handle_limitations(route, snapshot)

            if route.category == ROUTE_OPERATIONAL:
                return self._handle_operational(route, snapshot)

            # ROUTE_GENERAL or unknown — assemble a combined overview
            return self._handle_general(route, snapshot)

        except Exception as exc:
            return SelfInfoResult(
                is_self_info=True,
                context_block="[Self-Info engine error — structured data unavailable]",
                route=QueryRoute(is_self_info=True, category=ROUTE_GENERAL, original_query=user_input),
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Category handlers
    # ------------------------------------------------------------------

    def _handle_identity(self, route: QueryRoute, snapshot: Optional[Any]) -> SelfInfoResult:
        identity = get_identity()
        lines = [
            "IDENTITY (assistant identity; project metadata from project_context.json):",
            f"  Name:     {identity['name']} ({identity['full_name']})",
            f"  Short name: {identity['short_name']}",
            f"  Active persona: {identity['active_persona']}",
            f"  Version:  {identity['version']}",
            f"  Phase:    {identity['phase']}",
            f"  Purpose:  {identity['purpose']}",
            f"  Type:     {identity['project_type']}",
        ]
        if identity.get("creator"):
            lines.append(f"  Creator:  {identity['creator']}")
        if snapshot:
            lines.append(f"  Runtime identity: {snapshot.identity_name} v{snapshot.identity_version}")
        return SelfInfoResult(
            is_self_info=True,
            context_block="\n".join(lines),
            route=route,
        )

    def _handle_architecture(self, route: QueryRoute, snapshot: Optional[Any]) -> SelfInfoResult:
        arch = get_architecture()
        lines = ["ARCHITECTURE (verified from codebase):"]

        lines.append("\nMajor Modules:")
        for mod, desc in arch["major_modules"].items():
            lines.append(f"  {mod}: {desc}")

        lines.append("\nRuntime Flow:")
        for step in arch["runtime_flow"]:
            lines.append(f"  → {step}")

        lines.append("\nSubsystems:")
        for sub in arch["subsystems"]:
            lines.append(f"  • {sub}")

        if snapshot:
            lines.append(f"\nLive Connected Subsystems: {', '.join(snapshot.connected_subsystems) or 'none'}")
            if snapshot.disconnected_subsystems:
                lines.append(f"Live Offline Subsystems: {', '.join(snapshot.disconnected_subsystems)}")

        return SelfInfoResult(
            is_self_info=True,
            context_block="\n".join(lines),
            route=route,
        )

    def _handle_capabilities(self, route: QueryRoute, snapshot: Optional[Any]) -> SelfInfoResult:
        caps = get_capabilities()
        lines = ["CAPABILITIES (verified from codebase):"]

        for cap in caps:
            status = "✓ Implemented" if cap["implemented"] else "✗ Not implemented"
            ext    = " [requires external system]" if cap["requires_external"] else ""
            lines.append(f"  {status}{ext}: {cap['name']}")
            lines.append(f"    {cap['description']}")

        if snapshot and snapshot.capabilities:
            lines.append("\nLive Subsystem Capabilities (from runtime snapshot):")
            for c in snapshot.capabilities[:15]:
                lines.append(f"  • {c}")

        return SelfInfoResult(
            is_self_info=True,
            context_block="\n".join(lines),
            route=route,
        )

    def _handle_limitations(self, route: QueryRoute, snapshot: Optional[Any]) -> SelfInfoResult:
        lims = get_limitations()
        lines = ["LIMITATIONS (verified from codebase):"]

        # Group by category
        categories = {}
        for lim in lims:
            cat = lim.get("category", "other")
            categories.setdefault(cat, []).append(lim)

        category_labels = {
            "missing_capability":  "Missing Capabilities",
            "incomplete_system":   "Incomplete / Partially Implemented",
            "runtime_limitation":  "Runtime Limitations",
            "external_dependency": "External Dependencies",
        }
        for cat, label in category_labels.items():
            group = categories.get(cat, [])
            if group:
                lines.append(f"\n  [{label}]")
                for lim in group:
                    blocked = f" (blocked by: {lim['blocked_by']})" if lim.get("blocked_by") else ""
                    lines.append(f"  • {lim['name']}{blocked}")
                    lines.append(f"    {lim['description']}")

        if snapshot and snapshot.limitations:
            lines.append("\nLive Subsystem Limitations (from runtime snapshot):")
            for lim in snapshot.limitations[:10]:
                lines.append(f"  • {lim}")

        return SelfInfoResult(
            is_self_info=True,
            context_block="\n".join(lines),
            route=route,
        )

    def _handle_operational(self, route: QueryRoute, snapshot: Optional[Any]) -> SelfInfoResult:
        ops = get_operational_info()
        lines = ["OPERATIONAL STATE (verified from configuration/):"]

        lines.append(f"  Version:   {ops['current_version']}")
        lines.append(f"  Phase:     {ops['current_phase']}")
        lines.append(f"  Objective: {ops['current_objective']}")
        lines.append(f"  Next milestone: {ops['next_milestone']}")
        lines.append(f"  Active persona: {ops['active_persona']}")
        lines.append(f"  Configured brain mode: {ops['brain_mode']}")

        if ops["brains"]:
            lines.append("\nAvailable Brains:")
            for brain in ops["brains"]:
                net = "requires network" if brain["requires_network"] else "offline"
                ena = "enabled" if brain["enabled"] else "disabled"
                lines.append(f"  • {brain['id']}  [{brain['provider']} / {net} / {ena}]")
                lines.append(f"    Model: {brain['model']}")
                if brain["capabilities"]:
                    lines.append(f"    Capabilities: {', '.join(brain['capabilities'])}")

        if ops["blocked_features"]:
            lines.append(f"\nBlocked features (development rule): {', '.join(ops['blocked_features'])}")

        if snapshot:
            lines.append(f"\nLive Runtime Health: {snapshot.runtime_health.overall}")
            if snapshot.brains:
                lines.append("Live Brain Status: " + ", ".join(f"{k}={v}" for k, v in snapshot.brains.items()))

        return SelfInfoResult(
            is_self_info=True,
            context_block="\n".join(lines),
            route=route,
        )

    def _handle_task_assessment(
        self,
        user_input: str,
        route: QueryRoute,
        snapshot: Optional[Any],
    ) -> SelfInfoResult:
        target = route.assessment_target or user_input
        assessment = assess_task(target)

        # Build context block with assessment facts + relevant self-model data
        lines = [
            "TASK ASSESSMENT (based on verified Jarwis capabilities):",
            assessment.to_text(),
        ]

        # Enrich with a compact capabilities summary
        impl = get_implemented_capabilities()
        lines.append(f"\nCurrently implemented capabilities ({len(impl)}):")
        for cap_name in impl:
            lines.append(f"  • {cap_name}")

        return SelfInfoResult(
            is_self_info=True,
            context_block="\n".join(lines),
            route=route,
            assessment=assessment,
        )

    def _handle_general(self, route: QueryRoute, snapshot: Optional[Any]) -> SelfInfoResult:
        """Compact overview — catches catch-all self-info queries."""
        identity = get_identity()
        ops      = get_operational_info()
        impl     = get_implemented_capabilities()

        lines = [
            f"JARWIS OVERVIEW:",
            f"  {identity['name']} ({identity['full_name']}) — Version {identity['version']}",
            f"  Active persona: {identity['active_persona']}",
            f"  Phase: {identity['phase']}",
            f"  Purpose: {identity['purpose']}",
        ]
        if identity.get("creator"):
            lines.append(f"  Creator: {identity['creator']}")
        lines.append(f"\nImplemented capabilities ({len(impl)}):")
        for cap_name in impl:
            lines.append(f"  • {cap_name}")

        if ops["brains"]:
            lines.append(f"\nAvailable brains: " + ", ".join(b["id"] for b in ops["brains"]))

        if snapshot:
            lines.append(f"\nLive health: {snapshot.runtime_health.overall}")

        return SelfInfoResult(
            is_self_info=True,
            context_block="\n".join(lines),
            route=route,
        )
