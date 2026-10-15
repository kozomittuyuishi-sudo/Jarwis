"""
AG Beta — Control Layer, Phase A.

One cohesive subsystem, one runtime file (same principle as
schema_processor.py). Contains:

  1. Custom exceptions
  2. BrainRegistry
  3. AuthorizationManager
  4. Required-permissions mapping
  5. resolve_brain_request()

This layer is purely advisory:
  - never executes a model or makes a network/API call
  - never fabricates a model ID (all entries in brain_registry.json
    correspond to real IDs already used in brain.py/local_brain.py)
  - never changes the persistent brain mode (brain_config.json,
    read/written by brain.py, is untouched by this file entirely)
  - never prompts a user for input

Not wired into Jarwis.py yet (Phase A scope, per architecture rules).

Python standard library only. Deterministic. No LLM calls.
"""

import json
import os
from typing import Any, Dict, List, Optional

BRAIN_REGISTRY_FILE = "configuration/brain_registry.json"
PERMISSIONS_FILE = "configuration/permissions.json"

SENSITIVITY_RANK = {"low": 0, "medium": 1, "high": 2, "unknown": 2}


# ======================================================================
# 1. Custom exceptions
# ======================================================================


class ControlLayerError(Exception):
    """Base exception for Control Layer failures."""


class BrainRegistryError(ControlLayerError):
    pass


class AuthorizationError(ControlLayerError):
    pass


# ======================================================================
# 2. BrainRegistry
# ======================================================================


class BrainRegistry:
    """
    Registers and looks up local/cloud brains, resolves aliases, selects a
    suitable brain by capability/availability/priority/network requirement,
    builds fallback chains, and records success/failure/latency stats.

    Never executes a model. Never fabricates a brain/model entry -
    everything comes from brain_registry.json (or an in-memory dict
    passed to from_data(), useful for tests).
    """

    def __init__(self, registry_path: str = BRAIN_REGISTRY_FILE):
        self.registry_path = registry_path
        self._brains: Dict[str, Dict[str, Any]] = {}
        self._alias_map: Dict[str, str] = {}
        self._stats: Dict[str, Dict[str, Any]] = {}

        data: Dict[str, Any] = {}
        if os.path.exists(registry_path):
            try:
                with open(registry_path, "r", encoding="utf-8") as file:
                    data = json.load(file)
            except Exception as error:
                raise BrainRegistryError(f"Corrupt brain registry file '{registry_path}': {error}")

        self._ingest(data)

    @classmethod
    def from_data(cls, data: Dict[str, Any]) -> "BrainRegistry":
        instance = cls.__new__(cls)
        instance.registry_path = None
        instance._brains = {}
        instance._alias_map = {}
        instance._stats = {}
        instance._ingest(data)
        return instance

    def _ingest(self, data: Dict[str, Any]) -> None:
        for entry in data.get("brains", []):
            brain_id = entry.get("id")
            if not brain_id:
                continue

            self._brains[brain_id] = dict(entry)
            self._stats.setdefault(brain_id, {"success": 0, "failure": 0, "latencies": []})

            self._alias_map[brain_id.strip().lower()] = brain_id
            for alias in entry.get("aliases", []):
                self._alias_map[str(alias).strip().lower()] = brain_id

    def list_brains(self, provider: Optional[str] = None, enabled_only: bool = False) -> List[Dict[str, Any]]:
        brains = list(self._brains.values())
        if provider:
            brains = [b for b in brains if b.get("provider") == provider]
        if enabled_only:
            brains = [b for b in brains if b.get("enabled", True)]
        return sorted(brains, key=lambda b: (b.get("priority", 999), b.get("id", "")))

    def resolve_alias(self, name: str) -> Optional[str]:
        if not name:
            return None
        return self._alias_map.get(str(name).strip().lower())

    def get_brain(self, brain_id: str) -> Optional[Dict[str, Any]]:
        brain = self._brains.get(brain_id)
        return dict(brain) if brain is not None else None

    def _candidates(
        self,
        provider: Optional[str],
        required_capabilities: Optional[List[str]],
        network_available: bool,
    ) -> List[Dict[str, Any]]:
        required = set(required_capabilities or [])
        candidates = []

        for brain in self._brains.values():
            if not brain.get("enabled", True):
                continue
            if provider and brain.get("provider") != provider:
                continue
            if brain.get("requires_network") and not network_available:
                continue
            if required and not required.issubset(set(brain.get("capabilities", []))):
                continue
            candidates.append(brain)

        candidates.sort(key=lambda b: (b.get("priority", 999), b.get("id", "")))
        return candidates

    def select_brain(
        self,
        provider: Optional[str] = None,
        required_capabilities: Optional[List[str]] = None,
        network_available: bool = True,
    ) -> Optional[Dict[str, Any]]:
        candidates = self._candidates(provider, required_capabilities, network_available)
        return dict(candidates[0]) if candidates else None

    def build_fallback_chain(
        self,
        provider: Optional[str] = None,
        required_capabilities: Optional[List[str]] = None,
        network_available: bool = True,
    ) -> List[str]:
        candidates = self._candidates(provider, required_capabilities, network_available)
        return [b["id"] for b in candidates]

    def record_result(self, brain_id: str, success: bool, latency: Optional[float] = None) -> None:
        stats = self._stats.setdefault(brain_id, {"success": 0, "failure": 0, "latencies": []})
        if success:
            stats["success"] += 1
        else:
            stats["failure"] += 1
        if latency is not None:
            stats["latencies"].append(latency)

    def get_stats(self, brain_id: str) -> Dict[str, Any]:
        return dict(self._stats.get(brain_id, {"success": 0, "failure": 0, "latencies": []}))


# ======================================================================
# 3. AuthorizationManager
# ======================================================================


class AuthorizationManager:
    """
    Evaluates permissions. Undefined permissions deny by default.
    Distinguishes authorization ("allowed") from confirmation
    ("requires_confirmation") - these are separate axes, not the same
    thing. Never prompts or executes anything; purely evaluates.
    """

    def __init__(self, permissions_path: str = PERMISSIONS_FILE):
        self.permissions_path = permissions_path
        self._permissions: Dict[str, Dict[str, Any]] = {}

        data: Dict[str, Any] = {}
        if os.path.exists(permissions_path):
            try:
                with open(permissions_path, "r", encoding="utf-8") as file:
                    data = json.load(file)
            except Exception as error:
                raise AuthorizationError(f"Corrupt permissions file '{permissions_path}': {error}")

        self._permissions = data.get("permissions", {})

    @classmethod
    def from_data(cls, data: Dict[str, Any]) -> "AuthorizationManager":
        instance = cls.__new__(cls)
        instance.permissions_path = None
        instance._permissions = data.get("permissions", {})
        return instance

    def get_permission(self, name: str) -> Dict[str, Any]:
        entry = self._permissions.get(name)
        if entry is None:
            # Deny undefined permissions by default.
            return {"allowed": False, "requires_confirmation": True, "sensitivity": "unknown", "defined": False}

        result = dict(entry)
        result.setdefault("allowed", False)
        result.setdefault("requires_confirmation", True)
        result.setdefault("sensitivity", "unknown")
        result["defined"] = True
        return result

    def authorize(self, required_permissions: List[str]) -> Dict[str, Any]:
        """AND semantics: every required permission must independently be allowed."""
        checked = []
        allowed = True
        requires_confirmation = False
        highest_sensitivity = "low"

        for name in required_permissions:
            entry = self.get_permission(name)
            checked.append({"permission": name, **entry})

            if not entry["allowed"]:
                allowed = False

            if entry["requires_confirmation"]:
                requires_confirmation = True

            if SENSITIVITY_RANK.get(entry["sensitivity"], 2) > SENSITIVITY_RANK.get(highest_sensitivity, 0):
                highest_sensitivity = entry["sensitivity"]

        denied_permissions = [c["permission"] for c in checked if not c["allowed"]]

        return {
            "allowed": allowed,
            "requires_confirmation": requires_confirmation,
            "sensitivity": highest_sensitivity,
            "denied_permissions": denied_permissions,
            "checked": checked,
        }


# ======================================================================
# 4. Required-permissions mapping
# ======================================================================


def _required_permissions_for(provider: str, extra_permissions: Optional[List[str]] = None) -> List[str]:
    permissions: List[str] = []

    if provider == "local":
        permissions.append("brain.local.use")
    elif provider == "cloud":
        permissions.append("brain.cloud.use")
        permissions.append("internet.access")

    if extra_permissions:
        permissions.extend(extra_permissions)

    return permissions


# ======================================================================
# 5. resolve_brain_request()
# ======================================================================


def resolve_brain_request(
    provider: str,
    registry: BrainRegistry,
    authorization: AuthorizationManager,
    required_capabilities: Optional[List[str]] = None,
    network_available: bool = True,
    extra_permissions: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    1. Determine required permissions for this provider.
    2. Authorize the request.
    3. If denied -> no selected brain.
    4. If allowed -> select a suitable brain (and build a fallback chain).
    5. Return selected brain, fallback chain, confirmation requirement, reason.

    Never calls an API. Never changes persistent brain mode - this
    function has no knowledge of brain_config.json and cannot write it.
    """
    permissions = _required_permissions_for(provider, extra_permissions)
    auth_result = authorization.authorize(permissions)

    if not auth_result["allowed"]:
        return {
            "selected_brain": None,
            "fallback_chain": [],
            "requires_confirmation": auth_result["requires_confirmation"],
            "authorized": False,
            "reason": f"Denied: missing or disallowed permission(s) {auth_result['denied_permissions']}.",
            "required_permissions": permissions,
        }

    selected = registry.select_brain(
        provider=provider,
        required_capabilities=required_capabilities,
        network_available=network_available,
    )

    fallback_chain = registry.build_fallback_chain(
        provider=None,  # broaden fallback beyond just this provider
        required_capabilities=required_capabilities,
        network_available=network_available,
    )
    if selected:
        fallback_chain = [brain_id for brain_id in fallback_chain if brain_id != selected["id"]]

    if selected is None:
        return {
            "selected_brain": None,
            "fallback_chain": fallback_chain,
            "requires_confirmation": auth_result["requires_confirmation"],
            "authorized": True,
            "reason": f"No enabled '{provider}' brain is currently available (network_available={network_available}).",
            "required_permissions": permissions,
        }

    return {
        "selected_brain": selected["id"],
        "fallback_chain": fallback_chain,
        "requires_confirmation": auth_result["requires_confirmation"],
        "authorized": True,
        "reason": f"Selected '{selected['id']}' for provider '{provider}'.",
        "required_permissions": permissions,
    }
