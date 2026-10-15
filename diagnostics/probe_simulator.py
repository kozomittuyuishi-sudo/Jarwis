"""
AG Probe Simulator — Phase A.

Standalone risk/outcome simulator for actions before AG executes them.
Deterministic rules only, no LLM calls, no external dependencies.
Not wired into Jarwis.py yet — this module can be called and tested on its own.

Expects action dicts shaped like conversation_manager.py's action schema:
{
  "action_type": str, "operation": str, "target": any,
  "confidence": float, "depends_on": list[str],
  "parameters": dict, "execution_policy": dict, ...
}
context (optional) may carry "missing_data": list[str] from an
action_result if the action dict itself doesn't already have one.
"""

from typing import Any, Dict, List, Optional

HIGH_RISK_KEYWORDS = [
    "delete", "erase", "remove", "overwrite", "wipe", "purge", "destroy", "format"
]

LOW_RISK_ACTION_TYPES = {"continuation", "summary"}
LOW_RISK_OPERATIONS = {"summarize", "continue", "explain"}


def estimate_risk(action: Dict[str, Any]) -> str:
    """Deterministic risk classification: 'low' | 'medium' | 'high'."""
    action_type = str(action.get("action_type", "")).lower()
    operation = str(action.get("operation", "")).lower()
    target = str(action.get("target", "")).lower()
    text_blob = " ".join([action_type, operation, target])

    # Deletion / overwrite / erase / file operations / permanent brain switch
    if any(keyword in text_blob for keyword in HIGH_RISK_KEYWORDS):
        return "high"

    if action_type == "brain_mode_change" or operation == "set_default_brain":
        return "high"

    if "file" in action_type or "file" in operation:
        return "high"

    if action_type == "unknown":
        return "high"

    # Temporary cloud/local brain usage for a single thread
    if operation == "temporary_brain_use" or action_type == "brain_routing":
        return "medium"

    # Normal explanation, summary, continuation, ordinary task/storage/project work
    if action_type in LOW_RISK_ACTION_TYPES or operation in LOW_RISK_OPERATIONS:
        return "low"

    if action_type in ("storage", "task", "project_control"):
        return "low"

    return "medium"


def detect_dependencies(action: Dict[str, Any]) -> List[str]:
    """Collects explicit depends_on entries plus inferred structural dependencies."""
    dependencies = list(action.get("depends_on") or [])

    action_type = action.get("action_type")
    operation = action.get("operation")
    parameters = action.get("parameters") or {}

    if operation == "temporary_brain_use" or parameters.get("temporary"):
        dependencies.append("requires the target reasoning provider (cloud/local) to be reachable")

    if action_type == "brain_mode_change":
        dependencies.append("requires brain_config.json to be writable")

    if action_type == "storage":
        dependencies.append("requires the target memory category to exist or be creatable")

    if action_type == "task":
        dependencies.append("requires tasks.json to be writable")

    return dependencies


def generate_alternatives(action: Dict[str, Any]) -> List[str]:
    operation = action.get("operation")
    risk_level = estimate_risk(action)

    if operation == "temporary_brain_use":
        return [
            "Answer using the current default brain mode instead",
            "Ask for clarification on scope before switching providers"
        ]

    if operation == "set_default_brain":
        return ["Use temporary brain use instead, to avoid a permanent mode change"]

    if risk_level == "high":
        return [
            "Request explicit confirmation before proceeding",
            "Perform a dry run or preview instead of executing directly"
        ]

    return ["Proceed as requested"]


def needs_confirmation(action: Dict[str, Any], risk_level: str) -> bool:
    if risk_level in ("high", "medium"):
        return True

    confidence = action.get("confidence")
    if confidence is None:
        confidence = 1.0
    if confidence < 0.7:
        return True

    execution_policy = action.get("execution_policy") or {}
    if execution_policy.get("requires_confirmation"):
        return True

    return False


def build_probe_report(action: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    context = context or {}

    action_type = action.get("action_type", "unknown")
    operation = action.get("operation", "unknown")

    confidence = action.get("confidence")
    if confidence is None:
        confidence = context.get("confidence", 1.0)
    if confidence is None:
        confidence = 1.0

    missing_data = action.get("missing_data") or context.get("missing_data") or []

    risk_level = estimate_risk(action)
    dependencies = detect_dependencies(action)
    alternatives = generate_alternatives(action)
    requires_confirmation = needs_confirmation(action, risk_level)

    risks: List[str] = []

    if risk_level == "high":
        risks.append(f"'{operation}' is a high-impact, potentially irreversible operation.")
    elif risk_level == "medium":
        risks.append(f"'{operation}' changes AG's behavior for this request only, but still needs a clear yes.")

    if confidence < 0.7:
        risks.append(f"Confidence is low ({confidence:.2f}); intent may be misread.")

    if missing_data:
        risks.append(f"Missing required data: {', '.join(missing_data)}.")

    if action_type == "unknown":
        risks.append("Action type is unknown; AG cannot safely classify this request.")

    should_proceed = True

    if missing_data:
        should_proceed = False

    if action_type == "unknown":
        should_proceed = False

    if should_proceed:
        outcome = (
            f"AG can perform '{operation}', pending user confirmation."
            if requires_confirmation
            else f"AG can perform '{operation}' directly."
        )
    else:
        outcome = f"AG cannot safely perform '{operation}' yet."

    if not should_proceed:
        recommendation = "Do not execute. Gather missing information or clarify the request first."
    elif requires_confirmation:
        recommendation = "Ask the user to confirm before executing."
    else:
        recommendation = "Safe to execute without additional confirmation."

    return {
        "should_proceed": should_proceed,
        "risk_level": risk_level,
        "confidence": confidence,
        "requires_confirmation": requires_confirmation,
        "outcome": outcome,
        "risks": risks,
        "dependencies": dependencies,
        "alternatives": alternatives,
        "recommendation": recommendation
    }


def simulate_action(action: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Simulates the likely outcome of an action before AG executes it.
    Deterministic, no LLM calls, safe to call standalone.
    """
    if not isinstance(action, dict):
        action = {}

    return build_probe_report(action, context)