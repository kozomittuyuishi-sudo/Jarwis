"""
Runtime Integrity Check
Verifies: Jarwis.py startup, runtime chain, module loading,
          configuration loading, environment loading.
"""

import importlib
import importlib.util
import os
from pathlib import Path


# Core runtime modules that must be importable
RUNTIME_CHAIN = [
    "Jarwis",
    "runtime.runtime_adapter",
    "runtime.introspection",
    "runtime.introspection.introspection_engine",
    "runtime.introspection.snapshot_builder",
    "runtime.introspection.subsystem_registry",
    "cognition.executive",
    "cognition.executive_layer",
    "memory.working_memory",
    "conversation.conversation_manager",
    "conversation.response_processor",
    "brains.brain",
    "brains.brain_dispatcher",
    "brains.cloud_brain",
    "diagnostics.analytics_logger",
]

REQUIRED_CONFIG_FILES = [
    "configuration/memory.json",
    "configuration/tasks.json",
    "configuration/brain_config.json",
    "configuration/brain_registry.json",
    "configuration/project_context.json",
    "configuration/permissions.json",
]

ENV_FILE = ".venv/.env"


def run(project_root: str) -> dict:
    issues = []
    root = Path(project_root)

    # --- Module loading ---
    for module_name in RUNTIME_CHAIN:
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            issues.append(f"MODULE_NOT_FOUND: {module_name}")
        else:
            try:
                importlib.import_module(module_name)
            except Exception as exc:
                issues.append(f"MODULE_IMPORT_ERROR: {module_name} — {exc}")

    # --- Configuration loading ---
    for cfg_path in REQUIRED_CONFIG_FILES:
        full_path = root / cfg_path
        if not full_path.exists():
            issues.append(f"MISSING_CONFIG: {cfg_path}")
        else:
            try:
                import json
                with open(full_path, encoding="utf-8") as f:
                    json.load(f)
            except Exception as exc:
                issues.append(f"INVALID_CONFIG: {cfg_path} — {exc}")

    # --- Environment loading ---
    env_path = root / ENV_FILE
    if not env_path.exists():
        issues.append(f"MISSING_ENV: {ENV_FILE}")

    # --- Jarwis.py startup check (syntax + parseable) ---
    entry_path = root / "Jarwis.py"
    if not entry_path.exists():
        issues.append("MISSING_ENTRY_POINT: Jarwis.py not found at project root")
    else:
        try:
            import ast
            source = entry_path.read_text(encoding="utf-8")
            ast.parse(source)
        except SyntaxError as exc:
            issues.append(f"SYNTAX_ERROR_IN_ENTRY: Jarwis.py — {exc}")

    return {
        "check": "runtime_integrity",
        "issues": issues,
        "passed": len(issues) == 0,
    }
