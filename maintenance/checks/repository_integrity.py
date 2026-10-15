"""
Repository Integrity Check
Checks: misplaced files, empty Python files, duplicate modules,
        missing package __init__.py files, orphan modules.
"""

import os
from pathlib import Path
from collections import defaultdict


# Packages that must contain __init__.py
REQUIRED_PACKAGES = [
    "cognition",
    "memory",
    "conversation",
    "conversation/core",
    "conversation/cognitive",
    "brains",
    "brains/cloud_brain",
    "runtime",
    "runtime/introspection",
    "diagnostics",
    "storage",
    "pipeline",
    "tests",
]

# Files that belong only at root level
ROOT_ONLY_FILES = ["Jarwis.py"]

# Directories that should not contain .py files
NON_PYTHON_DIRS = [
    "scripts", "docs", "data", "platform-tools",
    ".venv", ".aider.tags.cache.v4", ".pytest_cache",
    "architecture", "automation timeline", "node_modules",
]


def run(project_root: str) -> dict:
    issues = []
    root = Path(project_root)

    python_files = [
        p for p in root.rglob("*.py")
        if ".venv" not in p.parts
        and "__pycache__" not in p.parts
        and "maintenance" not in p.parts
        and "node_modules" not in p.parts
    ]

    # --- Missing __init__.py ---
    for pkg in REQUIRED_PACKAGES:
        init_path = root / pkg / "__init__.py"
        if not init_path.exists():
            issues.append(f"MISSING_INIT: {pkg}/__init__.py")

    # --- Empty Python files ---
    for filepath in python_files:
        try:
            content = filepath.read_text(encoding="utf-8").strip()
        except Exception:
            continue
        if not content:
            issues.append(f"EMPTY_FILE: {filepath.relative_to(root)}")

    # --- Duplicate module names ---
    module_names: dict[str, list[str]] = defaultdict(list)
    for filepath in python_files:
        stem = filepath.stem
        if stem.startswith("__"):
            continue
        module_names[stem].append(str(filepath.relative_to(root)))

    for name, paths in module_names.items():
        if len(paths) > 1:
            issues.append(
                f"DUPLICATE_MODULE: '{name}' found at {', '.join(paths)}"
            )

    # --- Misplaced files: ROOT_ONLY_FILES found outside root ---
    for name in ROOT_ONLY_FILES:
        for filepath in root.rglob(name):
            if filepath.parent != root:
                if (
                    ".venv" not in filepath.parts
                    and "__pycache__" not in filepath.parts
                    and "node_modules" not in filepath.parts
                ):
                    issues.append(f"MISPLACED_FILE: {filepath.relative_to(root)}")

    # --- Python files in non-Python directories ---
    for filepath in python_files:
        for bad_dir in NON_PYTHON_DIRS:
            if bad_dir in filepath.parts:
                issues.append(
                    f"MISPLACED_PYTHON: {filepath.relative_to(root)} is in {bad_dir}/"
                )
                break

    # --- Orphan modules: .py files not imported anywhere ---
    all_source = {}
    for filepath in python_files:
        try:
            all_source[str(filepath.relative_to(root))] = filepath.read_text(encoding="utf-8")
        except Exception:
            continue

    combined = "\n".join(all_source.values())

    for filepath in python_files:
        stem = filepath.stem
        if stem.startswith("__") or stem == "Jarwis":
            continue
        # Check if the module name appears in any import statement
        pkg_path = filepath.relative_to(root)
        pkg_parts = list(pkg_path.with_suffix("").parts)
        dotted = ".".join(pkg_parts)
        if (
            f"import {stem}" not in combined
            and f"from {dotted}" not in combined
            and f"import {dotted}" not in combined
            and not str(pkg_path).startswith("tests" + os.sep)
            and not str(pkg_path).startswith("maintenance" + os.sep)
        ):
            issues.append(f"ORPHAN_MODULE: {pkg_path} — not imported anywhere")

    return {
        "check": "repository_integrity",
        "issues": issues,
        "passed": len(issues) == 0,
    }
