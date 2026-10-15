"""
self_info/task_assessor.py
===========================
Task / Capability Assessor

Answers "Can you do X?" type queries with a structured assessment
based solely on AB's verified capabilities and limitations.

Assessments are qualitative — no percentages.

Possible verdicts:
    FEASIBLE                  — AB can do this now.
    PARTIALLY_FEASIBLE        — AB can do part of it; something is missing.
    NOT_CURRENTLY_FEASIBLE    — AB cannot do this with current capabilities.
    REQUIRES_ADDITIONAL_CAPS  — The task would need new capabilities to be built.

Design rules:
- No LLM calls.
- Based on the CAPABILITIES and LIMITATIONS in ab_self_model.py.
- Returns a structured TaskAssessment object.
- Stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from self_info.ab_self_model import get_capabilities, get_limitations


# ---------------------------------------------------------------------------
# Verdict constants
# ---------------------------------------------------------------------------

VERDICT_FEASIBLE          = "FEASIBLE"
VERDICT_PARTIAL           = "PARTIALLY FEASIBLE"
VERDICT_NOT_FEASIBLE      = "NOT CURRENTLY FEASIBLE"
VERDICT_NEEDS_MORE        = "REQUIRES ADDITIONAL CAPABILITIES"


# ---------------------------------------------------------------------------
# Task Assessment result
# ---------------------------------------------------------------------------

@dataclass
class TaskAssessment:
    """
    Structured assessment for a requested task.

    Attributes
    ----------
    task_description : str
        The task the user asked about.
    verdict : str
        One of the four VERDICT_* constants.
    matching_capabilities : List[str]
        Capabilities AB has that are relevant to this task.
    blocking_limitations : List[str]
        Known limitations or missing capabilities that block full completion.
    dependencies : List[str]
        External systems or conditions required.
    notes : List[str]
        Additional context about the assessment.
    """
    task_description: str
    verdict: str
    matching_capabilities: List[str] = field(default_factory=list)
    blocking_limitations: List[str]  = field(default_factory=list)
    dependencies: List[str]          = field(default_factory=list)
    notes: List[str]                 = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "task":                   self.task_description,
            "verdict":                self.verdict,
            "matching_capabilities":  self.matching_capabilities,
            "blocking_limitations":   self.blocking_limitations,
            "dependencies":           self.dependencies,
            "notes":                  self.notes,
        }

    def to_text(self) -> str:
        """Return a plain-text summary suitable for injecting into a brain prompt."""
        lines = [
            f"Task: {self.task_description}",
            f"Assessment: {self.verdict}",
        ]
        if self.matching_capabilities:
            lines.append("Relevant capabilities: " + "; ".join(self.matching_capabilities))
        if self.blocking_limitations:
            lines.append("Blockers: " + "; ".join(self.blocking_limitations))
        if self.dependencies:
            lines.append("Dependencies: " + "; ".join(self.dependencies))
        if self.notes:
            lines.append("Notes: " + " ".join(self.notes))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Keyword maps — task keywords → capability/limitation keywords
# ---------------------------------------------------------------------------

# Each entry: (task_keyword, capability_name_fragment, limitation_name_fragment)
# We check if the task description contains the task_keyword; if so we look up
# whether the corresponding capability is present / that limitation is active.

_CAPABILITY_SIGNALS: List[tuple] = [
    # (task_keyword,         capability_name_substring)
    ("talk",                 "Natural-language conversation"),
    ("speak",                "Natural-language conversation"),
    ("chat",                 "Natural-language conversation"),
    ("converse",             "Natural-language conversation"),
    ("conversation",         "Natural-language conversation"),
    ("remember",             "Persistent memory"),
    ("memory",               "Persistent memory"),
    ("store",                "Persistent memory"),
    ("recall",               "Persistent memory"),
    ("forget",               "Persistent memory"),
    ("task",                 "Task management"),
    ("todo",                 "Task management"),
    ("to-do",                "Task management"),
    ("remind",               "Task management"),
    ("project",              "Project context awareness"),
    ("version",              "Project context awareness"),
    ("milestone",            "Project context awareness"),
    ("status",               "Project context awareness"),
    ("offline",              "Local brain (Ollama)"),
    ("local",                "Local brain (Ollama)"),
    ("ollama",               "Local brain (Ollama)"),
    ("cloud",                "Cloud brain (OpenRouter)"),
    ("openrouter",           "Cloud brain (OpenRouter)"),
    ("online",               "Cloud brain (OpenRouter)"),
    ("brain",                "Adaptive brain selection"),
    ("switch brain",         "Adaptive brain selection"),
    ("introspect",           "Runtime introspection"),
    ("runtime",              "Runtime introspection"),
    ("snapshot",             "Runtime introspection"),
    ("entity",               "Entity tracking"),
    ("track",                "Entity tracking"),
    ("reference",            "Reference resolution"),
    ("pronoun",              "Reference resolution"),
    ("token",                "Token budget estimation"),
    ("budget",               "Token budget estimation"),
    ("safe",                 "Response safety processing"),
    ("sanitize",             "Response safety processing"),
    ("log",                  "Analytics event logging"),
    ("analytics",            "Analytics event logging"),
    ("event",                "Analytics event logging"),
    ("file",                 "Memory file viewer"),
    ("open memory",          "Memory file viewer"),
    ("implement",            "Workspace and filesystem control"),
    ("implementation",       "Workspace and filesystem control"),
    ("edit code",            "Workspace and filesystem control"),
    ("modify code",          "Workspace and filesystem control"),
    ("change existing behavior", "Workspace and filesystem control"),
    ("code",                 "Workspace and filesystem control"),
    ("edit",                 "Workspace and filesystem control"),
    ("modify",               "Workspace and filesystem control"),
]

_LIMITATION_SIGNALS: List[tuple] = [
    # (task_keyword,  limitation_name_substring)
    ("voice",        "Voice recognition"),
    ("speak",        "Voice recognition"),
    ("listen",       "Voice recognition"),
    ("audio",        "Voice recognition"),
    ("microphone",   "Voice recognition"),
    ("camera",       "Camera awareness"),
    ("image",        "Camera awareness"),
    ("photo",        "Camera awareness"),
    ("video",        "Camera awareness"),
    ("screen",       "Camera awareness"),
    ("see",          "Camera awareness"),
    ("click",        "Desktop automation"),
    ("automate",     "Desktop automation"),
    ("automation",   "Desktop automation"),
    ("desktop",      "Desktop automation"),
    ("window",       "Desktop automation"),
    ("application",  "Desktop automation"),
    ("snap",         "Snap detection"),
    ("gesture",      "Snap detection"),
    ("background",   "Background mode"),
    ("daemon",       "Background mode"),
    ("always on",    "Background mode"),
    ("persistent session", "Ephemeral session state"),
    ("cross-session",      "Ephemeral session state"),
    ("long-term memory",   "Persistent storage integration"),
    ("database",           "Persistent storage integration"),
    ("search history",     "Persistent storage integration"),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def assess_task(task_description: str) -> TaskAssessment:
    """
    Assess whether AB can perform the described task.

    Parameters
    ----------
    task_description : str
        The task or capability the user asked about (e.g. "build a website",
        "remember things", "do voice recognition").

    Returns
    -------
    TaskAssessment
        Structured verdict with matched capabilities and blockers.
    """
    norm = task_description.strip().lower()
    caps = get_capabilities()
    lims = get_limitations()

    # --- Step 1: find matching capabilities ---
    matched_cap_names: List[str] = []
    matched_cap_records = []

    for signal, cap_name in _CAPABILITY_SIGNALS:
        if signal in norm:
            for cap in caps:
                if cap_name.lower() in cap["name"].lower() and cap["name"] not in matched_cap_names:
                    matched_cap_names.append(cap["name"])
                    matched_cap_records.append(cap)
                    break

    # --- Step 2: find blocking limitations ---
    blocking_lim_names: List[str] = []
    blocking_lim_records = []

    for signal, lim_name in _LIMITATION_SIGNALS:
        if signal in norm:
            for lim in lims:
                if lim_name.lower() in lim["name"].lower() and lim["name"] not in blocking_lim_names:
                    blocking_lim_names.append(lim["name"])
                    blocking_lim_records.append(lim)
                    break

    # --- Step 3: check if the task directly names a blocked feature ---
    # "blocked_features" from project_context — voice, camera, etc.
    project_blocked = {"voice", "camera", "snap detection", "desktop automation", "background mode"}
    directly_blocked = any(b in norm for b in project_blocked)
    if directly_blocked and not blocking_lim_records:
        for lim in lims:
            if lim.get("blocked_by") == "development_rule":
                for b in project_blocked:
                    if b in norm and b in lim["name"].lower():
                        blocking_lim_names.append(lim["name"])
                        blocking_lim_records.append(lim)

    # --- Step 4: collect dependencies ---
    dependencies: List[str] = []
    for cap in matched_cap_records:
        if cap.get("requires_external"):
            dependencies.append(f"External: {cap['name']} requires external system")
    for lim in blocking_lim_records:
        if lim.get("blocked_by") not in ("development_rule", "by_design", None):
            dependencies.append(f"{lim['name']} blocked by: {lim.get('blocked_by', 'unknown')}")

    # --- Step 5: determine verdict ---
    notes: List[str] = []

    if directly_blocked or (blocking_lim_records and not matched_cap_records):
        verdict = VERDICT_NOT_FEASIBLE
        notes.append("This capability is explicitly not implemented and blocked by the project development rule.")

    elif blocking_lim_records and matched_cap_records:
        verdict = VERDICT_PARTIAL
        notes.append("Jarwis has some relevant capabilities, but one or more blockers prevent full completion.")

    elif matched_cap_records:
        verdict = VERDICT_FEASIBLE
        notes.append("Jarwis has verified capabilities that cover this task.")
        if any(
            signal in norm
            for signal in (
                "implement", "implementation", "edit code", "modify code",
                "change existing behavior", "code", "edit", "modify",
            )
        ):
            notes.append(
                "File changes are limited to an accessible workspace and the "
                "existing filesystem controls; this is not unrestricted "
                "self-modification."
            )

    else:
        # No signals matched — we don't know
        verdict = VERDICT_NEEDS_MORE
        notes.append(
            "This task does not map to any known Jarwis capability or limitation. "
            "It may require capabilities that have not yet been built."
        )

    return TaskAssessment(
        task_description=task_description,
        verdict=verdict,
        matching_capabilities=matched_cap_names,
        blocking_limitations=blocking_lim_names,
        dependencies=dependencies,
        notes=notes,
    )
