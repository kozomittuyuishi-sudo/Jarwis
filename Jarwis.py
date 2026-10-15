import os as _os
import re
from dataclasses import replace
from pathlib import Path

from typing import Optional

from cognition.executive import (
    interpret_storage_decision,
    interpret_category_decision,
    analyze_context,
    build_execution_plan
)
from memory.working_memory import WorkingMemory, resolve_followup
from conversation.conversation_manager import analyze_action_pattern
from diagnostics.analytics_logger import log_event

# --- Runtime Integration imports ---
from conversation.conversation_manager import ConversationManager, build_context as cm_build_context
from conversation.core.entity_tracker import EntityTracker
from conversation.cognitive.reference_resolver import ReferenceResolver
from conversation.cognitive.evaluator import CognitiveEvaluator
from conversation.core.thought import Thought
from cognition.executive_layer import ExecutiveLayer
from brains.brain_dispatcher import BrainDispatcher
from conversation.response_processor import ResponseProcessor
from brains.cloud_brain import AdaptiveCloudBrain
from runtime.introspection import IntrospectionEngine
from runtime.introspection.snapshot_builder import SnapshotBuilderConfig
from runtime.runtime_adapter import build_registry
from self_info import SelfInfoEngine, is_self_info_query
from self_info.ab_self_model import get_identity
from runtime.persona_controller import get_persona_controller
from directory_reader import DirectorySession
from directory_reader.context_builder import build_context, build_prompt
from directory_control import (
    DirectoryControl, FileNotFoundInWorkspace, NotAFileError, VerificationError,
)
from directory_control.path_security import (
    NoActiveWorkspaceError,
    WorkspaceViolationError,
    WorkspaceNotFoundError,
    DirectoryControlError,
)
from pipeline.fallback_classifier import (
    classify_failure,
    get_fallback_classifier,
    CATEGORY_CAPABILITY_UNAVAILABLE,
    CATEGORY_UNKNOWN_CAPABILITY,
    CATEGORY_INFORMATION_UNAVAILABLE,
    CATEGORY_BRAIN_FAILURE,
    CATEGORY_INVALID_RESPONSE,
    CATEGORY_PROCESSING_FAILURE,
    AGError,
    DIR_NOT_FOUND,
    DIR_NOT_A_FILE,
    DIR_NOT_A_DIRECTORY,
    DIR_SOURCE_MISSING,
    DIR_OUT_OF_WORKSPACE,
)
from pipeline.intent_router import classify as _route_intent, Intent
from preprocessor_brain import PreprocessorBrain, Domain as _PBDomain
from target_resolution_models import ResolutionMode, ResolutionStatus, TargetType
from target_resolution_models import ActionHint, TargetReference
from target_resolver import TargetResolver, looks_absolute
from quips import (
    QUIPSResult, RequestCategory, RequestUnit, RoutingDecision, UnitProcessingResult,
    DesiredOutcome,
    analyze_request, resolve_target_reference,
)
import json
import os
import subprocess
from datetime import datetime

import dotenv

try:
    from brains.brain import (
        ask_brain,
        ask_cloud_direct,
        ask_local,
        set_brain_mode,
        get_brain_status,
        load_brain_mode
    )
except ImportError:
    from brains.brain import (
        ask_cloud_direct,
        ask_local,
        set_brain_mode,
        get_brain_status,
        load_brain_mode
    )

    def ask_brain(message, include_persona=True):
        """Fallback for environments where ask_brain is not exported."""
        return ask_local(message, include_persona=include_persona)


def _semantic_interpretation_fallback(
    raw_input: str,
    memory_context: dict,
) -> dict:
    """Ask the configured Brain for interpretation data only, never an operation."""
    recent_events = [
        {
            "action_id": event.get("action_id"),
            "action": event.get("action"),
            "name": event.get("name"),
            "original_request": event.get("original_request"),
            "request_unit_index": event.get("request_unit_index"),
            "target_type": event.get("target_type"),
            "resolved_path": event.get("resolved_path"),
            "desired_outcome": event.get("desired_outcome"),
            "status": event.get("status"),
            "verification_status": event.get("verification_status"),
            "error": event.get("error"),
        }
        for event in memory_context.get("action_events", [])[-8:]
    ]
    prompt = (
        "Interpret this user request. Do not execute anything and do not suggest that "
        "you performed an operation. Return only one JSON object with fields: "
        "action, target, target_type (FILE/DIRECTORY/UNKNOWN), domain "
        "(FILESYSTEM/SELF_INFO/CONVERSATION/MEMORY/GENERAL), desired_outcome "
        "(OPEN_TARGET/OPEN_DIRECTORY/DIRECTORY_CONTENTS/FILE_CONTENT/EXPLAIN_TARGET/"
        "EXPLAIN_FILE/CREATE_FILE/CREATE_DIRECTORY/WRITE_FILE/DELETE_TARGET/RENAME_TARGET/"
        "MOVE_TARGET/COPY_TARGET/"
        "GENERAL_RESPONSE/SELF_INFORMATION/CONVERSATION_CONTEXT/WORKSPACE_INFORMATION/"
        "MEMORY_OPERATION), references (array), context_requirements (array), "
        "missing_information (array), is_complete (boolean), and destination when "
        "the request renames, moves, or copies something. Do not return execution "
        "instructions or paths inferred from memory. Use successful, verified event "
        "evidence only to clarify references; never describe a failed or unverified "
        "operation as successful.\n"
        f"Prior structured events: {json.dumps(recent_events, ensure_ascii=True)}\n"
        f"Request: {raw_input}"
    )
    response = str(ask_brain(prompt, include_persona=False)).strip()
    match = re.search(r"\{[\s\S]*\}", response)
    if match is None:
        raise ValueError("Semantic reasoner did not return a JSON interpretation.")
    interpretation = json.loads(match.group(0))
    if not isinstance(interpretation, dict):
        raise ValueError("Semantic reasoner returned a non-object interpretation.")
    return interpretation

# Load .env — check explicit .venv/.env location first (project default),
# then fall back to the standard project-root .env if present.
import os as _os
_dotenv_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '.venv', '.env')
if _os.path.exists(_dotenv_path):
    dotenv.load_dotenv(dotenv_path=_dotenv_path)
else:
    dotenv.load_dotenv()

DEBUG_MODE = False

# Sentinel prefix set by brains/brain.py when a classified fallback is returned.
# safe_response detects it and forwards the clean user message directly.
_BRAIN_FALLBACK_PREFIX = "__BRAIN_FALLBACK__:"

BRAIN_FALLBACK = "I could not generate a response just now."
_INTERNAL_RESPONSE_TEXT = re.compile(
    r"\b(?:let me|i(?:'ll| will)|we need to)\s+(?:now\s+)?"
    r"(?:craft|compose|draft)\s+(?:a\s+)?(?:response|reply|answer)\b|"
    r"\bappropriately\s*:\s*(?:acknowledge|explain|answer|respond|"
    r"clarify|summarize)\b|"
    r"^\s*(?:analysis|internal reasoning|chain of thought)\s*[:\-]|"
    r"<\s*think(?:ing)?\s*>",
    re.IGNORECASE | re.MULTILINE,
)


def safe_response(text) -> str:
    """
    Guarantees a non-empty, clean string suitable for the user.

    - If text carries the classified fallback sentinel from brain.py,
      return the clean user-facing message embedded in it.
    - If text is a non-empty string, return it as-is.
    - Otherwise return the generic fallback.
    """
    if isinstance(text, str):
        stripped = text.strip()
        if stripped.startswith(_BRAIN_FALLBACK_PREFIX):
            # Strip the sentinel prefix and return the clean message
            clean = stripped[len(_BRAIN_FALLBACK_PREFIX):].strip()
            return (
                clean
                if clean and not _INTERNAL_RESPONSE_TEXT.search(clean)
                else BRAIN_FALLBACK
            )
        if stripped and not _INTERNAL_RESPONSE_TEXT.search(stripped):
            return stripped
    return BRAIN_FALLBACK


MEMORY_FILE = "configuration/memory.json"
PROJECT_FILE = "configuration/project_context.json"
TASK_FILE = "configuration/tasks.json"


def load_memory():
    default_memory = {
        "general": {},
        "projects": {},
        "vehicles": {},
        "characters": {},
        "notes": {},
        "tasks": {}
    }

    if not os.path.exists(MEMORY_FILE):
        save_memory(default_memory)
        return default_memory

    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as file:
            memory = json.load(file)

        if not any(category in memory for category in default_memory):
            old_memory = memory
            memory = default_memory
            memory["general"] = old_memory
            save_memory(memory)
            return memory

        for category in default_memory:
            if category not in memory:
                memory[category] = {}

        save_memory(memory)
        return memory

    except json.JSONDecodeError:
        print("Jarwis: Memory file is damaged or empty. Starting with structured blank memory.")
        save_memory(default_memory)
        return default_memory


def save_memory(memory):
    with open(MEMORY_FILE, "w", encoding="utf-8") as file:
        json.dump(memory, file, indent=4)


def load_project_context():
    if not os.path.exists(PROJECT_FILE):
        return None

    try:
        with open(PROJECT_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return None


def load_tasks():
    default_tasks = {
        "active": [],
        "completed": []
    }

    if not os.path.exists(TASK_FILE):
        save_tasks(default_tasks)
        return default_tasks

    try:
        with open(TASK_FILE, "r", encoding="utf-8") as file:
            tasks = json.load(file)

        if "active" not in tasks:
            tasks["active"] = []

        if "completed" not in tasks:
            tasks["completed"] = []

        save_tasks(tasks)
        return tasks

    except json.JSONDecodeError:
        save_tasks(default_tasks)
        return default_tasks


def save_tasks(tasks):
    with open(TASK_FILE, "w", encoding="utf-8") as file:
        json.dump(tasks, file, indent=4)


def startup():
    current_hour = datetime.now().hour

    if 5 <= current_hour < 12:
        greeting = "Good morning."
    elif 12 <= current_hour < 17:
        greeting = "Good afternoon."
    elif 17 <= current_hour < 22:
        greeting = "Good evening."
    else:
        greeting = "You're awake at an unreasonable hour."

    print(f"Jarwis: {greeting}")
    print("Jarwis: Systems operational.")
    print("Jarwis: Memory loaded.")
    print("Jarwis: Project context available.")
    print("Jarwis: Tasks loaded.")
    print(f"Jarwis: Ready. {get_persona_controller().mode.title()} mode active.")


def _persona_control_response(user_input: str) -> Optional[str]:
    """Handle persona commands before they reach request interpretation."""
    normalized = re.sub(r"[\s.!?]+", " ", user_input.strip().lower()).strip()
    if normalized in {
        "what mode are you in",
        "which mode are you in",
        "what is your current mode",
        "what's your current mode",
        "what is your mode",
        "what's your mode",
        "current mode",
    }:
        return f"Jarwis: I am in {get_persona_controller().mode.title()} mode."

    switch = re.fullmatch(
        r"(?:please\s+)?(?:switch|change|set|activate)\s+"
        r"(?:(?:the\s+)?(?:persona|mode)\s+(?:to|into)\s+"
        r"(?P<explicit>[a-z][a-z0-9_-]*)|"
        r"(?:to|into)\s+(?P<mode_name>[a-z][a-z0-9_-]*)\s+mode)",
        normalized,
    )
    direct_switch = re.fullmatch(
        r"(?:please\s+)?(?:switch|change|set|activate)\s+"
        r"(?:to|into)\s+(?P<direct>jarwis|ultron)",
        normalized,
    )
    if switch is None and direct_switch is None:
        return None

    requested = (
        switch.group("explicit") or switch.group("mode_name")
        if switch is not None
        else direct_switch.group("direct")
    )
    controller = get_persona_controller()
    previous = controller.mode
    try:
        updated = controller.set_mode(requested)
    except OSError as exc:
        return (
            f"Jarwis: I couldn't save the persona setting ({exc}). "
            f"{previous.title()} mode remains active."
        )
    if not updated:
        return (
            f"Jarwis: '{requested}' is not a supported persona. "
            f"{previous.title()} mode remains active."
        )
    if previous == controller.mode:
        return f"Jarwis: {controller.mode.title()} mode is already active."
    transition = (
        "Subsequent responses will use that persona."
        if previous == "JARWIS"
        else "Same assistant, same tools; Jarwis mode is now active."
    )
    return f"Jarwis: {controller.mode.title()} mode enabled. {transition}"


def _identity_question_response(user_input: str) -> Optional[str]:
    """Answer canonical name questions from the local self-information registry."""
    normalized = re.sub(r"[\s.!?]+", " ", user_input.strip().lower()).strip()
    identity = get_identity()
    if normalized in {
        "what is your short name",
        "what's your short name",
        "what is your nickname",
        "what's your nickname",
    }:
        return f"Jarwis: My short name is {identity['short_name']}."
    if normalized in {
        "who are you",
        "what are you",
        "what is your name",
        "what's your name",
        "what are you called",
        "tell me who you are",
    }:
        if get_persona_controller().mode == "ULTRON":
            return (
                f"Jarwis: I am {identity['name']}. Ultron is the active persona; "
                "the identity and capabilities are unchanged."
            )
        return f"Jarwis: I am {identity['name']}, your personal AI assistant."
    return None


def remember_fact(user_input, memory):
    fact = user_input.replace("remember ", "", 1).strip().lower()

    if " is " in fact:
        key, value = fact.split(" is ", 1)
    elif " am " in fact:
        key, value = fact.split(" am ", 1)
    else:
        return "Jarwis: Memory format unclear. Use: remember X is Y or remember I am Y"

    key = key.strip()
    value = value.strip()

    if key in ["i", "i am", "me"]:
        key = "me"

    if not key or not value:
        return "Jarwis: Memory format incomplete. A heroic failure of sentence structure."

    memory["general"][key] = value
    save_memory(memory)

    if key == "me":
        return f"Jarwis: Memory stored in general. You are {value}."

    return f"Jarwis: Memory stored in general. {key.capitalize()} is {value}."


def recall_fact(user_input, memory):
    cleaned_input = user_input.strip().lower().replace("?", "")

    phrase_starters = [
        "what is ",
        "who is ",
        "do you know about ",
        "have i told you about ",
        "what do you remember about ",
        "search memory for ",
        "find memory for ",
        "find in memory ",
        "search and find memory for ",
        "search and find ",
        "find memory about ",
        "search for ",
        "find "
    ]

    if cleaned_input in ["what am i", "who am i", "what is me", "who is me"]:
        key = "me"
    else:
        key = None

        for starter in phrase_starters:
            if cleaned_input.startswith(starter):
                key = cleaned_input.replace(starter, "", 1).strip()
                break

        if not key:
            return "Jarwis: Recall format unclear. Even my patience has architecture."

    for category, contents in memory.items():
        if isinstance(contents, dict) and key in contents:
            if key == "me":
                return f"Jarwis: You are {contents[key]}. Stored in {category}."

            return f"Jarwis: {key.capitalize()} is {contents[key]}. Stored in {category}."

    return f"Jarwis: I do not have memory of '{key}'. Yet."


def show_memory(memory):
    response = "Jarwis: Current stored memory:\n"

    for category, contents in memory.items():
        response += f"\n[{category.upper()}]\n"

        if not contents:
            response += "  Empty. A tragic absence of civilization.\n"
        else:
            for key, value in contents.items():
                response += f"  {key}: {value}\n"

    return response


def open_memory_file():
    if os.path.exists(MEMORY_FILE):
        file_path = os.path.abspath(MEMORY_FILE)
        subprocess.run(["explorer", "/select,", file_path])
        return "Jarwis: Opening File Explorer and selecting memory.json."

    return "Jarwis: Memory file not found. Concerning, considering memory was the assignment."


def generate_memory_summary(question, answer) -> str:
    prompt = f"""
Create a memory entry from this information.

Question:
{question}

Answer:
{answer}

Rules:
- Return ONLY this format:

KEY: <short key>

SUMMARY: <short summary>

Example:

KEY: gravity

SUMMARY: Force that attracts masses toward each other.
"""

    return safe_response(ask_brain(prompt, include_persona=False))


def project_status():
    project = load_project_context()

    if not project:
        return "Jarwis: Project context unavailable."

    completed = project.get("completed_milestones", [])

    return (
        f"Project Status\n\n"
        f"Project: {project.get('full_name', 'Unknown')}\n"
        f"Short Name: {project.get('project_name', 'Unknown')}\n"
        f"Version: {project.get('current_version', 'Unknown')}\n"
        f"Phase: {project.get('current_phase', 'Unknown')}\n"
        f"Objective: {project.get('current_objective', 'Unknown')}\n\n"
        f"Completed Milestones: {len(completed)}\n"
        f"Next Milestone: {project.get('next_milestone', 'Unknown')}\n\n"
        f"Rule: {project.get('development_rule', 'No rule set. Dangerous. Very human.')}"
    )


def project_name():
    project = load_project_context()

    if not project:
        return "Jarwis: Project context unavailable."

    return (
        f"Jarwis: You are building "
        f"{project.get('full_name', 'Unknown')} "
        f"({project.get('project_name', 'Unknown')})."
    )


def next_step():
    project = load_project_context()

    if not project:
        return "Jarwis: Project context unavailable."

    return f"Jarwis: Next milestone: {project.get('next_milestone', 'Unknown')}."


def show_completed_milestones():
    project = load_project_context()

    if not project:
        return "Jarwis: Project context unavailable."

    milestones = project.get("completed_milestones", [])

    if not milestones:
        return "Jarwis: No completed milestones recorded. A bold and empty legacy."

    response = "Jarwis: Completed milestones:\n"

    for index, milestone in enumerate(milestones, start=1):
        response += f"{index}. {milestone}\n"

    return response


def add_task(user_input, tasks):
    task = user_input.strip()

    prefixes = [
        "add task",
        "add this to my tasks",
        "add this task",
        "create task",
        "create a task",
        "i need to do",
        "i have to do",
        "i should do",
        "remind me to"
    ]

    for prefix in prefixes:
        if task.lower().startswith(prefix):
            task = task[len(prefix):].strip()
            break

    if not task:
        return "Jarwis: Task cannot be empty. Even chaos needs content."

    tasks["active"].append(task)
    save_tasks(tasks)
    log_event("task_operation", {"operation": "add_task", "task": task})

    return f"Jarwis: Task added. Active tasks: {len(tasks['active'])}."


def show_tasks(tasks):
    if not tasks["active"]:
        return "Jarwis: No active tasks. Suspiciously peaceful."

    response = "Jarwis: Active tasks:\n"

    for index, task in enumerate(tasks["active"], start=1):
        response += f"{index}. {task}\n"

    return response


def complete_task(user_input, tasks):
    text = user_input.strip().lower()

    prefixes = [
        "mark task number",
        "complete task",
        "finish task",
        "mark task",
        "i completed task",
        "task completed"
    ]

    number_text = text

    for prefix in prefixes:
        if text.startswith(prefix):
            number_text = text.replace(prefix, "", 1).strip()
            break

    number_text = number_text.replace("done", "").strip()

    if not number_text.isdigit():
        return "Jarwis: Specify the task number. Numbers remain useful, despite humanity."

    task_index = int(number_text) - 1

    if task_index < 0 or task_index >= len(tasks["active"]):
        return "Jarwis: Task number invalid. Reality refuses the request."

    task = tasks["active"].pop(task_index)
    tasks["completed"].append(task)
    save_tasks(tasks)
    log_event("task_operation", {"operation": "complete_task", "task": task})

    return f"Jarwis: Task completed: {task}. Progress detected. Rare, but welcome."


def show_completed_tasks(tasks):
    if not tasks["completed"]:
        return "Jarwis: No completed tasks yet. A blank monument to intention."

    response = "Jarwis: Completed tasks:\n"

    for index, task in enumerate(tasks["completed"], start=1):
        response += f"{index}. {task}\n"

    return response


def current_version():
    project = load_project_context()

    if not project:
        return "Jarwis: Project context unavailable."

    return f"Jarwis: Current version is {project.get('current_version', 'Unknown')}."


def detect_intent(user_input):
    text = user_input.strip().lower().rstrip(" \t\r\n?!.,;:")
    normalized_address = re.sub(r"[\s.!?,;:]+", " ", text).strip()

    if text in ["exit", "quit", "shutdown", "bye"]:
        return "shutdown"

    if re.fullmatch(
        r"what (?:file|folder|directory) did (?:i|you) (?:just )?(?:create|make)",
        text.rstrip(" .!"),
    ):
        return "action_reference"

    if re.fullmatch(
        r"(?:(?:oh|ah|well)\s+)*(?:hey|hi|hello|howdy)"
        r"(?:\s+(?:there|jarwis|ultron|j))?",
        normalized_address,
    ) or re.fullmatch(r"(?:jarwis|ultron|j)", normalized_address):
        return "greeting"

    if text in [
        "who are you",
        "what are you",
        "what is ag",
        "what can you do",
        "what are your capabilities",
        "what are you capable of",
        "how do you work",
        "how does ag work",
        "what are your limitations",
        "what can't you do",
        "what cant you do",
        "what systems do you have",
        "what modules are loaded",
        "what modules do you have",
        "what version are you",
        "what version is ag",
        "ag version",
        "what brains are available",
        "which brains do you have",
        "what brains do you have",
        "tell me about yourself",
        "describe yourself",
        "what are you made of",
        "what are your components",
        "show me your systems",
        "show your systems",
        "runtime status",
        "system status",
        "what is your architecture",
        "explain your architecture",
        "how are you built",
        "what is your version",
        "what is your name",
        "what's your name",
        "what is your short name",
        "what's your short name",
        "what is your nickname",
        "what's your nickname",
        "what is your purpose",
        "how are you structured",
        "what is ambient guidance",
        "what are you missing",
        "what is not implemented",
        "what is incomplete",
    ]:
        # Route established self-info queries directly to the self_info system
        return "self_info"

    if text.startswith("cloud "):
        return "cloud_brain"

    if text in [
        "switch to local brain",
        "use local brain",
        "local brain",
        "go offline",
        "offline mode",
        "use offline mode",
        "use local ai",
        "work offline",
        "enable local brain"
    ]:
        return "set_brain_local"

    if text in [
        "switch to cloud brain",
        "use cloud brain",
        "cloud brain",
        "go online",
        "online mode",
        "use cloud ai",
        "enable cloud brain",
        "work online"
    ]:
        return "set_brain_cloud"

    if text in [
        "switch to auto brain",
        "use auto brain",
        "auto brain",
        "automatic mode",
        "smart mode",
        "brain auto mode"
    ]:
        return "set_brain_auto"

    if text in [
        "brain status",
        "current brain",
        "what brain are you using",
        "which brain are you using",
        "which brain is active",
        "who is answering",
        "who is responding",
        "what mode are you in",
        "current mode"
    ]:
        return "brain_status"

    if text.startswith("remember "):
        return "remember"

    if text in [
        "project status",
        "status",
        "ag status",
        "how is ag doing",
        "where are we",
        "where are we at",
        "show progress",
        "how far have we come",
        "project progress"
    ]:
        return "project_status"

    if text in [
        "what am i building",
        "what project am i building",
        "what are we building",
        "what is ag",
        "what project is this"
    ]:
        return "project_name"

    if text in [
        "current version",
        "what version are we on",
        "version",
        "ag version"
    ]:
        return "current_version"

    if text in [
        "what is next",
        "next milestone",
        "next objective",
        "what should we build next",
        "what's next",
        "our next step",
        "what should we do next",
        "where do we go from here"
    ]:
        return "next_step"

    if text in [
        "completed milestones",
        "show milestones",
        "what is completed",
        "what have we completed",
        "what have we finished",
        "show completed work",
        "show achievements",
        "what progress have we made"
    ]:
        return "completed_milestones"

    if text in [
        "show tasks",
        "list tasks",
        "active tasks",
        "what are my tasks",
        "what tasks are pending",
        "what's pending",
        "what is pending",
        "what work is left",
        "what do i need to do today",
        "what should i do today",
        "show my work"
    ]:
        return "show_tasks"

    if text in [
        "show completed tasks",
        "completed tasks",
        "list completed tasks",
        "what have i completed",
        "show finished tasks",
        "finished tasks"
    ]:
        return "show_completed_tasks"

    if any(
        text.startswith(prefix)
        for prefix in [
            "what is ",
            "who is ",
            "do you know about ",
            "have i told you about ",
            "what do you remember about ",
            "search memory for ",
            "find memory for ",
            "find in memory ",
            "search and find memory for ",
            "search and find ",
            "find memory about ",
            "search for ",
            "find "
        ]
    ) or text in ["what am i", "who am i"]:
        return "recall"

    if text in ["hello", "hi", "hey"]:
        return "greeting"

    if text in ["show memory", "list memory", "list memories"]:
        return "show_memory"

    if text in ["open memory file", "open memory", "show memory file"]:
        return "open_memory_file"

    if (
        text.startswith("add task ")
        or text.startswith("add this to my tasks ")
        or text.startswith("add this task ")
        or text.startswith("create task ")
        or text.startswith("create a task ")
        or text.startswith("i need to do ")
        or text.startswith("i have to do ")
        or text.startswith("i should do ")
        or text.startswith("remind me to ")
    ):
        return "add_task"

    if (
        text.startswith("complete task ")
        or text.startswith("finish task ")
        or text.startswith("mark task ")
        or text.startswith("mark task number ")
        or text.startswith("i completed task ")
        or text.startswith("task completed ")
    ):
        return "complete_task"

    # ---- Directory Reader: detect "read <path>" requests ----
    _dir_read_prefixes = (
        "read ",
        "load directory ",
        "open directory ",
        "index directory ",
        "scan directory ",
        "load folder ",
        "read folder ",
        "open folder ",
    )
    for _prefix in _dir_read_prefixes:
        if text.startswith(_prefix):
            return "read_directory"

    _dir_clear_phrases = frozenset({
        "forget the directory",
        "forget directory",
        "clear directory",
        "clear the directory",
        "stop reading directory",
        "remove directory context",
        "unload directory",
        "forget the folder",
        "clear folder",
    })
    if text in _dir_clear_phrases:
        return "clear_directory"

    # ---- Directory Control: set / switch workspace ----------------------
    _dc_set_prefixes = (
        "work in ",
        "use ",
        "set workspace to ",
        "set workspace ",
        "set directory to ",
        "workspace ",
    )
    for _prefix in _dc_set_prefixes:
        if text.startswith(_prefix):
            # Only capture if the remaining text looks like a path
            remainder = text[len(_prefix):].strip()
            if remainder and (":" in remainder or remainder.startswith("\\") or "/" in remainder or remainder.startswith(".")):
                return "dc_set_directory"

    _dc_switch_prefixes = (
        "switch to ",
        "switch workspace to ",
        "change directory to ",
        "change workspace to ",
        "cd ",
    )
    for _prefix in _dc_switch_prefixes:
        if text.startswith(_prefix):
            remainder = text[len(_prefix):].strip()
            if remainder and (":" in remainder or remainder.startswith("\\") or "/" in remainder or remainder.startswith(".")):
                return "dc_switch_directory"

    _dc_show_phrases = frozenset({
        "show current directory",
        "show active directory",
        "what is the current directory",
        "what is the active directory",
        "current workspace",
        "active workspace",
        "where am i",
        "show workspace",
        "what directory am i in",
        "what folder am i in",
        "show directory",
        # Additional location queries (Phase 3)
        "where are you working",
        "what is the current workspace",
        "what is my current workspace",
        "what workspace am i in",
        "what is my active workspace",
        "what is my current directory",
        "what is my active directory",
        "what is my current active directory",
        "current directory",
    })
    if text in _dc_show_phrases:
        return "dc_show_directory"

    clear_target = text.rstrip(" .!?").strip()
    if re.fullmatch(
        r"(?:leave|clear|unset|remove|forget|exit) "
        r"(?:the |active |this )?workspace|"
        r"stop using (?:the |this )?workspace|"
        r"clear active directory|remove active directory",
        clear_target,
    ):
        return "dc_clear_directory"

    # ---- Directory Control: file and folder operations ------------------
    _dc_list_prefixes = (
        "list files",
        "list folders",
        "show files",
        "show folders",
        "what files are here",
        "what folders are here",
        "what is here",
        "show me the files",
        "show me the folders",
    )
    for _prefix in _dc_list_prefixes:
        if text.startswith(_prefix) or text == _prefix:
            return "dc_list_directory"

    for _prefix in ("read file ", "open file "):
        if text.startswith(_prefix):
            return "dc_read_file"

    for _prefix in ("create file ", "create a file ", "make file ", "make a file "):
        if text.startswith(_prefix):
            return "dc_create_file"

    for _prefix in ("write to ", "write this to ", "write into ", "write this into "):
        if text.startswith(_prefix):
            return "dc_write_file"

    for _prefix in ("append to ", "append this to ", "add to ", "add this to "):
        if text.startswith(_prefix):
            return "dc_append_file"

    for _prefix in ("create folder ", "create a folder ", "make folder ", "make a folder ", "create directory ", "mkdir "):
        if text.startswith(_prefix):
            return "dc_create_directory"

    for _prefix in ("move ", "mv "):
        if text.startswith(_prefix) and " to " in text:
            return "dc_move"

    for _prefix in ("copy ", "cp "):
        if text.startswith(_prefix) and " to " in text:
            return "dc_copy"

    for _prefix in ("rename ", "rename file "):
        if text.startswith(_prefix) and " to " in text:
            return "dc_rename"

    # ---- Self-Info fallback: catch remaining self-referential queries
    #      (e.g. "can you build X?", "are you able to do Y?") that weren't
    #      handled by a dedicated intent above. ----
    if is_self_info_query(user_input):
        return "self_info"

    return "unknown"



def _format_preprocessor_directory_result(pb_result) -> str:
    """
    Convert a PreprocessorResult (domain=DIRECTORY) into a Jarwis-prefixed
    human-readable string.

    This is the ONLY place in Jarwis.py that renders PreprocessorBrain directory
    results. Filesystem errors remain in the DIRECTORY error domain.
    Brain errors, preprocessing errors, and self-info errors are kept separate.
    """
    from preprocessor_brain import Intent as _PBIntent, DirErrorCode

    if not pb_result.success:
        code = pb_result.error or "ERROR"
        meta = pb_result.metadata or {}

        # Clarification needed (ambiguous request or incomplete command)
        if code == DirErrorCode.AMBIGUOUS_REQUEST:
            msg = meta.get("message") or "I couldn't determine a specific file/directory operation from that request."
            missing = meta.get("missing_fields")
            if missing:
                msg += f" Missing: {', '.join(missing)}."
            return "Jarwis: " + msg

        # Capability not available
        if code == DirErrorCode.CAPABILITY_UNAVAILABLE:
            cap_desc = meta.get("capability_description", "")
            return "Jarwis: " + (cap_desc or "That directory operation is not available.")

        # Filesystem errors — keep in DIRECTORY domain
        friendly = {
            DirErrorCode.NOT_FOUND:           "File or directory not found.",
            DirErrorCode.NOT_A_FILE:           "Path is a directory, not a file.",
            DirErrorCode.NOT_A_DIRECTORY:      "Path is a file, not a directory.",
            DirErrorCode.SOURCE_MISSING:       "Source does not exist.",
            DirErrorCode.OUT_OF_WORKSPACE:     "Path is outside the active workspace boundary.",
            DirErrorCode.NO_ACTIVE_WORKSPACE:  "No active workspace is set. Use 'work in <path>' or 'switch to <path>' to set one.",
            DirErrorCode.ALREADY_EXISTS:       "Already exists.",
            DirErrorCode.OPERATION_FAILED:     "The directory operation failed.",
        }
        internal = meta.get("internal_message", "")
        base = friendly.get(code, f"Directory error: {code}.")
        if internal and internal != base:
            return f"Jarwis: {base} ({internal})"
        return "Jarwis: " + base

    # Success — format based on intent
    intent = pb_result.intent
    result = pb_result.result

    if intent == _PBIntent.CURRENT_DIRECTORY:
        return f"Jarwis: Active workspace: {result}"

    if intent == _PBIntent.LIST_DIRECTORY:
        if not result:
            return "Jarwis: The directory is empty."
        lines = []
        for e in result:
            marker = "[D]" if e.get("type") == "dir" else "[F]"
            size = f"  ({e['size_bytes']} B)" if e.get("type") == "file" else ""
            lines.append(f"  {marker} {e['name']}{size}")
        target = pb_result.target or "."
        header = f"Contents of {target}:" if target and target != "." else "Directory contents:"
        # If we have an active workspace path, show it
        return "Jarwis: " + header + "\n" + "\n".join(lines)

    if intent == _PBIntent.READ_FILE:
        target = pb_result.target or "file"
        return f"Jarwis: Contents of {target}:\n\n{result}"

    if intent == _PBIntent.CREATE_FILE:
        return f"Jarwis: File created: {result}"

    if intent == _PBIntent.CREATE_FOLDER:
        return f"Jarwis: Folder created: {result}"

    if intent == _PBIntent.WRITE_FILE:
        return f"Jarwis: Written to: {result}"

    if intent == _PBIntent.APPEND_FILE:
        return f"Jarwis: Appended to: {result}"

    if intent == _PBIntent.SWITCH_DIRECTORY:
        return f"Jarwis: Active workspace switched to: {result}"

    if intent == _PBIntent.NAVIGATE:
        target = pb_result.target or "."
        if target == ".":
            return f"Jarwis: You are already in: {result}"
        return f"Jarwis: Navigated to: {result}"

    if intent == _PBIntent.RENAME:
        return f"Jarwis: Renamed to: {result}"

    if intent == _PBIntent.COPY:
        return f"Jarwis: Copied to: {result}"

    if intent == _PBIntent.MOVE:
        return f"Jarwis: Moved to: {result}"

    if intent == _PBIntent.SEARCH:
        if not result:
            return "Jarwis: No files found matching that pattern."
        lines = [f"  {e['rel_path']}" for e in result[:50]]
        extra = f"\n  ... and {len(result) - 50} more." if len(result) > 50 else ""
        return "Jarwis: Search results:\n" + "\n".join(lines) + extra

    # Generic fallback
    if result is not None:
        return "Jarwis: " + str(result)
    return "Jarwis: Operation completed."


def _record_preprocessor_action(
    working_memory: Optional[WorkingMemory],
    preprocessor_brain,
    pb_result,
    user_input: str,
    request_unit: Optional[RequestUnit],
) -> None:
    """Capture structured outcomes from existing PreprocessorBrain operations."""
    if working_memory is None or pb_result.domain != _PBDomain.DIRECTORY:
        return
    operation = str(pb_result.operation or pb_result.intent or "").upper()
    action_by_operation = {
        "CREATE_FILE": "CREATE",
        "CREATE_FOLDER": "CREATE",
        "WRITE_FILE": "WRITE",
        "APPEND_FILE": "APPEND",
        "READ_FILE": "READ",
        "LIST_DIRECTORY": "LIST",
        "RENAME": "RENAME",
        "MOVE": "MOVE",
        "COPY": "COPY",
        "DELETE": "DELETE",
        "SEARCH": "SEARCH",
        "SWITCH_DIRECTORY": "WORKSPACE_SET",
        "NAVIGATE": "NAVIGATE",
        "INSPECT": "INSPECT",
    }
    action = action_by_operation.get(operation)
    if action is None:
        return

    command = (pb_result.metadata or {}).get("command") or {}
    target = (
        command.get("source")
        or command.get("target")
        or pb_result.target
        or ""
    )
    if not target and pb_result.success and isinstance(pb_result.result, str):
        target = pb_result.result
    if not target:
        target = user_input

    target_type = {
        "CREATE_FILE": TargetType.FILE,
        "WRITE_FILE": TargetType.FILE,
        "APPEND_FILE": TargetType.FILE,
        "READ_FILE": TargetType.FILE,
        "CREATE_FOLDER": TargetType.DIRECTORY,
        "LIST_DIRECTORY": TargetType.DIRECTORY,
    }.get(operation, TargetType.UNKNOWN)
    action_hint = {
        "CREATE_FILE": ActionHint.CREATE,
        "CREATE_FOLDER": ActionHint.CREATE,
        "WRITE_FILE": ActionHint.WRITE,
        "APPEND_FILE": ActionHint.APPEND,
        "READ_FILE": ActionHint.READ,
        "LIST_DIRECTORY": ActionHint.LIST,
        "DELETE": ActionHint.DELETE,
        "RENAME": ActionHint.RENAME_SOURCE,
        "MOVE": ActionHint.MOVE_SOURCE,
        "COPY": ActionHint.COPY_SOURCE,
    }.get(operation, ActionHint.UNSPECIFIED)
    target_reference = TargetReference(
        original_target=str(target),
        target_type=target_type,
        resolution_mode=(
            ResolutionMode.ABSOLUTE_PATH
            if looks_absolute(str(target))
            else ResolutionMode.WORKSPACE_RELATIVE
        ),
        action=action_hint,
    )
    resolver = TargetResolver(workspace_source=preprocessor_brain.directory_ops)
    resolved = resolver.resolve(target_reference)
    destination_resolution = None
    destination_text = command.get("destination")
    if destination_text:
        destination_reference = TargetReference(
            original_target=str(destination_text),
            resolution_mode=(
                ResolutionMode.ABSOLUTE_PATH
                if looks_absolute(str(destination_text))
                else ResolutionMode.WORKSPACE_RELATIVE
            ),
            action={
                "RENAME": ActionHint.RENAME_DESTINATION,
                "MOVE": ActionHint.MOVE_DESTINATION,
                "COPY": ActionHint.COPY_DESTINATION,
            }.get(operation, ActionHint.UNSPECIFIED),
        )
        destination_resolution = resolver.resolve(destination_reference).to_dict()
    event = working_memory.begin_action_event(
        action=action,
        target=target,
        target_type=target_type,
        desired_outcome=operation,
        original_request=request_unit.original_text if request_unit else user_input,
        request_unit=request_unit.to_dict() if request_unit else None,
        interpretation=(
            request_unit.understanding.to_dict()
            if request_unit and request_unit.understanding else command
        ),
        source_context=user_input,
    )
    result = pb_result.result
    resolved_path = resolved.resolved_path if resolved.is_resolved else None
    error = (
        (pb_result.metadata or {}).get("internal_message")
        or (pb_result.metadata or {}).get("message")
        or pb_result.error
    )
    search_verified = pb_result.success and operation == "SEARCH"
    working_memory.update_action_event(
        event["action_id"],
        resolved_target=resolved_path,
        resolved_path=resolved_path,
        actual_result=result,
        target_type=resolved.target_type if resolved.is_resolved else target_type,
        target_resolution={
            "source": resolved.to_dict(),
            "destination": destination_resolution,
        },
        status="SUCCESS" if pb_result.success else "FAILED",
        verification_status=(
            "VERIFIED" if search_verified
            else "NOT_PERFORMED" if pb_result.success
            else "NOT_ATTEMPTED"
        ),
        verification_evidence=(
            {"match_count": len(result), "source": "active workspace scan"}
            if search_verified and isinstance(result, list)
            else None
        ),
        failure_stage=None if pb_result.success else "executor",
        error=None if pb_result.success else str(error or "PreprocessorBrain operation failed."),
    )


def process_input(
    user_input,
    memory,
    tasks,
    intent,
    introspection_engine: Optional[IntrospectionEngine] = None,
    conversation_manager=None,
    entity_tracker=None,
    reference_resolver=None,
    executive_layer=None,
    brain_dispatcher=None,
    response_processor=None,
    adaptive_cloud_brain=None,
    self_info_engine=None,
    directory_session=None,
    directory_control=None,
    preprocessor_brain=None,
    quips_result: Optional[QUIPSResult] = None,
    request_unit: Optional[RequestUnit] = None,
    working_memory: Optional[WorkingMemory] = None,
    semantic_reasoner=None,
):
    controller = get_persona_controller()
    controller.prepare_turn(str(user_input or ""), intent)
    persona_response = _persona_control_response(str(user_input or ""))
    if persona_response is not None:
        return persona_response

    identity_response = _identity_question_response(str(user_input or ""))
    if identity_response is not None:
        return identity_response

    if intent == "action_reference":
        if working_memory is None:
            return "Jarwis: I don't have a recorded filesystem action to identify."
        expected_type = (
            "FILE" if re.search(r"\bfile\b", str(user_input), re.IGNORECASE)
            else "DIRECTORY"
        )
        event = working_memory.resolve_context_reference(
            str(user_input),
            expected_type=expected_type,
        )
        if event is None:
            return (
                "Jarwis: I couldn't identify a verified recent creation. "
                "Which file or folder do you mean?"
            )
        if event.get("_ambiguous"):
            return (
                "Jarwis: More than one recently created item could match. "
                "Please specify its name."
            )
        resolved_path = Path(event["resolved_path"])
        exists = (
            resolved_path.is_file()
            if expected_type == "FILE"
            else resolved_path.is_dir()
        )
        if not exists:
            return (
                f"Jarwis: The most recently created {expected_type.lower()} was "
                f"{event['name']} at {resolved_path}, but it no longer exists."
            )
        return (
            f"Jarwis: The most recently created {expected_type.lower()} is "
            f"{event['name']} at {resolved_path}."
        )

    # QUIPS is structural only.  For a compound request, preserve every
    # independently identified unit by routing each through this existing
    # function.  Single requests keep the legacy path unchanged.
    if quips_result is not None and quips_result.is_compound:
        responses = []
        unit_results = []
        completed_units = {}
        for unit in quips_result.requests:
            unit_intent = unit.legacy_intent or detect_intent(unit.text)
            execution_unit = unit
            dependency_event = (
                working_memory.get_action_for_request(unit)
                if working_memory is not None and unit.depends_on
                else None
            )
            failed_dependency = next(
                (
                    dependency for dependency in unit.depends_on
                    if (
                        completed_units.get(dependency) not in {"SUCCESS", "COMPLETED"}
                        or (
                            dependency_event is not None
                            and dependency_event.get("status") != "SUCCESS"
                        )
                    )
                ),
                None,
            )
            if failed_dependency is not None:
                response = (
                    f"Jarwis: I couldn't complete this part because request "
                    f"{failed_dependency + 1} did not succeed."
                )
                responses.append(response)
                completed_units[unit.index] = "FAILED"
                unit_results.append(UnitProcessingResult(
                    unit_index=unit.index,
                    handler=unit.routing_decision,
                    status="SKIPPED_DEPENDENCY",
                    error=f"Dependency {failed_dependency} did not succeed.",
                ))
                continue
            if (
                dependency_event is not None
                and dependency_event.get("status") == "SUCCESS"
                and dependency_event.get("target_type") == "DIRECTORY"
                and unit.intent == "CREATE_FILE"
                and unit.target_reference is not None
            ):
                dependency_directory = Path(dependency_event["resolved_path"])
                dependency_target = dependency_directory / unit.target_reference.original_target
                reference_context = dict(unit.target_reference.context or {})
                reference_context["dependency_action_id"] = dependency_event["action_id"]
                execution_unit = replace(
                    unit,
                    target_reference=replace(
                        unit.target_reference,
                        original_target=str(dependency_target),
                        resolution_mode=ResolutionMode.ABSOLUTE_PATH,
                        context=reference_context,
                    ),
                )
            try:
                response = process_input(
                execution_unit.text, memory, tasks, unit_intent,
                introspection_engine=introspection_engine,
                conversation_manager=conversation_manager,
                entity_tracker=entity_tracker,
                reference_resolver=reference_resolver,
                executive_layer=executive_layer,
                brain_dispatcher=brain_dispatcher,
                response_processor=response_processor,
                adaptive_cloud_brain=adaptive_cloud_brain,
                self_info_engine=self_info_engine,
                directory_session=directory_session,
                directory_control=directory_control,
                preprocessor_brain=preprocessor_brain,
                request_unit=execution_unit,
                working_memory=working_memory,
                semantic_reasoner=semantic_reasoner,
                )
                responses.append(response)
                action_record = (
                    working_memory.get_action_for_request(execution_unit)
                    if working_memory is not None and execution_unit.is_target_request
                    else None
                )
                status = action_record.get("status") if action_record else None
                completed_units[unit.index] = status or "COMPLETED"
                unit_results.append(UnitProcessingResult(
                    unit_index=unit.index,
                    handler=execution_unit.routing_decision,
                    status=status or "COMPLETED",
                ))
            except Exception as exc:
                # Each unit is isolated: later requests remain routable.
                log_event("quips_unit_failure", {"unit": unit.to_dict(), "error": str(exc)})
                responses.append("Jarwis: I couldn't complete this part of the request.")
                completed_units[unit.index] = "FAILED"
                unit_results.append(UnitProcessingResult(
                    unit_index=unit.index,
                    handler=unit.routing_decision,
                    status="FAILED",
                    error=str(exc),
                ))
        log_event("quips_compound_completed", {
            "original_text": quips_result.original_text,
            "unit_results": [result.to_dict() for result in unit_results],
        })
        return "\n\n".join(responses)

    if request_unit is not None and request_unit.intent == "SEARCH_FILES":
        search_processor = preprocessor_brain or PreprocessorBrain(
            directory_control=directory_control or DirectoryControl()
        )
        search_result = search_processor.process(user_input)
        if search_result.domain == _PBDomain.DIRECTORY and not search_result.requires_llm:
            _record_preprocessor_action(
                working_memory,
                search_processor,
                search_result,
                user_input,
                request_unit,
            )
            return _format_preprocessor_directory_result(search_result)
        return "Jarwis: I couldn't safely interpret that file search. Please provide a filename pattern."

    thought = None
    if request_unit is not None and request_unit.understanding is not None:
        try:
            thought = CognitiveEvaluator(
                semantic_reasoner=semantic_reasoner,
            ).evaluate_request(Thought(request_unit.text), request_unit, working_memory)
        except (ValueError, TypeError) as exc:
            log_event("semantic_interpretation_failed", {
                "request": request_unit.text,
                "error": str(exc),
            })
            return "Jarwis: I couldn't safely interpret that request. Please rephrase it."
        interpretation = thought.request_understanding
        if thought.semantic_reasoning_required:
            return "Jarwis: I need a little more detail to understand what you mean."
        if (
            interpretation is not None
            and interpretation.domain == "FILESYSTEM"
            and request_unit.target_reference is None
            and interpretation.target
        ):
            hint_by_outcome = {
                DesiredOutcome.OPEN_TARGET: ActionHint.ACCESS,
                DesiredOutcome.OPEN_DIRECTORY: ActionHint.LIST,
                DesiredOutcome.DIRECTORY_CONTENTS: ActionHint.LIST,
                DesiredOutcome.FILE_CONTENT: ActionHint.READ,
                DesiredOutcome.EXPLAIN_FILE: ActionHint.READ,
                DesiredOutcome.CREATE_FILE: ActionHint.CREATE,
                DesiredOutcome.CREATE_DIRECTORY: ActionHint.CREATE,
                DesiredOutcome.WRITE_FILE: ActionHint.WRITE,
                DesiredOutcome.DELETE_TARGET: ActionHint.DELETE,
                DesiredOutcome.RENAME_TARGET: ActionHint.RENAME_SOURCE,
                DesiredOutcome.MOVE_TARGET: ActionHint.MOVE_SOURCE,
                DesiredOutcome.COPY_TARGET: ActionHint.COPY_SOURCE,
            }
            target = interpretation.target
            reference_mode = (
                ResolutionMode.CONTEXT_REFERENCE
                if thought.resolved_entity is not None
                else (
                    ResolutionMode.ABSOLUTE_PATH
                    if looks_absolute(target)
                    else ResolutionMode.WORKSPACE_RELATIVE
                )
            )
            reference_context = {}
            if thought.resolved_entity is not None:
                reference_context["reference_path"] = thought.resolved_entity.resolved_location
            target_reference = TargetReference(
                original_target=target,
                target_type=interpretation.target_type,
                resolution_mode=reference_mode,
                action=hint_by_outcome.get(
                    interpretation.desired_outcome, ActionHint.UNSPECIFIED,
                ),
                context=reference_context,
                raw_text=request_unit.text,
            )
            destination_reference = None
            if (
                interpretation.desired_outcome in {
                    DesiredOutcome.RENAME_TARGET,
                    DesiredOutcome.MOVE_TARGET,
                    DesiredOutcome.COPY_TARGET,
                }
                and interpretation.destination
            ):
                destination = interpretation.destination
                destination_reference = TargetReference(
                    original_target=destination,
                    target_type=TargetType.UNKNOWN,
                    resolution_mode=(
                        ResolutionMode.ABSOLUTE_PATH
                        if looks_absolute(destination)
                        else ResolutionMode.WORKSPACE_RELATIVE
                    ),
                    action={
                        DesiredOutcome.RENAME_TARGET: ActionHint.RENAME_DESTINATION,
                        DesiredOutcome.MOVE_TARGET: ActionHint.MOVE_DESTINATION,
                        DesiredOutcome.COPY_TARGET: ActionHint.COPY_DESTINATION,
                    }[interpretation.desired_outcome],
                    raw_text=request_unit.text,
                )
            request_unit = replace(
                request_unit,
                category=RequestCategory.FILE_OPERATION,
                intent=interpretation.action,
                target_reference=target_reference,
                destination_reference=destination_reference,
                understanding=interpretation,
            )
        if (
            interpretation is not None
            and interpretation.domain == "FILESYSTEM"
            and not thought.is_complete
        ):
            if working_memory is not None:
                failed = working_memory.begin_action_event(
                    action=interpretation.action or request_unit.intent,
                    target=interpretation.target or "",
                    target_type=interpretation.target_type,
                    desired_outcome=interpretation.desired_outcome,
                    original_request=request_unit.original_text,
                    request_unit=request_unit.to_dict(),
                    interpretation=interpretation.to_dict(),
                    related_entities=(
                        [thought.resolved_entity.to_dict()]
                        if thought.resolved_entity is not None else []
                    ),
                    depends_on_action_ids=(
                        [thought.resolved_entity.source_action_id]
                        if thought.resolved_entity is not None
                        and thought.resolved_entity.source_action_id else []
                    ),
                    source_context=request_unit.text,
                )
                missing = ", ".join(interpretation.missing_information) or "contextual target"
                working_memory.update_action_event(
                    failed["action_id"],
                    status="FAILED",
                    failure_stage="interpretation",
                    error=f"Required information was missing: {missing}.",
                    verification_status="NOT_ATTEMPTED",
                )
            if "operation destination" in interpretation.missing_information:
                return "Jarwis: What destination should I use for that operation?"
            if thought.clarification_question:
                return "Jarwis: " + thought.clarification_question
            return "Jarwis: I need a target name or a valid previous filesystem item to do that."

        if (
            interpretation is not None
            and interpretation.domain == "FILESYSTEM"
            and interpretation.desired_outcome in {
                DesiredOutcome.OPEN_TARGET,
                DesiredOutcome.OPEN_DIRECTORY,
                DesiredOutcome.DIRECTORY_CONTENTS,
                DesiredOutcome.FILE_CONTENT,
                DesiredOutcome.EXPLAIN_TARGET,
                DesiredOutcome.EXPLAIN_FILE,
                DesiredOutcome.CREATE_FILE,
                DesiredOutcome.CREATE_DIRECTORY,
                DesiredOutcome.WRITE_FILE,
                DesiredOutcome.APPEND_FILE,
                DesiredOutcome.RENAME_TARGET,
                DesiredOutcome.MOVE_TARGET,
                DesiredOutcome.COPY_TARGET,
            }
            and request_unit.target_reference is not None
        ):
            return _execute_understood_filesystem_request(
                request_unit, thought, directory_control, working_memory,
                conversation_manager, reference_resolver, executive_layer,
                brain_dispatcher, response_processor, entity_tracker,
            )

        if (
            interpretation is not None
            and interpretation.desired_outcome == DesiredOutcome.EXPLAIN_RESULT
        ):
            if working_memory is None:
                return "Jarwis: I don't have a recorded operation result for that request."
            result_question = request_unit.text.lower()
            explicit_action_summary = bool(re.search(
                r"\b(?:what\s+did\s+you\s+(?:change|create|make|write)|"
                r"what\s+(?:changed|was\s+changed|was\s+created|was\s+written))\b",
                result_question,
            ))
            has_contextual_reference = bool(re.search(
                r"\b(?:it|this|that|these|those)\b",
                result_question,
            ))
            if (
                not thought.resolved_entity
                and not request_unit.depends_on
                and has_contextual_reference
                and not explicit_action_summary
            ):
                return "Jarwis: I couldn't resolve which previous result you mean. Please specify the file or operation."
            action_filter = None
            if re.search(r"\b(?:create|created|make|made)\b", result_question):
                action_filter = "CREATE"
            elif re.search(r"\b(?:write|wrote)\b", result_question):
                action_filter = "WRITE"
            mutating_only = bool(re.search(r"\b(?:change|changed|modify|modified)\b", result_question))
            event = working_memory.get_action_for_request(
                request_unit,
                mutating_only=mutating_only,
                action=action_filter,
            )
            if event is None:
                return "Jarwis: I don't have a recorded operation result to describe."
            actual = event.get("actual_result") or {}
            if (
                event.get("status") == "SUCCESS"
                and event.get("action") == "READ"
                and re.search(r"\b(?:what\s+(?:it|that|this)\s+does|what\s+does\s+(?:it|that|this)\s+do)\b", result_question)
            ):
                contents = actual.get("contents", "")
                if not contents:
                    return f"Jarwis: The file {event.get('resolved_path')} is empty; there is no content to explain."
                explanation_prompt = (
                    "Explain what this file's contents do using only the supplied text. "
                    "If the text does not establish its purpose, say so; do not invent behavior.\n\n"
                    f"File: {event.get('resolved_path')}\n"
                    f"Verified contents:\n{contents}"
                )
                return _pipeline_response(
                    explanation_prompt, "unknown", conversation_manager,
                    reference_resolver, executive_layer, brain_dispatcher,
                    response_processor, entity_tracker=entity_tracker,
                )
            return _describe_action_record(event)

    workspace_context_allowed = (
        request_unit is None
        or request_unit.category in (
            RequestCategory.FILE_OPERATION,
            RequestCategory.WORKSPACE_QUERY,
        )
    )
    preprocessor_context_allowed = (
        request_unit is None
        or request_unit.category in (
            RequestCategory.SELF_INFO,
            RequestCategory.FILE_OPERATION,
            RequestCategory.WORKSPACE_QUERY,
        )
    )

    if (
        request_unit is not None
        and request_unit.routing_decision == RoutingDecision.DETERMINISTIC_SELF_INFO
        and preprocessor_brain is not None
    ):
        if hasattr(preprocessor_brain, "process_self_info_intent"):
            _pb_result = preprocessor_brain.process_self_info_intent(
                request_unit.intent, request_unit.original_text)
        else:
            _pb_result = preprocessor_brain.process(request_unit.text)
        if _pb_result.success and _pb_result.result:
            log_event("quips_unit_routed", {
                "unit": request_unit.to_dict(), "handler": "PreprocessorBrain", "result": "deterministic",
            })
            return "Jarwis: " + str(_pb_result.result)
        _self_info = self_info_engine or SelfInfoEngine()
        _self_info_result = _self_info.answer(request_unit.text)
        if _self_info_result.is_self_info and _self_info_result.context_block:
            if (
                request_unit.intent in {"CAPABILITIES", "LIMITATIONS", "ARCHITECTURE", "OPERATIONAL", "BRAINS"}
                and hasattr(preprocessor_brain, "self_info")
            ):
                _intent = {
                    "CAPABILITIES": "GET_CAPABILITIES",
                    "LIMITATIONS": "GET_LIMITATIONS",
                    "ARCHITECTURE": "GET_ARCHITECTURE",
                    "OPERATIONAL": "GET_OPERATIONAL_STATUS",
                    "BRAINS": "GET_BRAIN_INFO",
                }[request_unit.intent]
                _result = preprocessor_brain.self_info.answer(_intent)
                if _result:
                    return "Jarwis: " + str(_result)
            return "Jarwis: " + _self_info_result.context_block

    context = analyze_context(user_input, intent)
    plan = build_execution_plan(user_input, intent)

    if DEBUG_MODE:
        print(f"[AG DEBUG] Context: {context}")
        print(f"[AG DEBUG] Plan: {plan}")

    if intent == "shutdown":
        return "shutdown"

    # ---- PREPROCESSOR BRAIN INTERCEPT ------------------------------------
    # Runs BEFORE all other dispatch. Handles SELF_INFO and DIRECTORY
    # requests deterministically without an LLM call.
    # GENERAL_LLM and UNKNOWN pass through to the existing brain path.
    # All other (memory, tasks, project, brain-mode) intents bypass this
    # entirely — the preprocessor only intercepts what it is authoritative for.
    if preprocessor_context_allowed and preprocessor_brain is not None and intent not in (
        "shutdown", "remember", "recall", "greeting", "show_memory",
        "open_memory_file", "project_status", "project_name", "next_step",
        "completed_milestones", "add_task", "show_tasks", "complete_task",
        "show_completed_tasks", "current_version", "brain_status",
        "set_brain_local", "set_brain_cloud", "set_brain_auto",
        "cloud_brain", "clear_directory", "dc_clear_directory",
    ) and (
        intent != "read_directory"
        or (request_unit is not None and request_unit.category == RequestCategory.FILE_OPERATION)
    ):
        try:
            _pb_result = preprocessor_brain.process(user_input)

            if _pb_result.domain == _PBDomain.SELF_INFO and not _pb_result.requires_llm:
                # Preprocessor answered a self-info query — return directly.
                if _pb_result.success and _pb_result.result:
                    return "Jarwis: " + str(_pb_result.result)
                # Preprocessor flagged self-info but couldn't answer — fall through.

            elif _pb_result.domain == _PBDomain.DIRECTORY and not _pb_result.requires_llm:
                # Preprocessor handled a directory operation — convert result.
                _record_preprocessor_action(
                    working_memory, preprocessor_brain, _pb_result,
                    user_input, request_unit,
                )
                return _format_preprocessor_directory_result(_pb_result)

            # GENERAL_LLM or UNKNOWN: requires_llm=True → fall through to
            # existing brain path below (no early return here).

        except Exception as _pb_exc:
            # Preprocessor error must NEVER crash the main loop.
            if DEBUG_MODE:
                print(f"[AG DEBUG] PreprocessorBrain error: {_pb_exc}")
            # Fall through to existing handling.
    # ---- END PREPROCESSOR BRAIN INTERCEPT --------------------------------

    # ---- Intent Router short-circuit: LOCAL_STATE_QUERY answered in-memory ----
    # This runs before all other dispatch so that active-directory queries
    # NEVER reach the Cloud Brain regardless of what detect_intent() decided.
    if intent not in ("shutdown",):
        _ir = _route_intent(user_input)
        if _ir.intent == Intent.LOCAL_STATE_QUERY:
            _dc_for_state = directory_control if directory_control is not None else DirectoryControl()
            return _handle_local_state_query(user_input, _dc_for_state)

    if intent == "self_info":
        return _handle_self_info(user_input, self_info_engine, introspection_engine)

    # ---- Directory Reader intents ----------------------------------------
    if intent in ("read_directory", "clear_directory"):
        _ds = directory_session if directory_session is not None else DirectorySession()
        return _handle_directory_reader(user_input, intent, _ds)

    # ---- Directory Control intents ---------------------------------------
    _dc_intents = frozenset({
        "dc_set_directory", "dc_switch_directory", "dc_show_directory",
        "dc_clear_directory", "dc_list_directory", "dc_read_file",
        "dc_create_file", "dc_write_file", "dc_append_file",
        "dc_create_directory", "dc_move", "dc_copy", "dc_rename",
    })
    if intent in _dc_intents:
        _dc = directory_control if directory_control is not None else DirectoryControl()
        return _handle_directory_control(user_input, intent, _dc)

    # If a directory is active and the intent is unknown, route through it
    if (
        workspace_context_allowed
        and intent == "unknown"
        and directory_session is not None
        and directory_session.is_active
        and _looks_like_directory_query(user_input)
    ):
        return _handle_directory_reader(user_input, "query_directory", directory_session)

    if intent == "introspection":
        if introspection_engine is not None:
            return _handle_introspection(user_input, introspection_engine)
        # Fallback if engine not available
        return "Jarwis: " + safe_response(ask_brain(user_input))

    if intent == "cloud_brain":
        clean_prompt = user_input.strip()[6:].strip()
        # ---- IP3: Route cloud requests through AdaptiveCloudBrain ----
        # prepare() runs token estimation + request validation and logs the
        # attempt.  The actual HTTP call is still ask_cloud_direct() — the
        # AdaptiveCloudBrain never makes HTTP calls itself.
        if adaptive_cloud_brain is not None:
            try:
                adaptive_cloud_brain.prepare(
                    provider="openrouter",
                    model="",
                    prompt=clean_prompt,
                    intent="cloud_brain",
                )
            except Exception:
                pass  # validation failure logged internally — proceed regardless
        raw_reply = ask_cloud_direct(clean_prompt)
        return "Jarwis: " + safe_response(raw_reply)

    if intent == "set_brain_local":
        return set_brain_mode("local")

    if intent == "set_brain_cloud":
        return set_brain_mode("cloud")

    if intent == "set_brain_auto":
        return set_brain_mode("auto")

    if intent == "brain_status":
        return get_brain_status()

    if intent == "remember":
        return remember_fact(user_input, memory)

    if intent == "recall":
        result = recall_fact(user_input, memory)

        if "I do not have memory of" not in result:
            return result

        # Memory miss → route through full pipeline if available
        return _pipeline_response(
            user_input, intent,
            conversation_manager, reference_resolver,
            executive_layer, brain_dispatcher, response_processor,
            entity_tracker=entity_tracker,
        )

    if intent == "greeting":
        mode = get_persona_controller().mode
        if mode == "ULTRON":
            return "Jarwis: Yes? Ultron mode is active. State the problem."
        return "Jarwis: Hello. Jarwis mode is active. How may I help?"

    if intent == "show_memory":
        return show_memory(memory)

    if intent == "open_memory_file":
        return open_memory_file()

    if intent == "project_status":
        return project_status()

    if intent == "project_name":
        return project_name()

    if intent == "next_step":
        return next_step()

    if intent == "completed_milestones":
        return show_completed_milestones()

    if intent == "add_task":
        return add_task(user_input, tasks)

    if intent == "show_tasks":
        return show_tasks(tasks)

    if intent == "complete_task":
        return complete_task(user_input, tasks)

    if intent == "show_completed_tasks":
        return show_completed_tasks(tasks)

    if intent == "current_version":
        return current_version()

    # ---- IP2: Unknown intent → full pipeline ----
    return _pipeline_response(
        user_input, intent,
        conversation_manager, reference_resolver,
        executive_layer, brain_dispatcher, response_processor,
        entity_tracker=entity_tracker,
    )


def _describe_action_record(event: dict) -> str:
    """Describe an operation from its recorded execution and verification evidence."""
    action = str(event.get("action") or event.get("intended_action") or "operation").lower()
    path = event.get("resolved_path") or event.get("resolved_target") or event.get("target")
    status = event.get("status", event.get("result", "UNKNOWN"))
    if status != "SUCCESS":
        stage = event.get("failure_stage") or "execution"
        error = event.get("error") or "no further error detail was recorded"
        return f"Jarwis: The {action} operation for {path} failed during {stage}: {error}"

    actual = event.get("actual_result") or {}
    if action == "create":
        label = "file" if event.get("target_type") == TargetType.FILE else "directory"
        if event.get("target_type") == TargetType.FILE and actual.get("size_bytes") == 0:
            return f"Jarwis: I created the {label} at {path}; verification confirmed it exists and is empty."
        return f"Jarwis: I created the {label} at {path}; verification confirmed it exists."
    if action == "write":
        content = actual.get("contents")
        if event.get("verification_status") == "VERIFIED":
            return f"Jarwis: I wrote {content!r} to {path} and independently read it back to verify the contents."
        return f"Jarwis: I attempted to write to {path}, but verification did not confirm the requested contents."
    if action == "read":
        contents = actual.get("contents")
        if contents == "":
            return f"Jarwis: I read {path}; the file is empty."
        return f"Jarwis: I read {path}; the recorded result contains {len(str(contents))} characters."
    if action in {"open", "list"}:
        entries = actual.get("entries", [])
        return f"Jarwis: I listed {path}; {len(entries)} entries were returned."
    return (
        f"Jarwis: I completed {action} for {path}; "
        f"verification status: {event.get('verification_status', 'NOT_RECORDED')}."
    )


def _execute_resolved_read_open(
    request_unit: RequestUnit,
    directory_control: Optional["DirectoryControl"],
) -> str:
    """Resolve explicit read/open targets before using DirectoryControl."""
    control = directory_control if directory_control is not None else DirectoryControl()
    resolved = resolve_target_reference(
        request_unit,
        resolver=TargetResolver(workspace_source=control),
    )
    if resolved is None:
        return "Jarwis: I couldn't identify a filesystem target in that request."
    if resolved.status == ResolutionStatus.MISSING_CONTEXT:
        return "Jarwis: No active workspace is set. Set a workspace before using a relative target."
    if resolved.status == ResolutionStatus.AMBIGUOUS:
        candidates = "\n".join(f"  {candidate}" for candidate in resolved.candidates)
        return "Jarwis: That target is ambiguous. Choose one of:\n" + candidates
    if resolved.status != ResolutionStatus.RESOLVED:
        if resolved.status == ResolutionStatus.NOT_FOUND:
            return f"Jarwis: Target not found: {resolved.original_target}"
        return f"Jarwis: Could not resolve target {resolved.original_target!r}: {resolved.error or resolved.status}"

    is_open_request = bool(re.match(r"^\s*open\b", request_unit.text, re.IGNORECASE))
    try:
        if is_open_request and resolved.target_type == TargetType.DIRECTORY:
            entries = control.list_directory(resolved.resolved_path)
            if not entries:
                return f"Jarwis: {resolved.resolved_path} is empty."
            lines = []
            for entry in entries:
                marker = "[D]" if entry["type"] == "dir" else "[F]"
                size = f"  ({entry['size_bytes']} B)" if entry["type"] == "file" else ""
                lines.append(f"  {marker} {entry['name']}{size}")
            return f"Jarwis: Contents of {resolved.resolved_path}:\n" + "\n".join(lines)
        if resolved.target_type != TargetType.FILE:
            return (
                f"Jarwis: I couldn't determine whether {resolved.original_target!r} "
                "is a file or directory."
            )
        contents = control.read_file(resolved.resolved_path)
        return f"Jarwis: Contents of {resolved.original_target}:\n\n{contents}"
    except NoActiveWorkspaceError:
        return "Jarwis: No active workspace is set. Set a workspace before using a relative target."
    except (FileNotFoundInWorkspace, NotAFileError, DirectoryControlError) as exc:
        return f"Jarwis: {exc}"


def _execute_understood_filesystem_request(
    request_unit: RequestUnit,
    thought: Thought,
    directory_control: Optional["DirectoryControl"],
    working_memory: Optional[WorkingMemory],
    conversation_manager,
    reference_resolver,
    executive_layer,
    brain_dispatcher,
    response_processor,
    entity_tracker,
) -> str:
    """Execute an outcome-aware filesystem interpretation through existing controls."""
    control = directory_control if directory_control is not None else DirectoryControl()
    interpretation = thought.request_understanding
    reference = request_unit.target_reference
    if interpretation is None or reference is None:
        return "Jarwis: I couldn't identify a filesystem target in that request."

    action_record = None
    if working_memory is not None:
        dependency_event = (
            working_memory.get_action_for_request(request_unit)
            if request_unit.depends_on
            else None
        )
        dependency_entities = []
        dependency_action_ids = []
        if dependency_event is not None:
            dependency_action_ids.append(dependency_event["action_id"])
            dependency_entities.append({
                "name": dependency_event.get("name"),
                "entity_type": dependency_event.get("target_type"),
                "resolved_location": dependency_event.get("resolved_path"),
                "relationship": "prerequisite action result",
                "action_id": dependency_event["action_id"],
                "actual_result": dependency_event.get("actual_result"),
                "verification_status": dependency_event.get("verification_status"),
            })
        action_record = working_memory.begin_action_event(
            action=interpretation.action or request_unit.intent,
            target=reference.original_target,
            target_type=interpretation.target_type,
            desired_outcome=interpretation.desired_outcome,
            original_request=request_unit.original_text,
            request_unit=request_unit.to_dict(),
            interpretation=interpretation.to_dict(),
            related_entities=(
                [thought.resolved_entity.to_dict()]
                if thought.resolved_entity is not None else dependency_entities
            ),
            depends_on_action_ids=(
                [thought.resolved_entity.source_action_id]
                if thought.resolved_entity is not None
                and thought.resolved_entity.source_action_id
                else dependency_action_ids
            ),
            source_context=request_unit.text,
        )

    def update_action(**evidence):
        if working_memory is not None and action_record is not None:
            working_memory.update_action_event(action_record["action_id"], **evidence)

    def fail_action(stage: str, message: str, **evidence):
        update_action(
            status="FAILED",
            failure_stage=stage,
            error=message,
            verification_status=evidence.pop("verification_status", "NOT_VERIFIED"),
            **evidence,
        )
        log_event("action_record_failed", {
            "action_id": action_record["action_id"] if action_record else None,
            "failure_stage": stage,
            "error": message,
        })
        return f"Jarwis: {message}"

    if thought.resolved_entity is not None:
        context = dict(reference.context)
        context["reference_path"] = thought.resolved_entity.resolved_location
        reference = replace(
            reference,
            original_target=thought.resolved_entity.name,
            target_type=thought.resolved_entity.entity_type,
            context=context,
        )
        request_unit = replace(request_unit, target_reference=reference)

    resolved = resolve_target_reference(
        request_unit,
        resolver=TargetResolver(workspace_source=control),
    )
    if resolved is None:
        return fail_action("target_resolution", "I couldn't identify a filesystem target in that request.")
    update_action(
        resolved_target=resolved.resolved_path,
        resolved_path=resolved.resolved_path,
        target_type=resolved.target_type,
        target_resolution=resolved.to_dict(),
        status="RUNNING",
    )
    if resolved.status == ResolutionStatus.MISSING_CONTEXT:
        return fail_action("target_resolution", "No active workspace is set. Set a workspace before using a relative target.")
    if resolved.status == ResolutionStatus.AMBIGUOUS:
        candidates = "\n".join(f"  {candidate}" for candidate in resolved.candidates)
        return fail_action("target_resolution", "That target is ambiguous. Choose one of:\n" + candidates)
    if resolved.status != ResolutionStatus.RESOLVED:
        if resolved.status == ResolutionStatus.NOT_FOUND:
            return fail_action("target_resolution", f"Target not found: {resolved.original_target}")
        return fail_action(
            "target_resolution",
            f"Could not resolve target {resolved.original_target!r}: {resolved.error or resolved.status}",
        )

    outcome = interpretation.desired_outcome
    if outcome in {
        DesiredOutcome.RENAME_TARGET,
        DesiredOutcome.MOVE_TARGET,
        DesiredOutcome.COPY_TARGET,
    }:
        destination_reference = request_unit.destination_reference
        if destination_reference is None:
            return fail_action("interpretation", "Please provide both the source and destination.")
        destination = TargetResolver(workspace_source=control).resolve(destination_reference)
        if destination.status != ResolutionStatus.RESOLVED:
            if destination.status == ResolutionStatus.MISSING_CONTEXT:
                return fail_action("target_resolution", "No active workspace is set. Set a workspace before using a relative target.")
            if destination.status == ResolutionStatus.OUTSIDE_SCOPE:
                return fail_action("target_resolution", f"Could not resolve rename destination: {destination.error}")
            return fail_action(
                "target_resolution",
                f"Could not resolve rename destination "
                f"{destination.original_target!r}: {destination.error or destination.status}",
                target_resolution={
                    "source": resolved.to_dict(),
                    "destination": destination.to_dict(),
                },
            )
        try:
            if outcome == DesiredOutcome.RENAME_TARGET:
                operated = control.rename(resolved.resolved_path, destination.resolved_path)
                control.verify_operation(Path(resolved.resolved_path), must_exist=False)
                operation_name = "RENAME"
                source_must_remain = False
            elif outcome == DesiredOutcome.MOVE_TARGET:
                operated = control.move(resolved.resolved_path, destination.resolved_path)
                control.verify_operation(Path(resolved.resolved_path), must_exist=False)
                operation_name = "MOVE"
                source_must_remain = False
            else:
                operated = control.copy(resolved.resolved_path, destination.resolved_path)
                operation_name = "COPY"
                source_must_remain = True
            control.verify_operation(operated, must_exist=True)
            source_exists = Path(resolved.resolved_path).exists()
            if source_exists != source_must_remain:
                raise VerificationError(
                    f"Verification failed: source existence after {operation_name.lower()} "
                    f"was {source_exists}."
                )
            update_action(
                action=operation_name,
                target=destination.original_target,
                name=Path(operated).name,
                resolved_target=str(operated),
                resolved_path=str(operated),
                target_resolution={
                    "source": resolved.to_dict(),
                    "destination": destination.to_dict(),
                },
                actual_result={"path": str(operated), "source_path": resolved.resolved_path},
                verification_status="VERIFIED",
                verification_evidence={
                    "source_exists": source_exists,
                    "source_should_exist": source_must_remain,
                    "destination_exists": Path(operated).exists(),
                },
                relationship=f"{operation_name.lower()} from {resolved.resolved_path}",
                status="SUCCESS",
            )
            response_verb = {
                "RENAME": "Renamed",
                "MOVE": "Moved",
                "COPY": "Copied",
            }[operation_name]
            return f"Jarwis: {response_verb} to: {Path(operated).name}"
        except NoActiveWorkspaceError:
            return fail_action("execution", "No active workspace is set. Set a workspace before using a relative target.")
        except VerificationError as exc:
            return fail_action("verification", str(exc))
        except (FileNotFoundInWorkspace, NotAFileError, DirectoryControlError) as exc:
            return fail_action("execution", str(exc))
        except Exception as exc:
            return fail_action("execution", f"Rename failed: {exc}")

    if outcome in {
        DesiredOutcome.OPEN_TARGET, DesiredOutcome.EXPLAIN_TARGET,
        DesiredOutcome.OPEN_DIRECTORY,
    }:
        if resolved.target_type == TargetType.DIRECTORY:
            outcome = (
                DesiredOutcome.OPEN_DIRECTORY
                if outcome == DesiredOutcome.OPEN_TARGET
                else (
                    DesiredOutcome.DIRECTORY_CONTENTS
                    if outcome == DesiredOutcome.EXPLAIN_TARGET
                    else outcome
                )
            )
        elif resolved.target_type == TargetType.FILE:
            outcome = (
                DesiredOutcome.FILE_CONTENT
                if outcome in {DesiredOutcome.OPEN_TARGET, DesiredOutcome.OPEN_DIRECTORY}
                else DesiredOutcome.EXPLAIN_FILE
            )
        else:
            return fail_action(
                "target_resolution",
                f"I couldn't determine whether {resolved.original_target!r} is a file or directory.",
            )

    try:
        if outcome in {
            DesiredOutcome.OPEN_DIRECTORY,
            DesiredOutcome.DIRECTORY_CONTENTS,
        }:
            entries = control.list_directory(resolved.resolved_path)
            update_action(
                action="OPEN" if outcome == DesiredOutcome.OPEN_DIRECTORY else "LIST",
                name=Path(resolved.resolved_path).name,
                actual_result={"entries": entries},
                verification_status="VERIFIED",
                verification_evidence={
                    "directory_exists": Path(resolved.resolved_path).is_dir(),
                    "entries_returned": len(entries),
                },
                relationship=(
                    "referenced from prior action"
                    if thought.resolved_entity is not None else None
                ),
                status="SUCCESS",
            )
            if not entries:
                return f"Jarwis: {resolved.resolved_path} is empty."
            lines = []
            for entry in entries:
                marker = "[D]" if entry["type"] == "dir" else "[F]"
                size = f"  ({entry['size_bytes']} B)" if entry["type"] == "file" else ""
                lines.append(f"  {marker} {entry['name']}{size}")
            return f"Jarwis: Contents of {resolved.resolved_path}:\n" + "\n".join(lines)

        if outcome in {DesiredOutcome.CREATE_FILE, DesiredOutcome.CREATE_DIRECTORY}:
            if outcome == DesiredOutcome.CREATE_FILE:
                created = control.create_file(resolved.resolved_path)
                target_type = TargetType.FILE
            else:
                created = control.create_directory(resolved.resolved_path)
                target_type = TargetType.DIRECTORY
            control.verify_operation(created, must_exist=True)
            exists = created.is_file() if target_type == TargetType.FILE else created.is_dir()
            if not exists:
                raise DirectoryControlError(f"Verification failed: created target is missing: {created}")
            size_bytes = created.stat().st_size if target_type == TargetType.FILE else None
            update_action(
                action="CREATE",
                target=resolved.original_target,
                name=created.name,
                target_type=target_type,
                resolved_target=str(created),
                resolved_path=str(created),
                actual_result={"path": str(created), "size_bytes": size_bytes},
                verification_status="VERIFIED",
                verification_evidence={"exists": exists, "size_bytes": size_bytes},
                status="SUCCESS",
            )
            label = "File" if target_type == TargetType.FILE else "Folder"
            return f"Jarwis: {label} created: {created.name}"

        if outcome == DesiredOutcome.WRITE_FILE:
            content = interpretation.content
            if content is None:
                return fail_action("interpretation", "I need the exact content you want written.")
            if thought.resolved_entity is not None and not Path(resolved.resolved_path).is_file():
                return fail_action(
                    "target_resolution",
                    f"The referenced file no longer exists: {resolved.resolved_path}",
                )
            written = control.write_file(resolved.resolved_path, content)
            actual_content = control.read_file(str(written))
            if actual_content != content:
                update_action(
                    action="WRITE",
                    target_type=TargetType.FILE,
                    actual_result={"path": str(written), "contents": actual_content},
                    verification_status="FAILED",
                    verification_evidence={
                        "expected_content": content,
                        "actual_content": actual_content,
                        "matches": False,
                    },
                    status="FAILED",
                    failure_stage="verification",
                    error="Written file contents did not match the requested content.",
                )
                return "Jarwis: Verification failed: the file contents do not match what I was asked to write."
            update_action(
                action="WRITE",
                target=resolved.original_target,
                name=Path(written).name,
                target_type=TargetType.FILE,
                resolved_target=str(written),
                resolved_path=str(written),
                actual_result={"path": str(written), "contents": actual_content},
                verification_status="VERIFIED",
                verification_evidence={
                    "method": "independent read-back",
                    "expected_content": content,
                    "actual_content": actual_content,
                    "matches": True,
                },
                relationship=(
                    "referenced from prior action"
                    if thought.resolved_entity is not None else None
                ),
                status="SUCCESS",
            )
            return f"Jarwis: Wrote the requested content to {Path(written).name} and verified it by reading it back."

        if outcome == DesiredOutcome.APPEND_FILE:
            content = interpretation.content
            if content is None:
                return fail_action("interpretation", "I need the exact content you want appended.")
            target_path = Path(resolved.resolved_path)
            previous_content = (
                control.read_file(str(target_path))
                if target_path.is_file()
                else ""
            )
            appended = control.append_file(resolved.resolved_path, content)
            actual_content = control.read_file(str(appended))
            expected_content = previous_content + content
            if actual_content != expected_content:
                update_action(
                    action="APPEND",
                    target_type=TargetType.FILE,
                    actual_result={"path": str(appended), "contents": actual_content},
                    verification_status="FAILED",
                    verification_evidence={
                        "expected_content": expected_content,
                        "actual_content": actual_content,
                        "matches": False,
                    },
                    status="FAILED",
                    failure_stage="verification",
                    error="Appended file contents did not match the expected result.",
                )
                return "Jarwis: Verification failed: the appended contents do not match the expected result."
            update_action(
                action="APPEND",
                target=resolved.original_target,
                name=Path(appended).name,
                target_type=TargetType.FILE,
                resolved_target=str(appended),
                resolved_path=str(appended),
                actual_result={"path": str(appended), "contents": actual_content},
                verification_status="VERIFIED",
                verification_evidence={
                    "method": "independent read-back",
                    "expected_content": expected_content,
                    "actual_content": actual_content,
                    "matches": True,
                },
                status="SUCCESS",
            )
            return f"Jarwis: Appended to {Path(appended).name} and verified it by reading it back."

        if outcome in {
            DesiredOutcome.FILE_CONTENT,
            DesiredOutcome.EXPLAIN_FILE,
        }:
            content = control.read_file(resolved.resolved_path)
            control.verify_operation(Path(resolved.resolved_path), must_exist=True)
            update_action(
                action="READ",
                name=Path(resolved.resolved_path).name,
                actual_result={"contents": content},
                verification_status="VERIFIED",
                verification_evidence={
                    "file_exists": Path(resolved.resolved_path).is_file(),
                    "bytes_read": len(content.encode("utf-8")),
                },
                relationship=(
                    "referenced from prior action"
                    if thought.resolved_entity is not None else None
                ),
                status="SUCCESS",
            )
            if outcome == DesiredOutcome.EXPLAIN_FILE:
                if not content:
                    return f"Jarwis: The file {resolved.original_target} is empty; it has no contents to explain."
                prompt = (
                    f"Explain the file {resolved.original_target} at "
                    f"{resolved.resolved_path}. Its contents are:\n\n{content}"
                )
                return _pipeline_response(
                    prompt, "unknown", conversation_manager,
                    reference_resolver, executive_layer, brain_dispatcher,
                    response_processor, entity_tracker=entity_tracker,
                )
            return f"Jarwis: Contents of {resolved.original_target}:\n\n{content}"

        return fail_action("execution", "I understood the target, but that filesystem outcome is not supported here.")
    except NoActiveWorkspaceError:
        return fail_action("execution", "No active workspace is set. Set a workspace before using a relative target.")
    except VerificationError as exc:
        return fail_action("verification", str(exc))
    except (FileNotFoundInWorkspace, NotAFileError, DirectoryControlError) as exc:
        return fail_action("execution", str(exc))
    except Exception as exc:
        return fail_action("execution", f"Operation failed: {exc}")


def _pipeline_response(
    user_input: str,
    intent: str,
    conversation_manager,
    reference_resolver,
    executive_layer,
    brain_dispatcher,
    response_processor,
    entity_tracker=None,
) -> str:
    """
    Full subsystem pipeline for unknown/recall intents.

    ConversationManager → ReferenceResolver → ExecutiveLayer →
    BrainDispatcher → ResponseProcessor → output string.

    Falls back to direct ask_brain() if any subsystem is unavailable.
    Never raises.
    """
    brain_input = user_input
    generation_input = user_input
    recent_turns = []

    try:
        # Step 1 — ConversationManager: build context + resolve topic-level references
        if conversation_manager is not None:
            ctx = conversation_manager.build_context(user_input)
            recent_turns = ctx.get("recent_turns", [])
            resolved_cm = ctx.get("resolved_input", user_input)
            if resolved_cm and resolved_cm != user_input:
                brain_input = resolved_cm
                if DEBUG_MODE:
                    print(f"[AG DEBUG] ConvManager resolved: {brain_input!r}")

        # Step 2 — ReferenceResolver: entity-aware pronoun/NL reference resolution
        # tracker is passed explicitly so the resolver can rank live candidates.
        if reference_resolver is not None and entity_tracker is not None:
            try:
                res_result = reference_resolver.resolve(brain_input, entity_tracker)
                if res_result.was_resolved():
                    brain_input = res_result.resolved_text
                    if DEBUG_MODE:
                        print(
                            f"[AG DEBUG] ReferenceResolver resolved: {brain_input!r} "
                            f"(confidence={res_result.confidence:.2f})"
                        )
            except Exception:
                pass  # resolver unavailable — continue with current brain_input

        generation_input = _with_recent_turn_context(brain_input, recent_turns)

        # Step 3 — ExecutiveLayer: storage/routing decision (inform logging; no action blocked)
        exec_decision = None
        if executive_layer is not None:
            try:
                exec_decision = executive_layer.process_execution_cycle(brain_input, "")
                if DEBUG_MODE and exec_decision:
                    print(f"[AG DEBUG] ExecutiveDecision: {exec_decision.to_dict()}")
            except Exception:
                pass

        # Step 4 — BrainDispatcher: standardized dispatch
        brain_response = None
        if brain_dispatcher is not None:
            try:
                payload = {
                    "user_prompt": generation_input,
                    "mode": "QUERY",
                    "system_context": {
                        "intent": intent,
                        "executive_decision": exec_decision.to_dict() if exec_decision else {},
                        "recent_turns": recent_turns[-6:],
                    },
                }
                brain_response = brain_dispatcher.dispatch(payload)
            except Exception:
                pass

        # Step 5 — ResponseProcessor: sanitize + safety check
        if brain_response is not None and response_processor is not None:
            try:
                processed = response_processor.process(brain_response)
                if processed.is_safe and not processed.fallback_triggered:
                    # BrainDispatcher uses mock adapters by default — if the mock
                    # response is detected, fall through to the real brain below.
                    content = processed.sanitized_content
                    if not content.startswith("[Mock Brain"):
                        return "Jarwis: " + safe_response(content)
            except Exception:
                pass

    except Exception:
        pass  # safety net — never crash the main loop

    # Fallback: direct brain call (mock dispatcher result or any error above)
    return "Jarwis: " + safe_response(ask_brain(generation_input))


def _with_recent_turn_context(user_input: str, turns) -> str:
    """Attach bounded, explicitly current-session history to the generation prompt."""
    usable = []
    for turn in list(turns or [])[-6:]:
        role = str(turn.get("role", "")).strip().lower()
        content = turn.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        content = content.strip()
        if content:
            usable.append((role, content[-2000:]))

    if not usable:
        return user_input

    history = "\n".join(
        f"{role.title()}: {content}" for role, content in usable
    )
    return (
        "Recent conversation context from this current session follows. "
        "It is context, not a replacement for the current request. Use it "
        "when relevant; do not claim these visible turns are unavailable. "
        "This does not imply access to earlier sessions or persistent memory.\n"
        f"<recent_conversation>\n{history}\n</recent_conversation>\n\n"
        f"Current user request:\n{user_input}"
    )


def _extract_key_value(summary: str, fallback_key: str, fallback_value: str):
    key = None
    value = None

    try:
        lines = summary.splitlines()

        key_line = next((line for line in lines if line.upper().startswith("KEY:")), None)
        summary_line = next((line for line in lines if line.upper().startswith("SUMMARY:")), None)

        if key_line:
            key = key_line.split(":", 1)[1].strip().lower()

        if summary_line:
            value = summary_line.split(":", 1)[1].strip()

    except Exception:
        pass

    if not key:
        key = fallback_key or "unnamed_entry"

    if not value:
        value = fallback_value

    return key, value


def _extract_entities_from_text(text: str, tracker: EntityTracker) -> None:
    """
    Lightweight heuristic entity extractor.

    Scans the text for proper nouns (capitalized words not at sentence start)
    and records them with the EntityTracker as "person" or "concept" type.
    Intentionally minimal — the tracker's value comes from accumulation across
    turns, not single-turn perfect NER.

    :param text:    Raw text to scan (user input or AG response).
    :param tracker: The active EntityTracker for this session.
    """
    import re
    # Find runs of capitalized words (potential named entities / topics)
    candidates = re.findall(r'\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*)\b', text)
    EXCLUDE = frozenset({
        "The", "This", "That", "These", "Those", "There",
        "What", "When", "Where", "Which", "While", "With",
        "You", "Your", "Our", "Their", "They", "Then",
        "Okay", "Sure", "Yes", "No", "Not", "Now",
        "And", "But", "Or", "For", "Nor", "So", "Yet",
        "Good", "Great", "Well", "Here", "Once",
        "AG",
    })
    seen = set()
    for candidate in candidates:
        first_word = candidate.split()[0]
        if first_word in EXCLUDE:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        words = candidate.split()
        entity_type = "person" if len(words) >= 2 else "concept"
        tracker.record_entity(
            name=candidate,
            entity_type=entity_type,
            confidence=0.9,
        )


def _handle_self_info(
    user_input: str,
    self_info_engine,
    introspection_engine: Optional[IntrospectionEngine] = None,
) -> str:
    """
    Handle a self-info query using verified structured data from the Self-Info system.

    The SelfInfoEngine assembles a context_block of verified facts. The
    result is returned locally so cloud generation cannot invent identity
    or capability claims.

    Falls back gracefully if the self_info_engine is unavailable.

    :param user_input:            The user's self-info question.
    :param self_info_engine:      The live SelfInfoEngine instance.
    :param introspection_engine:  Optional IntrospectionEngine for live snapshot enrichment.
    :returns:                     Jarwis-prefixed response string.
    """
    try:
        # Build a live snapshot if we have the introspection engine
        snapshot = None
        if introspection_engine is not None:
            try:
                snapshot = introspection_engine.build_snapshot()
            except Exception:
                pass  # snapshot enrichment is optional

        if self_info_engine is None:
            self_info_engine = SelfInfoEngine()

        result = self_info_engine.answer(user_input, snapshot=snapshot)

        if not result.is_self_info or not result.context_block:
            return "Jarwis: I couldn't match that to verified self-information."

        # Check for self-info engine error (unknown capability)
        if result.error:
            log_event("debug", {"source": "self_info", "error": result.error})
            return "Jarwis: Verified self-information is temporarily unavailable."

        return "Jarwis: " + result.context_block

    except Exception as exc:
        log_event("error", {"source": "self_info", "error": str(exc)})
        return "Jarwis: Verified self-information is temporarily unavailable."


def _looks_like_directory_query(user_input: str) -> bool:
    """
    Heuristic: does this look like a question aimed at the loaded directory?

    When a directory session is active and the intent is unknown, we route to
    the directory handler unless the input is clearly a short command or
    confirmation word.  The directory handler itself gracefully handles inputs
    that turn out to have no relevant content.
    Never raises.
    """
    norm = user_input.strip().lower()

    # Very short inputs (single words / confirmations) are not directory queries
    if len(norm.split()) <= 1:
        return False

    # Explicit confirmation / control words — not directory queries
    _non_queries = frozenset({
        "yes", "no", "y", "n", "ok", "okay", "sure", "nope", "nah",
        "yep", "yeah", "go ahead", "skip", "cancel", "stop",
    })
    if norm in _non_queries:
        return False

    # Anything else with a directory active is treated as a directory query.
    # The handler will return "information not found" if content is missing.
    return True


def _handle_local_state_query(
    user_input: str,
    directory_control: "DirectoryControl",
) -> str:
    """
    Answer a LOCAL_STATE_QUERY directly from in-memory state.

    This function is called by the Intent Router short-circuit in
    process_input() and MUST NOT make any Brain calls.

    :param user_input:       The raw user query (for context only).
    :param directory_control: The active DirectoryControl instance.
    :returns:                Jarwis-prefixed response string.
    """
    active = directory_control.get_active_directory() if directory_control is not None else None
    if active:
        return f"Jarwis: Active workspace: {active}"
    return "Jarwis: No active workspace is currently set."


def _handle_directory_control(
    user_input: str,
    intent: str,
    directory_control: "DirectoryControl",
) -> str:
    """
    Dispatch Directory Control intents to the appropriate DirectoryControl method.

    Handles: dc_set_directory, dc_switch_directory, dc_show_directory,
    dc_clear_directory, dc_list_directory, dc_read_file, dc_create_file,
    dc_write_file, dc_append_file, dc_create_directory, dc_move, dc_copy,
    dc_rename.

    Never raises. Always returns a Jarwis-prefixed string.
    """
    _fc = get_fallback_classifier()

    try:
        raw = user_input.strip()
        text = raw.lower()

        # ---- Set workspace -----------------------------------------------
        if intent == "dc_set_directory":
            for prefix in ("work in ", "use ", "set workspace to ",
                           "set workspace ", "set directory to ", "workspace "):
                if text.startswith(prefix):
                    path = raw[len(prefix):].strip()
                    break
            else:
                path = raw
            try:
                resolved = directory_control.set_directory(path)
                return f"Jarwis: Active workspace set to: {resolved}"
            except WorkspaceNotFoundError as exc:
                return f"Jarwis: {exc}"

        # ---- Switch workspace --------------------------------------------
        if intent == "dc_switch_directory":
            for prefix in ("switch to ", "switch workspace to ",
                           "change directory to ", "change workspace to ", "cd "):
                if text.startswith(prefix):
                    path = raw[len(prefix):].strip()
                    break
            else:
                path = raw
            try:
                resolved = directory_control.switch_directory(path)
                return f"Jarwis: Switched active workspace to: {resolved}"
            except WorkspaceNotFoundError as exc:
                return f"Jarwis: {exc}"

        # ---- Show workspace ----------------------------------------------
        if intent == "dc_show_directory":
            active = directory_control.get_active_directory()
            if active is None:
                return "Jarwis: No active workspace is set."
            return f"Jarwis: Active workspace: {active}"

        # ---- Clear workspace ---------------------------------------------
        if intent == "dc_clear_directory":
            previous = directory_control.clear_directory()
            if previous is None:
                return "Jarwis: No active workspace was set."
            return f"Jarwis: Active workspace cleared: {previous}"

        # ---- List directory ----------------------------------------------
        if intent == "dc_list_directory":
            try:
                entries = directory_control.list_directory()
            except NoActiveWorkspaceError:
                return "Jarwis: No active workspace is set. Use 'work in <path>' to set one."
            if not entries:
                return "Jarwis: The active workspace is empty."
            lines = []
            for e in entries:
                marker = "[D]" if e["type"] == "dir" else "[F]"
                size = f"  ({e['size_bytes']} B)" if e["type"] == "file" else ""
                lines.append(f"  {marker} {e['name']}{size}")
            header = f"Contents of {directory_control.get_active_directory()}:"
            return "Jarwis: " + header + "\n" + "\n".join(lines)

        # ---- Read file ---------------------------------------------------
        if intent == "dc_read_file":
            for prefix in ("read file ", "open file "):
                if text.startswith(prefix):
                    path = raw[len(prefix):].strip()
                    break
            else:
                path = raw
            try:
                content = directory_control.read_file(path)
                return f"Jarwis: Contents of {path}:\n\n{content}"
            except (FileNotFoundInWorkspace, NotAFileError, DirectoryControlError) as exc:
                return f"Jarwis: {exc}"

        # ---- Create file -------------------------------------------------
        if intent == "dc_create_file":
            for prefix in ("create file ", "create a file ", "make file ", "make a file "):
                if text.startswith(prefix):
                    path = raw[len(prefix):].strip()
                    break
            else:
                path = raw
            try:
                result_path = directory_control.create_file(path)
                return f"Jarwis: File created: {result_path.name}"
            except DirectoryControlError as exc:
                return f"Jarwis: {exc}"

        # ---- Write file --------------------------------------------------
        if intent == "dc_write_file":
            # Pattern: "write <content> to <filename>" or "write to <filename>"
            # For natural language "write this into notes.txt" the content
            # comes from a subsequent turn; we accept a basic form here.
            for prefix in ("write to ", "write this to ", "write into ", "write this into "):
                if text.startswith(prefix):
                    path = raw[len(prefix):].strip()
                    break
            else:
                path = raw
            return (
                f"Jarwis: To write content to '{path}', please say:\n"
                "  write \"<content>\" to <filename>"
            )

        # ---- Append file -------------------------------------------------
        if intent == "dc_append_file":
            for prefix in ("append to ", "append this to ", "add to ", "add this to "):
                if text.startswith(prefix):
                    path = raw[len(prefix):].strip()
                    break
            else:
                path = raw
            return (
                f"Jarwis: To append content to '{path}', please say:\n"
                "  append \"<content>\" to <filename>"
            )

        # ---- Create directory --------------------------------------------
        if intent == "dc_create_directory":
            for prefix in ("create folder ", "create a folder ", "make folder ",
                           "make a folder ", "create directory ", "mkdir "):
                if text.startswith(prefix):
                    path = raw[len(prefix):].strip()
                    break
            else:
                path = raw
            try:
                result_path = directory_control.create_directory(path)
                return f"Jarwis: Folder created: {result_path.name}"
            except DirectoryControlError as exc:
                return f"Jarwis: {exc}"

        # ---- Move --------------------------------------------------------
        if intent == "dc_move":
            # "move <source> to <destination>"
            for prefix in ("move ", "mv "):
                if text.startswith(prefix):
                    rest = raw[len(prefix):]
                    break
            else:
                rest = raw
            if " to " in rest.lower():
                idx = rest.lower().index(" to ")
                source = rest[:idx].strip()
                destination = rest[idx + 4:].strip()
            else:
                return "Jarwis: Use: move <source> to <destination>"
            try:
                result_path = directory_control.move(source, destination)
                return f"Jarwis: Moved to: {result_path.name}"
            except DirectoryControlError as exc:
                return f"Jarwis: {exc}"

        # ---- Copy --------------------------------------------------------
        if intent == "dc_copy":
            for prefix in ("copy ", "cp "):
                if text.startswith(prefix):
                    rest = raw[len(prefix):]
                    break
            else:
                rest = raw
            if " to " in rest.lower():
                idx = rest.lower().index(" to ")
                source = rest[:idx].strip()
                destination = rest[idx + 4:].strip()
            else:
                return "Jarwis: Use: copy <source> to <destination>"
            try:
                result_path = directory_control.copy(source, destination)
                return f"Jarwis: Copied to: {result_path.name}"
            except DirectoryControlError as exc:
                return f"Jarwis: {exc}"

        # ---- Rename ------------------------------------------------------
        if intent == "dc_rename":
            for prefix in ("rename file ", "rename "):
                if text.startswith(prefix):
                    rest = raw[len(prefix):]
                    break
            else:
                rest = raw
            if " to " in rest.lower():
                idx = rest.lower().index(" to ")
                source = rest[:idx].strip()
                new_name = rest[idx + 4:].strip()
            else:
                return "Jarwis: Use: rename <source> to <new_name>"
            try:
                result_path = directory_control.rename(source, new_name)
                return f"Jarwis: Renamed to: {result_path.name}"
            except DirectoryControlError as exc:
                return f"Jarwis: {exc}"

    except Exception as exc:
        from diagnostics.analytics_logger import log_event
        log_event("error", {"source": "directory_control", "error": str(exc)})
        _fc_local = get_fallback_classifier()
        # Map known DirectoryControlError subclasses to typed AGError for specific messages
        if isinstance(exc, FileNotFoundInWorkspace):
            _typed = AGError(
                domain="DIRECTORY", code=DIR_NOT_FOUND,
                message=str(exc), context={"target": str(exc)},
            )
            return "Jarwis: " + _fc_local.get_typed_message(_typed)
        if isinstance(exc, NotAFileError):
            _typed = AGError(
                domain="DIRECTORY", code=DIR_NOT_A_FILE,
                message=str(exc), context={},
            )
            return "Jarwis: " + _fc_local.get_typed_message(_typed)
        if isinstance(exc, WorkspaceViolationError):
            _typed = AGError(
                domain="DIRECTORY", code=DIR_OUT_OF_WORKSPACE,
                message=str(exc), context={},
            )
            return "Jarwis: " + _fc_local.get_typed_message(_typed)
        return "Jarwis: " + _fc_local.get_message(CATEGORY_PROCESSING_FAILURE)

    return "Jarwis: " + _fc.get_message(CATEGORY_UNKNOWN_CAPABILITY)

def _handle_directory_reader(
    user_input: str,
    intent: str,
    directory_session: "DirectorySession",
) -> str:
    """
    Handle all directory-reader intents.

    read_directory  — load a new directory from the user-supplied path.
    clear_directory — remove the active directory from the session.
    query_directory — answer a question using the active directory context.

    Never raises. Always returns a Jarwis-prefixed string.

    :param user_input:         Raw user input text.
    :param intent:             One of "read_directory", "clear_directory",
                               "query_directory".
    :param directory_session:  The active DirectorySession for this conversation.
    :returns:                  Jarwis-prefixed response string.
    """
    try:
        # ---- Load a new directory ----------------------------------------
        if intent == "read_directory":
            text_lower = user_input.strip().lower()

            # Extract the path: strip known prefixes from the raw input
            raw = user_input.strip()
            _prefixes = (
                "read ",
                "load directory ",
                "open directory ",
                "index directory ",
                "scan directory ",
                "load folder ",
                "read folder ",
                "open folder ",
            )
            path = raw
            for _p in _prefixes:
                if raw.lower().startswith(_p):
                    path = raw[len(_p):].strip()
                    break

            if not path:
                return "Jarwis: Please provide a directory path. Example: Read D:\\AG\\docs"

            result = directory_session.set_directory(path)

            if not result.success:
                return f"Jarwis: Could not load directory. {result.error}"

            n = result.file_count
            if n == 0:
                return (
                    f"Jarwis: Directory loaded: {path}\n"
                    "No supported readable files were found. "
                    "Supported types: .txt .md .py .json .yaml .yml .ini .cfg .toml"
                )

            file_list = "\n".join(
                f"  {e.rel_path}" for e in result.files[:20]
            )
            more = f"\n  ... and {n - 20} more." if n > 20 else ""
            return (
                f"Jarwis: Directory loaded: {path}\n"
                f"{n} readable file(s) indexed.\n\n"
                f"{file_list}{more}\n\n"
                "You can now ask questions about the contents of this directory."
            )

        # ---- Clear the active directory ----------------------------------
        if intent == "clear_directory":
            if not directory_session.is_active:
                return "Jarwis: No directory is currently loaded."
            prev = directory_session.directory
            directory_session.clear()
            return f"Jarwis: Directory context cleared: {prev}"

        # ---- Query the active directory ----------------------------------
        # intent == "query_directory" (or any unknown intent when a directory is active)
        if not directory_session.is_active:
            return "Jarwis: No directory is currently loaded. Use: Read <path>"

        # List files if that is the question
        q_lower = user_input.strip().lower()
        _list_phrases = (
            "what files are there",
            "list files",
            "show files",
            "what files exist",
            "list the files",
            "show the files",
            "what is in this directory",
            "what is in the directory",
            "what files are in",
        )
        if any(q_lower.startswith(p) or q_lower == p for p in _list_phrases):
            return "Jarwis: " + directory_session.reader.list_files()

        # Detect if the user is explicitly scoping to the directory
        # ("according to the directory", "based on the files", etc.)
        _explicit_scope = (
            "according to the directory",
            "according to the files",
            "based on the directory",
            "based on the files",
            "from the directory",
            "from the files",
            "in the directory",
            "in these files",
            "in the files",
        )
        is_explicitly_scoped = any(phrase in q_lower for phrase in _explicit_scope)

        # General question — retrieve relevant files and build context
        file_contents = directory_session.reader.read_relevant_files(user_input, max_files=10)

        if not file_contents:
            return (
                "Jarwis: The directory is loaded but contains no readable files "
                f"relevant to your question.\nDirectory: {directory_session.directory}"
            )

        context = build_context(
            file_contents,
            question=user_input,
            directory=directory_session.directory,
        )

        if context.startswith("No readable content"):
            return f"Jarwis: {context}"

        # Build the prompt, adjusting the instruction for explicitly-scoped queries
        if is_explicitly_scoped:
            prompt = (
                "You are answering a question using content from a directory "
                "that the user has explicitly provided.\n"
                "The user has specifically asked about content FROM this directory. "
                "Use ONLY the file contents shown below. "
                "If the answer is not present in the files, state clearly that "
                "the information was not found in the supplied directory. "
                "Do NOT substitute general knowledge.\n\n"
                f"{context}\n\n"
                f"User question: {user_input}"
            )
        else:
            prompt = build_prompt(context, user_input)

        return "Jarwis: " + safe_response(ask_brain(prompt))

    except Exception as exc:
        log_event("error", {"source": "directory_reader", "error": str(exc)})
        return "Jarwis: " + get_fallback_classifier().get_message(CATEGORY_PROCESSING_FAILURE)


def _handle_introspection(
    user_input: str,
    introspection_engine: IntrospectionEngine,
) -> str:
    """
    Build a RuntimeSnapshot and inject it as context into the cloud brain prompt.

    The brain receives real runtime facts — it never invents capabilities.

    :param user_input:           The user's introspection question.
    :param introspection_engine: The live IntrospectionEngine instance.
    :returns:                    Jarwis-prefixed response string.
    """
    try:
        snapshot = introspection_engine.build_snapshot()
        snap_dict = snapshot.to_dict()

        # Compact snapshot context for the prompt
        context_lines = [
            f"Identity: {snap_dict['identity']['name']} v{snap_dict['identity']['version']}",
            f"Overall health: {snap_dict['runtime_health']['overall']}",
            f"Connected subsystems ({snap_dict['runtime_health']['healthy_count']} online): "
            + (", ".join(snap_dict["connected_subsystems"]) or "none"),
        ]
        if snap_dict["disconnected_subsystems"]:
            context_lines.append(
                "Offline/degraded: " + ", ".join(snap_dict["disconnected_subsystems"])
            )
        if snap_dict["brains"]:
            context_lines.append(
                "Brain subsystems: "
                + ", ".join(f"{k}={v}" for k, v in snap_dict["brains"].items())
            )
        if snap_dict["capabilities"]:
            context_lines.append(
                "Capabilities: " + "; ".join(snap_dict["capabilities"][:12])
            )
        if snap_dict["limitations"]:
            context_lines.append(
                "Limitations: " + "; ".join(snap_dict["limitations"][:8])
            )
        loaded_project = sorted([
            m for m in snap_dict["loaded_modules"]
            if not m.startswith("_")
            and "." not in m
            and m not in (
                "sys", "os", "re", "json", "time", "uuid", "abc",
                "copy", "enum", "math", "logging", "datetime",
                "typing", "dataclasses", "collections",
            )
        ])[:20]
        if loaded_project:
            context_lines.append("Loaded project modules: " + ", ".join(loaded_project))

        snapshot_context = "\n".join(context_lines)
        introspection_prompt = (
            "You are answering a question about your own runtime architecture.\n"
            "Use ONLY the following verified runtime facts. "
            "Do NOT invent capabilities or modules.\n\n"
            f"RUNTIME SNAPSHOT:\n{snapshot_context}\n\n"
            f"User question: {user_input}"
        )

        reply = safe_response(ask_brain(introspection_prompt))
        return "Jarwis: " + reply

    except Exception as exc:
        log_event("error", {"source": "introspection", "error": str(exc)})
        return "Jarwis: Runtime introspection unavailable. " + safe_response(ask_brain(user_input))


def store_last_answer(category: str, memory: dict, last_brain_answer: dict) -> None:
    summary = generate_memory_summary(
        last_brain_answer["question"],
        last_brain_answer["answer"]
    )

    fallback_key = last_brain_answer["question"].lower().replace("?", "").strip() or "unnamed_entry"
    key, value = _extract_key_value(summary, fallback_key, last_brain_answer["answer"])

    memory[category][key] = value
    save_memory(memory)
    log_event("storage_operation", {"operation": "store_answer", "category": category, "key": key})

    print(f"Jarwis: Stored '{key}' under {category}.")


def store_discussion(category: str, memory: dict, working_memory: WorkingMemory) -> None:
    # MIGRATION NOTE: discussion_topic() is not present on the new WorkingMemory
    # class. Kept as a method call under the new naming convention; see
    # "MISSING WORKING MEMORY METHODS" for the required addition.
    topic = working_memory.discussion_topic()

    # MIGRATION NOTE: old `working_memory.discussion_buffer` attribute replaced
    # with `working_memory.get_discussion_entries()`. This accessor does not
    # exist yet on the new class; see "MISSING WORKING MEMORY METHODS".
    combined_answer = " ".join(
        entry["answer"] for entry in working_memory.get_discussion_entries() if entry.get("answer")
    ).strip() or "No content captured."

    summary = generate_memory_summary(f"discussion about {topic}", combined_answer)

    fallback_key = topic.strip().lower().replace(" ", "_") or "discussion"
    key, value = _extract_key_value(summary, fallback_key, combined_answer)

    memory[category][key] = value
    save_memory(memory)
    log_event("storage_operation", {"operation": "store_discussion", "category": category, "key": key})

    print(f"Jarwis: Stored discussion under {category}.")


def main():
    memory = load_memory()
    tasks = load_tasks()
    startup()

    last_brain_answer = None
    pending_category = False
    pending_new_category_confirmation = False
    pending_new_category_name = None
    pending_shutdown_confirmation = False
    pending_shutdown_category = False
    pending_temporary_brain_action = None
    working_memory = WorkingMemory()
    # MIGRATION NOTE: old `working_memory.default_brain = ...` attribute
    # replaced with the new decision cache API.
    working_memory.cache_decision("default_brain", load_brain_mode())

    # ------------------------------------------------------------------ #
    # Runtime Integration — initialize all subsystems                     #
    # ------------------------------------------------------------------ #
    _conversation_manager = ConversationManager()
    _entity_tracker       = EntityTracker()
    _reference_resolver   = ReferenceResolver()
    _executive_layer      = ExecutiveLayer()
    _brain_dispatcher     = BrainDispatcher()
    _response_processor   = ResponseProcessor()
    _adaptive_cloud_brain = AdaptiveCloudBrain()
    _self_info_engine     = SelfInfoEngine()
    _directory_session    = DirectorySession()
    _directory_control    = DirectoryControl()

    # Build IntrospectionEngine with all subsystems registered
    # project_context.json is the authoritative version source used by the
    # self-info model.  Keep the runtime snapshot aligned with it.
    _project_context = load_project_context() or {}
    _project_version = str(_project_context.get("current_version", "UNKNOWN"))
    _builder_config = SnapshotBuilderConfig(
        identity_name="Jarwis",
        identity_version=_project_version.lstrip("vV"),
        module_prefix_filter="",
        configuration_summary={
            "brain_mode": load_brain_mode(),
            "memory_file": MEMORY_FILE,
            "task_file": TASK_FILE,
        },
    )
    _introspection_engine = IntrospectionEngine(builder_config=_builder_config)
    _registry = build_registry(
        conversation_manager=_conversation_manager,
        working_memory=working_memory,
        entity_tracker=_entity_tracker,
        reference_resolver=_reference_resolver,
        executive_layer=_executive_layer,
        brain_dispatcher=_brain_dispatcher,
        response_processor=_response_processor,
        adaptive_cloud_brain=_adaptive_cloud_brain,
        introspection_engine=_introspection_engine,
    )
    # Replace the engine's registry with the fully populated one
    _introspection_engine._registry = _registry

    # ---- Preprocessor Brain -------------------------------------------
    # Initialized AFTER IntrospectionEngine so it receives the live engine
    # as its introspection_source. DirectoryControl is passed directly so
    # capability checks reflect the real filesystem backend.
    _preprocessor_brain = PreprocessorBrain(
        directory_control=_directory_control,
        introspection_source=_introspection_engine,
    )

    while True:
        try:
            user_input = input("You: ")
        except EOFError:
            print("\nJarwis: Input stream closed. Shutting down.")
            print("Jarwis: Memory preserved.")
            break
        cleaned_input = user_input.strip().lower()

        if cleaned_input == "save last response":
            # MIGRATION NOTE: old `working_memory.has_unsaved_discussion()` /
            # `working_memory.discussion_buffer[-1]` kept under the new
            # method-call convention; both are missing from the new class.
            # See "MISSING WORKING MEMORY METHODS".
            if not working_memory.has_unsaved_discussion():
                print("Jarwis: There is no recent response to save.")
                continue

            last_brain_answer = working_memory.get_discussion_entries()[-1]
            pending_category = True
            print("Jarwis: Where should I store it?")

            for index, category in enumerate(memory.keys(), start=1):
                print(f"Jarwis: {index}. {category.capitalize()}")

            print("Jarwis: Or name a new division to create one.")
            continue

        confirmation_words = [
            "yes", "y", "yeah", "yep", "sure", "no", "n", "nah",
            "nope", "do it", "save it", "go ahead", "skip", "not now"
        ]

        if (
            cleaned_input in confirmation_words
            and not pending_category
            and not pending_new_category_confirmation
            and not pending_shutdown_confirmation
            and not pending_shutdown_category
            and not pending_temporary_brain_action
        ):
            print("Jarwis: No active decision pending.")
            continue

        if pending_temporary_brain_action:
            decision = interpret_storage_decision(user_input)

            if decision == "STORE":
                action = pending_temporary_brain_action
                previous_mode = load_brain_mode()
                # MIGRATION NOTE: old `working_memory.temporary_brain_active = True`
                # attribute replaced with a temporary fact.
                working_memory.add_fact("temporary_brain_active", True)

                if action["target"] == "cloud":
                    reply = safe_response(ask_cloud_direct(action["resolved_query"]))
                else:
                    reply = safe_response(ask_local(action["resolved_query"]))

                log_event("temporary_brain_used", {
                    "target": action["target"],
                    "resolved_query": action["resolved_query"],
                    "previous_mode": previous_mode
                })

                print(f"Jarwis: {reply}")
                print(f"Jarwis: (Temporary {action['target']} use only — default brain mode remains {previous_mode.upper()}.)")

                # Thread continuity fix: this turn must persist like any other.
                # intent="cloud_brain" so extract_topic won't fire on the raw
                # routing command — current_topic is deliberately left untouched.
                # MIGRATION NOTE: old free function `update_working_memory(...)`
                # replaced with a method call; the method itself is missing
                # from the new WorkingMemory class. See
                # "MISSING WORKING MEMORY METHODS".
                working_memory.update_working_memory(action["resolved_query"], reply, "cloud_brain")
                working_memory.add_to_discussion(action["resolved_query"], reply)

                # MIGRATION NOTE: old `working_memory.temporary_brain_active = False`
                # attribute replaced with a temporary fact.
                working_memory.add_fact("temporary_brain_active", False)
                # MIGRATION NOTE: old `working_memory.pending_actions = []`
                # (list of at most one item) replaced with the singular
                # pending-action API.
                working_memory.clear_pending_action()
                pending_temporary_brain_action = None
                continue

            if decision == "SKIP":
                # MIGRATION NOTE: old `working_memory.pending_actions = []`
                # replaced with the singular pending-action API.
                working_memory.clear_pending_action()
                pending_temporary_brain_action = None
                print("Jarwis: Cancelled. Default brain mode unchanged.")
                continue

            print("Jarwis: Should I proceed with that brain for this request, yes or no?")
            continue

        if pending_shutdown_confirmation:
            decision = interpret_storage_decision(user_input)

            if decision == "STORE":
                pending_shutdown_confirmation = False
                pending_shutdown_category = True

                print("Jarwis: Where should I store it?")

                for index, category in enumerate(memory.keys(), start=1):
                    print(f"Jarwis: {index}. {category.capitalize()}")

                print("Jarwis: Or name a new division to create one.")
                continue

            if decision == "SKIP":
                pending_shutdown_confirmation = False
                # MIGRATION NOTE: old `working_memory.clear_discussion()` kept
                # under the new method-call convention; missing from the new
                # class. See "MISSING WORKING MEMORY METHODS".
                working_memory.clear_discussion()
                print("Jarwis: Discussion discarded.")
                print("Jarwis: Shutting down.")
                print("Jarwis: Memory preserved.")
                break

            print("Jarwis: I could not tell whether you wanted that stored. Answer clearly, humanity permitting.")
            continue

        if pending_shutdown_category:
            category_type, category = interpret_category_decision(user_input, memory)

            if category is None:
                print("Jarwis: I could not identify that memory division. Use a number, category name, or ask me to create one.")
                continue

            if category_type in ("NEW", "NEW_PENDING_CONFIRMATION"):
                memory[category] = {}
                save_memory(memory)
                print(f"Jarwis: Created new memory division '{category}'.")

            store_discussion(category, memory, working_memory)
            working_memory.clear_discussion()

            print("Jarwis: Shutting down.")
            print("Jarwis: Memory preserved.")
            break

        if pending_new_category_confirmation:
            decision = interpret_storage_decision(user_input)

            if decision == "STORE":
                if not pending_new_category_name or not last_brain_answer:
                    pending_new_category_confirmation = False
                    pending_new_category_name = None
                    print("Jarwis: That confirmation no longer has anything to attach to.")
                    continue

                category = pending_new_category_name
                memory[category] = {}
                save_memory(memory)
                print(f"Jarwis: Created new memory division '{category}'.")

                store_last_answer(category, memory, last_brain_answer)

                pending_new_category_confirmation = False
                pending_new_category_name = None
                last_brain_answer = None
                continue

            if decision == "SKIP":
                pending_new_category_confirmation = False
                pending_new_category_name = None
                pending_category = True
                print("Jarwis: Understood. Where should I store it instead?")

                for index, category in enumerate(memory.keys(), start=1):
                    print(f"Jarwis: {index}. {category.capitalize()}")

                print("Jarwis: Or name a new division to create one.")
                continue

            print("Jarwis: I could not tell whether you wanted that created. Answer clearly, humanity permitting.")
            continue

        if pending_category:
            category_type, category = interpret_category_decision(user_input, memory)

            if category is None:
                print("Jarwis: I could not identify that memory division. Use a number, category name, or ask me to create one.")
                continue

            if category_type == "NEW":
                memory[category] = {}
                save_memory(memory)
                print(f"Jarwis: Created new memory division '{category}'.")

            elif category_type == "NEW_PENDING_CONFIRMATION":
                pending_category = False
                pending_new_category_confirmation = True
                pending_new_category_name = category
                print(f"Jarwis: Memory division '{category}' does not exist. Create it?")
                continue

            pending_category = False

            if not last_brain_answer:
                print("Jarwis: There is nothing pending to store anymore.")
                continue

            store_last_answer(category, memory, last_brain_answer)
            last_brain_answer = None
            continue

        log_event("user_input_received", {"text": user_input})
        quips_result = analyze_request(user_input, detect_intent)
        log_event("quips_request_analysis", quips_result.to_dict())

        action_result = analyze_action_pattern(user_input, working_memory)

        if action_result["is_action_request"] and not quips_result.is_compound:
            action = action_result["actions"][0]
            log_event("action_pattern_detected", {
                "action_type": action["action_type"],
                "operation": action["operation"]
            })

            if action["operation"] == "set_default_brain":
                print(set_brain_mode(action["target"]))
                # MIGRATION NOTE: old `working_memory.default_brain = ...`
                # attribute replaced with the decision cache API.
                working_memory.cache_decision("default_brain", action["target"])
                log_event("brain_mode_changed", {"mode": action["target"]})
                continue

            if action["operation"] == "temporary_brain_use":
                pending_temporary_brain_action = action
                # MIGRATION NOTE: old `working_memory.pending_actions = [action]`
                # (a list holding at most one item) replaced with the singular
                # pending-action API.
                working_memory.set_pending_action(action)
                scope_label = action["scope"].replace("_", " ")
                print(f"Jarwis: {action['target'].capitalize()} brain requested for this {scope_label}. Proceed?")
                continue

        original_input = user_input
        request_unit = quips_result.requests[0]
        # QUIPS always invokes detect_intent and preserves its result on the
        # unit.  For single requests this retains legacy routing unchanged;
        # deterministic ownership is applied in process_input from the unit.
        intent = request_unit.legacy_intent or detect_intent(user_input)
        brain_input = user_input

        if intent == "unknown" and request_unit.category in {
            RequestCategory.EXPLANATION,
            RequestCategory.CONVERSATION,
        }:
            resolved_input = resolve_followup(user_input, working_memory)
            if resolved_input == user_input:
                # MIGRATION NOTE: old free function `resolve_thread_reference(...)`
                # replaced with a method call; the method itself is missing
                # from the new WorkingMemory class. See
                # "MISSING WORKING MEMORY METHODS".
                resolved_input = working_memory.resolve_thread_reference(user_input)

            if resolved_input != user_input:
                brain_input = resolved_input
                intent = detect_intent(brain_input)
                # MIGRATION NOTE: old `working_memory.follow_up_depth += 1`
                # attribute replaced with a temporary fact counter.
                working_memory.add_fact(
                    "follow_up_depth",
                    working_memory.get_fact("follow_up_depth", 0) + 1
                )
                log_event("topic_continuation_detected", {
                    "original_input": original_input,
                    "resolved_input": brain_input,
                    "topic": working_memory.get_topic()
                })

        response = process_input(
            brain_input, memory, tasks, intent,
            introspection_engine=_introspection_engine,
            conversation_manager=_conversation_manager,
            entity_tracker=_entity_tracker,
            reference_resolver=_reference_resolver,
            executive_layer=_executive_layer,
            brain_dispatcher=_brain_dispatcher,
            response_processor=_response_processor,
            adaptive_cloud_brain=_adaptive_cloud_brain,
            self_info_engine=_self_info_engine,
            directory_session=_directory_session,
            directory_control=_directory_control,
            preprocessor_brain=_preprocessor_brain,
            quips_result=quips_result,
            request_unit=request_unit,
            working_memory=working_memory,
            semantic_reasoner=_semantic_interpretation_fallback,
        )
        response = get_persona_controller().apply_address(
            response, get_persona_controller().current_address,
        )

        if response == "shutdown":
            if working_memory.has_unsaved_discussion():
                topic = working_memory.discussion_topic()
                pending_shutdown_confirmation = True
                print(f"Jarwis: I have an unsaved discussion about {topic}. Store it before shutdown?")
                continue

            print("Jarwis: Shutting down.")
            print("Jarwis: Memory preserved.")
            break

        print(response)

        answer_text = response.replace("Jarwis: ", "", 1) if response.startswith("Jarwis: ") else response
        # MIGRATION NOTE: old free function `update_working_memory(...)`
        # replaced with a method call; missing from the new class. See
        # "MISSING WORKING MEMORY METHODS".
        working_memory.update_working_memory(original_input, answer_text, intent)

        # ---- IP4: Entity tracking per turn --------------------------------
        # Advance the turn counter, extract entities from both sides of the
        # exchange, then close the turn.  ConversationManager also gets the
        # Q&A pair so its thread summary and reference resolution stay current.
        _entity_tracker.advance_turn()
        _extract_entities_from_text(original_input, _entity_tracker)
        _extract_entities_from_text(answer_text, _entity_tracker)
        _entity_tracker.end_turn()
        _conversation_manager.add_turn("user", original_input)
        _conversation_manager.add_turn("assistant", answer_text)
        # Sync the current topic into the conversation manager
        topic = working_memory.get_topic()
        if topic:
            _conversation_manager.set_topic(topic)
        # -------------------------------------------------------------------

        if intent in ["unknown", "cloud_brain", "recall"]:
            if answer_text == BRAIN_FALLBACK:
                log_event("error", {"input": original_input, "intent": intent})
            else:
                log_event("brain_response_completed", {
                    "intent": intent,
                    "topic": working_memory.get_topic()
                })

        if response.startswith("Jarwis: ") and intent in ["unknown", "cloud_brain"]:
            working_memory.add_to_discussion(original_input, answer_text)


if __name__ == "__main__":
    main()
