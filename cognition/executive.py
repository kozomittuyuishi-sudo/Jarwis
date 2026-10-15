import re
from typing import Dict, Tuple, Optional, Any
from brains.brain import ask_brain


class ExecutiveState:
    def __init__(self):
        self.goal = None
        self.intent = None
        self.plan = None

        self.use_local_brain = False
        self.use_cloud_brain = False

        self.use_memory = False
        self.use_working_memory = False
        self.use_project_context = False
        self.use_tasks = False

        self.requires_clarification = False
        self.store_after_response = False

        self.confidence = 1.0

def clean_category_name(name: Any) -> str:
    name = str(name).strip().lower()
    name = name.replace(" ", "_")
    name = "".join(char for char in name if char.isalnum() or char == "_")
    return name


def interpret_storage_decision(user_input: Any) -> str:
    text = str(user_input).strip().lower()

    quick_store = ["yes", "y", "yeah", "yep", "sure", "store", "save", "save it", "store it", "go ahead", "do it"]
    quick_skip = ["no", "n", "nah", "nope", "skip", "not now", "dont", "don't", "leave it", "ignore it"]

    if text in quick_store:
        return "STORE"

    if text in quick_skip:
        return "SKIP"

    prompt = f"""
You are AG's Executive Layer.

AG asked:
"Should I store this for future access?"

User replied:
{text}

Classify the user's intention.

Return ONLY one word:
STORE
SKIP
CLARIFY
"""

    raw = ask_brain(prompt)
    decision = (raw or "").strip().upper()

    if "STORE" in decision:
        return "STORE"

    if "SKIP" in decision:
        return "SKIP"

    return "CLARIFY"


def interpret_category_decision(user_input: Any, memory: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    """
    Returns one of:
      ("EXISTING", category)
      ("NEW_PENDING_CONFIRMATION", category)  -> AG must ask before creating
      ("NEW", category)                       -> explicit request, create immediately
      ("UNKNOWN", None)
    """
    text = str(user_input).strip().lower()

    # Numbers are assigned in the same order Jarwis.py prints the menu,
    # so they always line up with what the user was shown.
    numbered_categories = {
        str(index): category
        for index, category in enumerate(memory.keys(), start=1)
    }

    if text in numbered_categories:
        return ("EXISTING", numbered_categories[text])

    for category in memory:
        if text == category:
            return ("EXISTING", category)

    # Explicit creation request -> create immediately, no confirmation needed.
    # e.g. "create a new division called robotics", "make a category named notes",
    # "create one with name as misc"
    create_match = re.search(
        r"\b(?:create|make|new)\b.*?\b(?:called|named|as)\b\s+([a-z0-9_ ]+)",
        text
    )
    if create_match:
        category = clean_category_name(create_match.group(1))
        if category:
            return ("NEW", category)

    # "put/store/save it in/under X" -> only ask for confirmation, don't assume.
    store_match = re.search(
        r"\b(?:put|store|save)\b.*?\b(?:in|under)\b\s+([a-z0-9_ ]+)",
        text
    )
    if store_match:
        category = clean_category_name(store_match.group(1))
        if category:
            if category in memory:
                return ("EXISTING", category)
            return ("NEW_PENDING_CONFIRMATION", category)

    # Bare single-word reply that isn't an existing category
    # (e.g. user just types "misc") -> ask for confirmation.
    if len(text.split()) == 1:
        category = clean_category_name(text)
        if category:
            return ("NEW_PENDING_CONFIRMATION", category)

    existing_categories = ", ".join(memory.keys())

    prompt = f"""
You are AG's Executive Layer.

AG is asking the user where to store a memory entry.

Existing memory divisions:
{existing_categories}

User replied:
{text}

Decide whether the user wants to use an existing memory division,
create a new memory division, or if the answer is unclear.

Return ONLY one of these formats:

USE:<existing_category>
CREATE:<new_category>
UNKNOWN

Strict rules:
- Use USE only if the user explicitly mentions an existing category.
- Never guess based on meaning.
- Never substitute a related category.
- If the user mentions a category that does not exist, return CREATE:<that_category>.
- If the user says "robotics", do NOT choose "vehicles" unless "vehicles" was explicitly mentioned.
- If the user invents a category name that is not in the existing list, never map it to an existing category. Always CREATE it.
- Category names must be lowercase with underscores.
- Do not explain.

Examples:

User: put it in projects
Response: USE:projects

User: put it in robotics
Response: CREATE:robotics

User: create one with name as misc
Response: CREATE:misc

User: store it in misc
Response: CREATE:misc if misc does not exist, otherwise USE:misc
"""

    raw = ask_brain(prompt)
    decision = (raw or "").strip()

    if decision.startswith("USE:"):
        category = clean_category_name(decision.replace("USE:", "", 1))

        if category in memory:
            return ("EXISTING", category)

        return ("UNKNOWN", None)

    if decision.startswith("CREATE:"):
        category = clean_category_name(decision.replace("CREATE:", "", 1))

        if category:
            # LLM-inferred creation is not explicit enough to skip confirmation.
            return ("NEW_PENDING_CONFIRMATION", category)

    return ("UNKNOWN", None)

def build_execution_plan(user_input: Any, intent: Any) -> Dict[str, Any]:
    plan: Dict[str, Any] = {
        "goal": None,
        "intent": intent,
        "action": "respond",

        "use_brain": False,
        "use_cloud": False,
        "use_memory": False,
        "use_tasks": False,
        "use_project_context": False,
        "use_working_memory": False,

        "requires_clarification": False,
        "store_after_response": False,
        "confidence": 1.0
    }

    if intent in ["remember", "recall", "show_memory", "open_memory_file"]:
        plan["goal"] = "handle_memory_request"
        plan["action"] = "memory_operation"
        plan["use_memory"] = True
        plan["store_after_response"] = False

    elif intent in [
        "add_task",
        "show_tasks",
        "complete_task",
        "show_completed_tasks"
    ]:
        plan["goal"] = "handle_task_request"
        plan["action"] = "task_operation"
        plan["use_tasks"] = True
        plan["store_after_response"] = False

    elif intent in [
        "project_status",
        "project_name",
        "next_step",
        "completed_milestones",
        "current_version"
    ]:
        plan["goal"] = "handle_project_request"
        plan["action"] = "project_operation"
        plan["use_project_context"] = True
        plan["store_after_response"] = False

    elif intent in [
        "set_brain_local",
        "set_brain_cloud",
        "set_brain_auto",
        "brain_status"
    ]:
        plan["goal"] = "handle_brain_mode_request"
        plan["action"] = "brain_control"
        plan["store_after_response"] = False

    elif intent == "cloud_brain":
        plan["goal"] = "force_cloud_response"
        plan["action"] = "brain_response"
        plan["use_brain"] = True
        plan["use_cloud"] = True
        plan["store_after_response"] = True

    elif intent == "unknown":
        plan["goal"] = "answer_general_query"
        plan["action"] = "brain_response"
        plan["use_brain"] = True
        plan["use_working_memory"] = True
        plan["store_after_response"] = True

    elif intent == "greeting":
        plan["goal"] = "greet_user"
        plan["action"] = "direct_response"
        plan["store_after_response"] = False

    elif intent == "shutdown":
        plan["goal"] = "shutdown_ag"
        plan["action"] = "shutdown"
        plan["store_after_response"] = False

    else:
        plan["goal"] = "unclear_request"
        plan["requires_clarification"] = True
        plan["confidence"] = 0.4
        plan["store_after_response"] = False

    return plan


def analyze_context(user_input: Any, intent: Any) -> Dict[str, Any]:
    """
    Context Analysis: figures out what kind of request this is,
    using plain text checks only (no LLM). Runs before Planning.
    """
    text = str(user_input).strip().lower()
    word_count = len(text.split())

    context: Dict[str, Any] = {
        "topic_type": "general",
        "user_state": "neutral",
        "request_type": "statement",
        "complexity": "simple",
        "is_follow_up": False,
        "needs_clarification": False,
        "likely_storage_value": False
    }

    # --- request_type ---
    command_starters = [
        "add", "remember", "store", "save", "delete", "remove", "set",
        "switch", "show", "list", "complete", "mark", "open", "create"
    ]
    if text.endswith("?"):
        context["request_type"] = "question"
    elif any(text.startswith(word) for word in command_starters):
        context["request_type"] = "command"
    else:
        context["request_type"] = "statement"

    # --- topic_type: trust intent first, fall back to text patterns ---
    if intent == "greeting":
        context["topic_type"] = "casual_greeting"

    elif intent == "shutdown":
        context["topic_type"] = "shutdown"

    elif intent in ["remember", "recall", "show_memory", "open_memory_file"]:
        context["topic_type"] = "memory_storage"

    elif intent in ["add_task", "show_tasks", "complete_task", "show_completed_tasks"]:
        context["topic_type"] = "task_management"

    elif intent in [
        "project_status", "project_name", "next_step",
        "completed_milestones", "current_version"
    ]:
        context["topic_type"] = "project_planning"

    else:
        code_keywords = [
            "error", "traceback", "bug", "exception", "function",
            "def ", "class ", "import ", "syntax", "compile", "debug",
            "code", "script", "variable"
        ]
        concept_keywords = [
            "explain", "what is", "what are", "how does", "why does",
            "define", "meaning of"
        ]
        confusion_keywords = [
            "confused", "i don't understand", "i dont understand",
            "what do you mean", "unclear", "makes no sense", "huh",
            "not sure what", "lost"
        ]

        if any(keyword in text for keyword in code_keywords):
            context["topic_type"] = "code_debugging"
        elif any(keyword in text for keyword in confusion_keywords):
            context["topic_type"] = "confusion"
        elif any(keyword in text for keyword in concept_keywords):
            context["topic_type"] = "conceptual_explanation"

    # --- user_state ---
    confusion_keywords = [
        "confused", "i don't understand", "i dont understand",
        "what do you mean", "unclear", "makes no sense", "huh",
        "not sure what", "lost"
    ]
    if any(keyword in text for keyword in confusion_keywords):
        context["user_state"] = "confused"

    # --- is_follow_up ---
    follow_up_starters = ["also", "and", "then", "what about", "additionally", "but", "so"]
    follow_up_refs = ["that", "it", "this", "those", "earlier", "before", "again"]
    if any(text.startswith(word) for word in follow_up_starters):
        context["is_follow_up"] = True
    elif word_count <= 6 and any(ref in text.split() for ref in follow_up_refs):
        context["is_follow_up"] = True

    # --- complexity ---
    if word_count > 12 or text.count(",") >= 2 or " and " in text:
        context["complexity"] = "complex"

    # --- needs_clarification ---
    if context["user_state"] == "confused":
        context["needs_clarification"] = True
    elif word_count <= 2 and context["request_type"] == "question":
        context["needs_clarification"] = True

    # --- likely_storage_value ---
    if (
        context["topic_type"] in ["conceptual_explanation", "code_debugging", "project_planning"]
        and context["request_type"] == "question"
    ):
        context["likely_storage_value"] = True

    return context