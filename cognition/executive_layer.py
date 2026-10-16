"""
executive_layer.py
==================
AG Executive Layer (Stage 5 - Phase A)

Role:
- High-level decision manager for Storage and Category routing.
- Validates QUIPS request structures and builds an inspectable ANVIL action plan.
- Consumes processed responses / cognitive payloads and decides:
  1. Storage Routing (Long-term memory.json vs Session Memory vs Temporary Context).
  2. Category Routing (Structured Categories vs Dynamic Categories created on the fly).
  3. Task & Objective persistence triggers.

Boundaries:
- Does not modify LLM reasoning or prompt text directly.
- Directs state mutation decisions to lower-level storage and memory handlers.
"""

from datetime import datetime, timezone
from dataclasses import asdict, dataclass, field
import re
from typing import Any, Dict, Optional

from quips import extract_connectors

# Type alias: return type for evaluate_category_decision
Tuple_Cat_Decision = Any


@dataclass
class ActionStep:
    """Validated, inspectable action-plan entry derived from a QUIPS unit."""

    step_id: str
    original_segment: str
    operation: str
    arguments: Dict[str, Any]
    dependencies: list[str] = field(default_factory=list)
    required_context: list[str] = field(default_factory=list)
    connectors: list[str] = field(default_factory=list)
    expected_output: Optional[str] = None
    constraints: list[str] = field(default_factory=list)
    validation_findings: list[str] = field(default_factory=list)
    execution_status: str = "PENDING"
    actual_result: Any = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ActionPlan:
    """ANVIL plan and diagnostics; creating one never executes an action."""

    original_request: str
    connectors: list[str] = field(default_factory=list)
    steps: list[ActionStep] = field(default_factory=list)
    valid: bool = True
    status: str = "PENDING"
    diagnostics: list[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def update_step(
        self,
        step_id: str,
        *,
        status: str,
        actual_result: Any = None,
        error: Optional[str] = None,
    ) -> None:
        for step in self.steps:
            if step.step_id == step_id:
                step.execution_status = status
                step.actual_result = actual_result
                step.error = error
                return


class ExecutiveDecision:
    """Standard payload containing operational routing decisions made by Executive Layer."""

    def __init__(

        self,
        should_persist_long_term: bool = False,
        target_category: Optional[str] = None,
        is_dynamic_category: bool = False,
        task_update: Optional[Dict[str, Any]] = None,
        memory_entry: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.should_persist_long_term = should_persist_long_term
        self.target_category = target_category
        self.is_dynamic_category = is_dynamic_category
        self.task_update = task_update
        self.memory_entry = memory_entry
        self.metadata = metadata or {}
        self.decided_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "should_persist_long_term": self.should_persist_long_term,
            "target_category": self.target_category,
            "is_dynamic_category": self.is_dynamic_category,
            "task_update": self.task_update,
            "memory_entry": self.memory_entry,
            "metadata": self.metadata,
            "decided_at": self.decided_at
        }


class ExecutiveLayer:
    """Stage 5 Manager handling Storage & Category Decisions."""

    # Default structured categories in AG
    STRUCTURED_CATEGORIES = {"core_facts", "project_metadata", "task_rules", "user_preferences"}

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def analyze_action_plan(self, quips_result: Any) -> ActionPlan:
        """Validate QUIPS output and convert its units into an ANVIL plan."""
        original = getattr(quips_result, "original_text", "")
        original_text = original if isinstance(original, str) else ""
        plan = ActionPlan(
            original_request=original_text,
            connectors=self._extract_connectors(original_text),
        )
        requests = getattr(quips_result, "requests", None)
        if not isinstance(requests, (list, tuple)):
            plan.valid = False
            self._plan_diagnostic(
                plan, "structure", None, "QUIPS result has no valid request list.",
                "Re-analyze the original request.",
            )
            return plan
        if not requests:
            plan.valid = False
            self._plan_diagnostic(
                plan, "parsing", None, "QUIPS produced no request units.",
                "Provide a non-empty request or ask for clarification.",
            )
            return plan

        index_to_step: Dict[int, str] = {}
        step_to_index: Dict[str, int] = {}
        unit_dependencies: Dict[int, list[int]] = {}
        seen_indices: set[int] = set()
        for ordinal, unit in enumerate(requests):
            index = getattr(unit, "index", None)
            if not isinstance(index, int) or index < 0 or index in seen_indices:
                plan.valid = False
                self._plan_diagnostic(
                    plan, "structure", None,
                    f"Request unit {ordinal} has a missing, invalid, or duplicate index.",
                    "Re-run request analysis before execution.",
                )
                continue
            seen_indices.add(index)
            step_id = f"step-{index + 1}"
            index_to_step[index] = step_id
            step_to_index[step_id] = index
            unit_dependencies[index] = []
            text = getattr(unit, "text", None)
            if not isinstance(text, str) or not text.strip():
                plan.valid = False
                self._plan_diagnostic(
                    plan, "structure", step_id, "Request segment is empty or malformed.",
                    "Clarify or restate this action.",
                )
                continue

            understanding = getattr(unit, "understanding", None)
            findings: list[str] = []
            if getattr(unit, "processing_status", "") == "CLASSIFICATION_FAILED":
                findings.append(str(getattr(unit, "result_state", None) or "Classification failed."))
                self._plan_diagnostic(
                    plan, "classification", step_id, findings[-1],
                    "Clarify the intended operation; no operation was inferred.",
                )
            missing = tuple(getattr(understanding, "missing_information", ()) or ())
            if missing:
                plan.valid = False
                findings.extend(f"Missing required information: {item}." for item in missing)
                self._plan_diagnostic(
                    plan, "validation", step_id,
                    "Required information is missing: " + ", ".join(missing) + ".",
                    "Supply the missing target or argument before execution.",
                )
            ambiguous = (
                getattr(unit, "confidence", None) == "low"
                or bool(getattr(understanding, "semantic_reasoning_required", False))
            )
            if ambiguous:
                plan.valid = False
                findings.append("Request structure or meaning is ambiguous.")
                self._plan_diagnostic(
                    plan, "ambiguity", step_id, findings[-1],
                    "Ask a clarifying question; do not guess a target.",
                )

            intent = str(getattr(unit, "intent", "UNKNOWN") or "UNKNOWN")
            operation = str(
                getattr(understanding, "action", None) or intent
            ).upper()
            supported = {
                "CREATE_FILE", "CREATE_FOLDER", "WRITE_FILE", "APPEND_FILE",
                "SEARCH_FILES", "LIST_DIRECTORY", "NAVIGATE_DIRECTORY",
                "READ_FILE", "EXPLAIN_FILE", "DELETE_TARGET", "RENAME_TARGET",
                "MOVE_TARGET", "COPY_TARGET", "OPEN_TARGET", "EXPLAIN_RESULT",
                "IDENTITY", "CAPABILITIES", "LIMITATIONS", "ARCHITECTURE",
                "OPERATIONAL", "BRAINS", "GENERAL_REASONING",
                "CONVERSATION_CONTEXT", "MEMORY", "TASK", "PROJECT",
                "WORKSPACE_QUERY", "WORKSPACE_SET",
            }
            if intent not in supported and operation not in supported:
                findings.append(f"Unsupported operation: {operation}.")
                self._plan_diagnostic(
                    plan, "routing", step_id, findings[-1],
                    "Use a supported deterministic operation or clarify the request.",
                )

            target_reference = getattr(unit, "target_reference", None)
            destination_reference = getattr(unit, "destination_reference", None)
            arguments = {
                "target": getattr(target_reference, "original_target", None),
                "destination": getattr(destination_reference, "original_target", None),
                "content": getattr(understanding, "content", None),
                "intent": intent,
            }
            constraints = self._extract_constraints(text)
            dependencies = getattr(unit, "depends_on", ()) or ()
            if not isinstance(dependencies, (list, tuple, set)):
                plan.valid = False
                dependencies = ()
                self._plan_diagnostic(
                    plan, "dependencies", step_id, "Dependency references are malformed.",
                    "Re-run request analysis before execution.",
                )
            unit_dependencies[index] = list(dependencies)
            plan.steps.append(ActionStep(
                step_id=step_id,
                original_segment=str(
                    getattr(unit, "original_segment", None) or text
                ),
                operation=operation,
                arguments=arguments,
                required_context=list(
                    getattr(understanding, "context_requirements", ()) or ()
                ),
                connectors=list(getattr(unit, "connectors", ()) or ()),
                expected_output=getattr(understanding, "desired_outcome", None),
                constraints=constraints,
                validation_findings=findings,
            ))

        for step in plan.steps:
            unit_index = step_to_index[step.step_id]
            dependencies = unit_dependencies.get(unit_index, [])
            for dependency in dependencies:
                if not isinstance(dependency, int) or dependency not in index_to_step:
                    plan.valid = False
                    self._plan_diagnostic(
                        plan, "dependencies", step.step_id,
                        f"Dependency reference {dependency!r} does not exist.",
                        "Correct the dependency before execution.",
                    )
                    continue
                step.dependencies.append(index_to_step[dependency])

        if self._has_dependency_cycle(plan.steps):
            plan.valid = False
            self._plan_diagnostic(
                plan, "dependencies", None, "The action dependency graph contains a cycle.",
                "Correct the dependency graph before execution.",
            )
        if any(
            item.get("stage") in {"structure", "parsing", "classification", "dependencies"}
            for item in plan.diagnostics
        ):
            plan.valid = False
        return plan

    @staticmethod
    def _plan_diagnostic(
        plan: ActionPlan,
        stage: str,
        step_id: Optional[str],
        message: str,
        recovery: str,
    ) -> None:
        plan.diagnostics.append({
            "stage": stage,
            "step_id": step_id,
            "message": message,
            "recovery": recovery,
        })

    @staticmethod
    def _has_dependency_cycle(steps: list[ActionStep]) -> bool:
        dependencies = {step.step_id: set(step.dependencies) for step in steps}
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(step_id: str) -> bool:
            if step_id in visiting:
                return True
            if step_id in visited:
                return False
            visiting.add(step_id)
            if any(
                dependency in dependencies and visit(dependency)
                for dependency in dependencies[step_id]
            ):
                return True
            visiting.remove(step_id)
            visited.add(step_id)
            return False

        return any(visit(step_id) for step_id in dependencies)

    @staticmethod
    def _extract_constraints(text: str) -> list[str]:
        patterns = (
            r"\bwithout\s+(?:modifying|changing|deleting)\b[^,.!?]*",
            r"\bonly\b[^,.!?]*",
            r"\bdo\s+not\b[^,.!?]*",
            r"\bdon't\b[^,.!?]*",
        )
        return list(dict.fromkeys(
            match.group(0).strip()
            for pattern in patterns
            for match in re.finditer(pattern, text, re.IGNORECASE)
        ))

    @staticmethod
    def _extract_connectors(text: str) -> list[str]:
        return list(extract_connectors(text))

    def evaluate_storage_decision(self, user_input: str, processed_content: str, mode: str) -> bool:
        """
        Determines whether an interaction should be stored in long-term memory (memory.json).
        """
        # Triggers for long-term persistence:
        # 1. Action mode or explicitly marked persistent facts/tasks
        # 2. Key phrases like "remember", "save this", "project rule"
        if mode in ("ACTION", "PLANNING"):
            return True

        text_lower = (user_input + " " + processed_content).lower()
        persistence_triggers = ["remember", "save this", "always", "never forget", "objective:", "rule:"]
        
        return any(trigger in text_lower for trigger in persistence_triggers)

    def evaluate_category_decision(self, category_candidate: Optional[str]) -> Tuple_Cat_Decision:
        """
        Determines whether a category is structured or requires a dynamic category creation on the fly.
        """
        if not category_candidate or not category_candidate.strip():
            return "uncategorized", False

        clean_cat = category_candidate.strip().lower().replace(" ", "_")

        if clean_cat in self.STRUCTURED_CATEGORIES:
            return clean_cat, False
        else:
            # Dynamic category created on the fly
            return clean_cat, True

    def process_execution_cycle(
        self,
        user_input: str,
        processed_response: Any,
        cog_state: Optional[Dict[str, Any]] = None
    ) -> ExecutiveDecision:
        """
        Consumes cognitive state & processed output to generate a consolidated ExecutiveDecision.
        """
        cog_state = cog_state or {}
        
        # Extract response content & mode safely
        if hasattr(processed_response, "sanitized_content"):
            content = processed_response.sanitized_content
            mode = getattr(processed_response, "mode", "QUERY")
        elif isinstance(processed_response, dict):
            content = processed_response.get("sanitized_content", "")
            mode = processed_response.get("mode", "QUERY")
        else:
            content = str(processed_response)
            mode = "QUERY"

        # 1. Storage Decision
        persist = self.evaluate_storage_decision(user_input, content, mode)

        # 2. Category Decision
        raw_topic = cog_state.get("topic") or cog_state.get("category")
        cat_name, is_dynamic = self.evaluate_category_decision(raw_topic)

        # 3. Construct Memory Entry if long-term persistence triggered
        memory_entry = None
        if persist:
            memory_entry = {
                "input": user_input,
                "summary": content[:200],
                "category": cat_name,
                "is_dynamic_category": is_dynamic,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        return ExecutiveDecision(
            should_persist_long_term=persist,
            target_category=cat_name,
            is_dynamic_category=is_dynamic,
            memory_entry=memory_entry,
            metadata={"processed_mode": mode}
        )
