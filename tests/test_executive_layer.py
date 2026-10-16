"""
test_executive_layer.py
========================
Unit tests for AG Executive Layer (Stage 5 - Phase A).
Tests Storage Decisions (long-term memory.json trigger) and
Category Decisions (Structured vs Dynamic on the fly).
"""

from cognition.executive_layer import ExecutiveLayer, ExecutiveDecision
from dataclasses import replace
from types import SimpleNamespace

from quips import analyze_request


def test_storage_decision_triggers():
    exec_layer = ExecutiveLayer()

    # ACTION mode -> should persist long-term
    assert exec_layer.evaluate_storage_decision("Run script", "Executed", "ACTION") is True

    # QUERY mode without trigger phrase -> temporary
    assert exec_layer.evaluate_storage_decision("What is 2+2?", "4", "QUERY") is False

    # Explicit trigger word in QUERY mode -> should persist
    assert exec_layer.evaluate_storage_decision("Remember that AG is v0.23", "Saved", "QUERY") is True
    print("✓ test_storage_decision_triggers passed")


def test_category_decision_structured_vs_dynamic():
    exec_layer = ExecutiveLayer()

    # Known structured category
    cat_structured, is_dyn1 = exec_layer.evaluate_category_decision("core_facts")
    assert cat_structured == "core_facts"
    assert is_dyn1 is False

    # Dynamic category created on the fly
    cat_dynamic, is_dyn2 = exec_layer.evaluate_category_decision("quantum_physics_notes")
    assert cat_dynamic == "quantum_physics_notes"
    assert is_dyn2 is True
    print("✓ test_category_decision_structured_vs_dynamic passed")


def test_process_execution_cycle():
    exec_layer = ExecutiveLayer()
    
    mock_processed = {
        "sanitized_content": "Rule: AG uses zero-dependency modules.",
        "mode": "PLANNING"
    }
    cog_state = {"topic": "system_design_rules"}

    decision = exec_layer.process_execution_cycle(
        user_input="Define system design rules",
        processed_response=mock_processed,
        cog_state=cog_state
    )

    assert isinstance(decision, ExecutiveDecision)
    assert decision.should_persist_long_term is True
    assert decision.target_category == "system_design_rules"
    assert decision.is_dynamic_category is True
    assert decision.memory_entry is not None
    print("✓ test_process_execution_cycle passed")


def test_anvil_builds_inspectable_plan_without_execution():
    analysis = analyze_request(
        "Create folder Experiments, then create notes.txt inside it, "
        "write 'Hello and then goodbye' into the file, and read it back."
    )

    plan = ExecutiveLayer().analyze_action_plan(analysis)

    assert plan.valid
    assert [step.step_id for step in plan.steps] == [
        "step-1", "step-2", "step-3", "step-4",
    ]
    assert plan.steps[2].arguments["content"] == "Hello and then goodbye"
    assert plan.connectors == ["then", "inside", "and"]
    assert plan.steps[1].dependencies == ["step-1"]
    assert plan.steps[2].dependencies == ["step-2"]
    assert plan.steps[3].dependencies == ["step-3"]
    assert all(step.execution_status == "PENDING" for step in plan.steps)


def test_anvil_reports_empty_and_malformed_quips_structures():
    executive = ExecutiveLayer()
    empty = executive.analyze_action_plan(analyze_request(""))
    malformed = executive.analyze_action_plan(SimpleNamespace(original_text="request"))

    assert empty.valid is False
    assert empty.diagnostics[0]["stage"] == "parsing"
    assert malformed.valid is False
    assert malformed.diagnostics[0]["stage"] == "structure"


def test_anvil_detects_missing_arguments_and_invalid_dependencies():
    executive = ExecutiveLayer()
    missing = executive.analyze_action_plan(analyze_request("create a file"))
    original = analyze_request("create first.txt and create second.txt")
    cyclic_units = [
        replace(original.requests[0], depends_on=(1,)),
        replace(original.requests[1], depends_on=(0,)),
    ]
    missing_dependency_units = [
        original.requests[0],
        replace(original.requests[1], depends_on=(99,)),
    ]
    cyclic = executive.analyze_action_plan(
        SimpleNamespace(original_text=original.original_text, requests=cyclic_units)
    )
    invalid_reference = executive.analyze_action_plan(
        SimpleNamespace(original_text=original.original_text, requests=missing_dependency_units)
    )
    ambiguous = executive.analyze_action_plan(
        analyze_request("create alpha.txt or beta.txt")
    )

    assert any(item["stage"] == "validation" for item in missing.diagnostics)
    assert cyclic.valid is False
    assert any(item["stage"] == "dependencies" for item in cyclic.diagnostics)
    assert invalid_reference.valid is False
    assert ambiguous.valid is False
    assert any(item["stage"] == "ambiguity" for item in ambiguous.diagnostics)


def run_all_tests():
    test_storage_decision_triggers()
    test_category_decision_structured_vs_dynamic()
    test_process_execution_cycle()
    print("\nAll Executive Layer Phase A tests passed successfully!")


if __name__ == "__main__":
    run_all_tests()