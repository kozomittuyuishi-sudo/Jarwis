"""
quips_tests.py
===============
Regression tests for quips.py, written against the section-17 regression
list plus the specific bugs found and fixed during this review.

IMPORTANT LIMITATION — read before trusting the self-info results here
-------------------------------------------------------------------------
quips.py delegates ALL self-info classification to
`self_info.query_router.route_query()`. That module's actual source was
never provided to this review, so these tests run against a minimal
LOCAL STUB (see `self_info/query_router.py` in this same sandbox, which
is NOT part of the delivered changes and should not be copied into the
real project). Self-info assertions below verify that QUIPS correctly
DELEGATES to and RESPECTS whatever route_query returns (precedence,
no re-interpretation, no corruption) — they say nothing about whether
the real router itself distinguishes e.g. "what are you capable of"
from "what are you" with sufficient precision. That must be verified
against the real self_info/query_router.py directly.

Everything else here (compound decomposition, target/scope extraction,
action/resolution-mode inference, workspace routing, error isolation)
tests quips.py's own logic end-to-end with no stubbing involved.
"""

from __future__ import annotations

from quips import (
    RequestCategory,
    RoutingDecision,
    analyze_request,
    resolve_target_reference,
)
from target_resolution_models import ActionHint, ResolutionMode, TargetType
from target_resolver import TargetResolver


_FAILURES = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if not condition:
        _FAILURES.append(f"{label}: {detail}")


# =============================================================================
# Self-info (delegation correctness only — see module docstring)
# =============================================================================

def test_self_info_delegation() -> None:
    single_cases = [
        "What can you do?",
        "What are your capabilities?",
        "What are you capable of?",
        "What functionality do you have?",
        "How do you work?",
        "How do you work internally?",
        "How do you process requests?",
        "What brains do you have?",
    ]
    for text in single_cases:
        r = analyze_request(text)
        check(f"self-info single unit: {text!r}", r.request_count == 1, r.to_dict())
        unit = r.requests[0]
        check(f"self-info category: {text!r}", unit.category == RequestCategory.SELF_INFO,
              unit.category)
        check(f"self-info deterministic routing: {text!r}",
              unit.routing_decision == RoutingDecision.DETERMINISTIC_SELF_INFO,
              unit.routing_decision)

    compound = analyze_request(
        "Tell me who you are, what you're capable of, and how you work internally."
    )
    check("self-info compound: splits into 3", compound.request_count == 3,
          compound.to_dict())
    check("self-info compound: all self-info",
          all(u.category == RequestCategory.SELF_INFO for u in compound.requests),
          [u.category for u in compound.requests])
    check("self-info compound: distinct intents",
          len({u.intent for u in compound.requests}) == 3,
          [u.intent for u in compound.requests])


# =============================================================================
# Filesystem — single-request correctness
# =============================================================================

def test_filesystem_single_requests() -> None:
    r = analyze_request("read test.txt")
    u = r.requests[0]
    check("read test.txt: category", u.category == RequestCategory.FILE_OPERATION)
    check("read test.txt: target", u.target_reference.original_target == "test.txt",
          u.target_reference.to_dict())
    check("read test.txt: mode", u.target_reference.resolution_mode == ResolutionMode.WORKSPACE_RELATIVE)
    check("read test.txt: action", u.target_reference.action == ActionHint.READ)
    check("read test.txt: not LLM fallback",
          u.routing_decision != RoutingDecision.LLM_FALLBACK)

    r2 = analyze_request("read test.txt in AG")
    u2 = r2.requests[0]
    check("read test.txt in AG: target", u2.target_reference.original_target == "test.txt")
    check("read test.txt in AG: scope", u2.target_reference.scope == "AG")
    check("read test.txt in AG: mode", u2.target_reference.resolution_mode == ResolutionMode.NAMED_SCOPE)

    r3 = analyze_request("read D:/AG/test.txt")
    u3 = r3.requests[0]
    check("read D:/AG/test.txt: target", u3.target_reference.original_target == "D:/AG/test.txt")
    check("read D:/AG/test.txt: mode", u3.target_reference.resolution_mode == ResolutionMode.ABSOLUTE_PATH)

    r4 = analyze_request("create test.txt")
    u4 = r4.requests[0]
    check("create test.txt: intent", u4.intent == "CREATE_FILE")
    check("create test.txt: target", u4.target_reference.original_target == "test.txt")
    check("create test.txt: action", u4.target_reference.action == ActionHint.CREATE)


def test_filesystem_compound_requests() -> None:
    r = analyze_request("create test.txt and explain what you created")
    check("dependent workflow: splits into 2", r.request_count == 2, r.to_dict())
    check("dependent workflow: unit0 is create", r.requests[0].intent == "CREATE_FILE")
    check("dependent workflow: unit1 is explain", r.requests[1].intent == "EXPLAIN_RESULT")
    check("dependent workflow: unit1 depends on unit0", r.requests[1].depends_on == (0,),
          r.requests[1].depends_on)

    r2 = analyze_request("read Brain.py, check the workspace, and tell me what you can do")
    check("multiple independent: splits into 3 (was 1 before this fix)",
          r2.request_count == 3, r2.to_dict())
    check("multiple independent: unit0 is the Brain.py read",
          r2.requests[0].target_reference is not None
          and r2.requests[0].target_reference.original_target == "Brain.py",
          r2.requests[0].to_dict())


# =============================================================================
# Workspace routing (the critical regression: previously fell to LLM_FALLBACK)
# =============================================================================

def test_workspace_routing() -> None:
    set_cases = [
        "set workspace to D:/AG",
        "switch workspace to D:/AG",
        "switch to D:/AG",
        "work in D:/AG",
    ]
    for text in set_cases:
        r = analyze_request(text)
        u = r.requests[0]
        check(f"workspace set routes deterministically: {text!r}",
              u.routing_decision != RoutingDecision.LLM_FALLBACK, u.to_dict())
        check(f"workspace set category: {text!r}",
              u.category == RequestCategory.WORKSPACE_QUERY, u.category)
        check(f"workspace set intent: {text!r}", u.intent == "WORKSPACE_SET", u.intent)

    query_cases = ["what is the current workspace?", "what's the current workspace?",
                   "where am i"]
    for text in query_cases:
        r = analyze_request(text)
        u = r.requests[0]
        check(f"workspace query routes deterministically: {text!r}",
              u.routing_decision != RoutingDecision.LLM_FALLBACK, u.to_dict())
        check(f"workspace query intent: {text!r}", u.intent == "WORKSPACE_QUERY", u.intent)


# =============================================================================
# Target Resolver integration shape (section 9's canonical example)
# =============================================================================

def test_target_resolver_integration_shape() -> None:
    r = analyze_request("access test.txt in AG")
    u = r.requests[0]
    check("access example: category", u.category == RequestCategory.FILE_OPERATION)
    tr = u.target_reference
    check("access example: action=ACCESS", tr.action == ActionHint.ACCESS, tr.action)
    check("access example: target=test.txt", tr.original_target == "test.txt", tr.original_target)
    check("access example: target_type=FILE", tr.target_type == TargetType.FILE, tr.target_type)
    check("access example: scope=AG", tr.scope == "AG", tr.scope)
    check("access example: mode=NAMED_SCOPE", tr.resolution_mode == ResolutionMode.NAMED_SCOPE,
          tr.resolution_mode)
    check("access example: QUIPS did not build the full path itself",
          "AG" not in tr.original_target and "\\" not in tr.original_target,
          tr.original_target)

    # End-to-end: hand the TargetReference to a real TargetResolver and
    # confirm Target Resolver (not QUIPS) is what turns it into a path.
    import tempfile, shutil
    from pathlib import Path
    tmp = Path(tempfile.mkdtemp())
    ws = tmp / "AG"
    ws.mkdir()
    (ws / "test.txt").write_text("hi", encoding="utf-8")
    try:
        resolver = TargetResolver(workspace_source=str(ws), scope_registry={"AG": str(ws)})
        resolved = resolve_target_reference(u, resolver=resolver)
        check("access example resolves end-to-end",
              resolved is not None and resolved.resolved_path == str(ws / "test.txt"),
              resolved.to_dict() if resolved else None)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# =============================================================================
# Bug-fix-specific regressions
# =============================================================================

def test_trailing_punctuation_does_not_leak() -> None:
    r = analyze_request("read test.txt.")
    check("trailing period stripped from target",
          r.requests[0].target_reference.original_target == "test.txt",
          r.requests[0].target_reference.original_target)

    r2 = analyze_request("create a file called test.txt in AG.")
    tr = r2.requests[0].target_reference
    check("called-X target extracted correctly (not 'a file called test.txt')",
          tr.original_target == "test.txt", tr.original_target)
    check("scope trailing period stripped", tr.scope == "AG", tr.scope)


def test_called_named_extraction() -> None:
    r = analyze_request("create a folder called AB_Test")
    tr = r.requests[0].target_reference
    check("'called X' folder extraction", tr.original_target == "AB_Test", tr.original_target)
    check("'called X' folder type", tr.target_type == TargetType.DIRECTORY, tr.target_type)

    r2 = analyze_request('create a file named "my notes.txt"')
    tr2 = r2.requests[0].target_reference
    check("'named \"X Y.txt\"' quoted extraction", tr2.original_target == "my notes.txt",
          tr2.original_target)


def test_relative_windows_path_not_misclassified_absolute() -> None:
    r = analyze_request("create a folder called backup\\test")
    tr = r.requests[0].target_reference
    check("relative nested windows path stays WORKSPACE_RELATIVE (was ABSOLUTE_PATH bug)",
          tr.resolution_mode == ResolutionMode.WORKSPACE_RELATIVE, tr.resolution_mode)


def test_target_type_substring_collision_fixed() -> None:
    r = analyze_request("read my_folder_notes.txt")
    tr = r.requests[0].target_reference
    check("filename containing 'folder' substring is still FILE",
          tr.target_type == TargetType.FILE, tr.target_type)


def test_action_hint_word_boundaries() -> None:
    r = analyze_request("rename notes.txt to remarks.txt")
    tr = r.requests[0].target_reference
    check("'rename' action not confused by substring", tr.action == ActionHint.MOVE_SOURCE,
          tr.action)


# =============================================================================
# Context references (must be preserved, never resolved by QUIPS)
# =============================================================================

def test_context_reference_preserved_not_resolved() -> None:
    r = analyze_request("read that file")
    tr = r.requests[0].target_reference
    check("context reference mode", tr.resolution_mode == ResolutionMode.CONTEXT_REFERENCE,
          tr.resolution_mode)
    check("context reference does not invent a path",
          tr.context.get("reference_path") == "that file", tr.context)
    check("context_references tuple flags 'that'", "that" in r.requests[0].context_references,
          r.requests[0].context_references)


# =============================================================================
# Error isolation (section 16)
# =============================================================================

def test_error_isolation() -> None:
    def flaky_classifier(text: str) -> str:
        if "brain.py" in text.lower():
            raise RuntimeError("simulated legacy classifier crash")
        return "SOME_LEGACY_INTENT"

    r = analyze_request(
        "read Brain.py, create test.txt, and what are your capabilities",
        intent_classifier=flaky_classifier,
    )
    check("error isolation: all 3 siblings still produced", r.request_count == 3, r.to_dict())
    check("error isolation: failing unit marked, not silently dropped",
          r.requests[0].processing_status == "CLASSIFICATION_FAILED", r.requests[0].processing_status)
    check("error isolation: sibling 1 unaffected", r.requests[1].category == RequestCategory.FILE_OPERATION,
          r.requests[1].category)
    check("error isolation: sibling 2 unaffected", r.requests[2].category == RequestCategory.SELF_INFO,
          r.requests[2].category)


def test_empty_and_unknown_input_safe() -> None:
    r = analyze_request("")
    check("empty input: no units, no crash", r.request_count == 0)

    r2 = analyze_request("asdkjaslkdj qwoiqwoe")
    check("gibberish input: does not crash", r2.request_count >= 1)
    check("gibberish input: not forced into a false-precise category",
          r2.requests[0].category in (RequestCategory.GENERAL, RequestCategory.UNKNOWN),
          r2.requests[0].category)


# =============================================================================
# Runner
# =============================================================================

def run_all() -> None:
    test_self_info_delegation()
    test_filesystem_single_requests()
    test_filesystem_compound_requests()
    test_workspace_routing()
    test_target_resolver_integration_shape()
    test_trailing_punctuation_does_not_leak()
    test_called_named_extraction()
    test_relative_windows_path_not_misclassified_absolute()
    test_target_type_substring_collision_fixed()
    test_action_hint_word_boundaries()
    test_context_reference_preserved_not_resolved()
    test_error_isolation()
    test_empty_and_unknown_input_safe()

    print("=" * 70)
    if _FAILURES:
        print(f"QUIPS TESTS: {len(_FAILURES)} FAILURE(S)\n")
        for f in _FAILURES:
            print(f"  - {f}")
        raise SystemExit(1)
    print("QUIPS TESTS: ALL PASSED")


if __name__ == "__main__":
    run_all()