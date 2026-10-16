"""Regression tests for deterministic QUIPS request decomposition."""

from quips import RequestCategory, RoutingDecision, UnitProcessingResult, analyze_request


def _legacy_intent(text: str) -> str:
    """Test double standing in for AG's existing detect_intent callback."""
    return "legacy:" + text.lower()


def test_self_info_compound_request_is_decomposed():
    result = analyze_request(
        "Tell me who you are, what you're capable of, and how you work internally.",
        _legacy_intent,
    )

    assert result.is_compound
    assert result.request_count >= 3
    assert [unit.category for unit in result.requests] == [RequestCategory.SELF_INFO] * 3
    assert [unit.intent for unit in result.requests] == ["IDENTITY", "CAPABILITIES", "ARCHITECTURE"]
    assert all(unit.route_intent.startswith("legacy:") for unit in result.requests)
    assert all(unit.routing_decision == RoutingDecision.DETERMINISTIC_SELF_INFO for unit in result.requests)
    assert all(unit.original_text == result.original_text for unit in result.requests)
    assert [unit.index for unit in result.requests] == [0, 1, 2]


def test_capabilities_and_limitations_are_independent_requests():
    result = analyze_request("What can you do and what can't you do?", _legacy_intent)

    assert result.is_compound
    assert result.request_count == 2
    assert [unit.intent for unit in result.requests] == ["CAPABILITIES", "LIMITATIONS"]


def test_file_operation_and_explanation_are_independent_requests():
    result = analyze_request("Create test.txt and tell me why you created it.", _legacy_intent)

    assert result.is_compound
    assert result.request_count == 2
    assert result.requests[0].category == RequestCategory.FILE_OPERATION
    assert result.requests[0].intent == "CREATE_FILE"
    assert result.requests[1].category == RequestCategory.GENERAL


def test_file_operation_and_result_explanation_are_independent_and_ordered():
    result = analyze_request("Create test.txt and tell me what you created.", _legacy_intent)

    assert result.is_compound
    assert [unit.intent for unit in result.requests] == ["CREATE_FILE", "EXPLAIN_RESULT"]
    assert result.requests[1].depends_on == (0,)


def test_single_requests_are_not_split():
    for query in ("What is my current workspace?", "Tell me about yourself."):
        result = analyze_request(query, _legacy_intent)
        assert not result.is_compound
        assert result.request_count == 1
        assert result.requests[0].original_text == query


def test_operational_workflow_is_not_treated_as_two_unrelated_requests():
    result = analyze_request("Create a folder named projects and put test.txt inside it.", _legacy_intent)

    assert not result.is_compound
    assert result.request_count == 1


def test_unknown_request_is_explicit_and_not_forced_into_a_handler():
    result = analyze_request("Florbulate the unclassified thing.", _legacy_intent)

    assert result.request_count == 1
    assert result.requests[0].category == RequestCategory.GENERAL
    assert result.requests[0].routing_decision == RoutingDecision.LLM_FALLBACK


def test_unit_processing_result_has_an_explicit_terminal_state():
    result = UnitProcessingResult(unit_index=1, handler="DETERMINISTIC_SELF_INFO", status="COMPLETED")

    assert result.to_dict()["status"] == "COMPLETED"


def test_sequential_file_workflow_is_split_and_linked():
    result = analyze_request(
        "Create a folder called Experiments, create notes.txt inside it, "
        "write 'Hello Jarwis' into the file, and read it back.",
        _legacy_intent,
    )

    assert [unit.intent for unit in result.requests] == [
        "CREATE_FOLDER", "CREATE_FILE", "WRITE_FILE", "READ_FILE",
    ]
    assert [unit.depends_on for unit in result.requests] == [(), (0,), (1,), (2,)]
    assert result.requests[2].understanding.content == "Hello Jarwis"
    assert result.requests[2].original_segment == "write 'Hello Jarwis' into the file"
    assert result.original_text.endswith("read it back.")


def test_quoted_connector_words_are_preserved_as_literal_content():
    result = analyze_request(
        "Create note.txt, write 'this and that then after' into it, and read it back.",
        _legacy_intent,
    )

    assert [unit.intent for unit in result.requests] == [
        "CREATE_FILE", "WRITE_FILE", "READ_FILE",
    ]
    assert result.requests[1].understanding.content == "this and that then after"
    assert result.requests[1].depends_on == (0,)
    assert result.requests[2].depends_on == (1,)


def test_several_independent_action_phrases_remain_distinct():
    result = analyze_request(
        "Create alpha.txt, create beta.txt, and list the current directory.",
        _legacy_intent,
    )

    assert [unit.intent for unit in result.requests] == [
        "CREATE_FILE", "CREATE_FILE", "LIST_DIRECTORY",
    ]
    assert all(unit.depends_on == () for unit in result.requests)


def test_then_connector_links_sequential_actions():
    result = analyze_request(
        "Create first.txt, then read first.txt.",
        _legacy_intent,
    )

    assert [unit.intent for unit in result.requests] == ["CREATE_FILE", "READ_FILE"]
    assert result.requests[1].depends_on == (0,)


def test_empty_input_preserves_empty_quips_result_without_index_assumptions():
    result = analyze_request("", _legacy_intent)

    assert result.request_count == 0
    assert result.requests == []


def test_contextual_read_reference_normalizes_back_suffix():
    result = analyze_request("read it back", _legacy_intent)

    assert result.requests[0].target_reference is not None
    assert result.requests[0].target_reference.resolution_mode == "CONTEXT_REFERENCE"


def test_recent_file_description_is_a_contextual_reference():
    result = analyze_request("read the file I just created", _legacy_intent)

    assert result.requests[0].target_reference is not None
    assert result.requests[0].target_reference.resolution_mode == "CONTEXT_REFERENCE"
    assert result.requests[0].understanding.context_requirements


def test_file_choice_is_flagged_as_ambiguous_instead_of_becoming_a_target():
    result = analyze_request("create alpha.txt or beta.txt", _legacy_intent)

    assert result.requests[0].understanding.semantic_reasoning_required
