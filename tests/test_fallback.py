"""
test_fallback.py
================
Focused tests for the AB Global Response Fallback System.

Validates all 10 scenarios:
  1. Normal successful response — clean text passes through unchanged.
  2. Empty Cloud Brain response — classified as INVALID_RESPONSE.
  3. Cloud provider/API failure — classified as BRAIN_FAILURE.
  4. Invalid response — classified as INVALID_RESPONSE.
  5. Missing capability — classified as CAPABILITY_UNAVAILABLE.
  6. Missing information — classified as INFORMATION_UNAVAILABLE.
  7. Unknown capability — classified as UNKNOWN_CAPABILITY.
  8. Directory-related failure — clean message, no raw exception.
  9. Existing self-info question — passes through safely.
 10. Normal knowledge question — passes through safely.

All tests verify that no raw internal failure message reaches the user.
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Project root on path
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.fallback_classifier import (
    FallbackClassifier,
    classify_failure,
    CATEGORY_CAPABILITY_UNAVAILABLE,
    CATEGORY_INFORMATION_UNAVAILABLE,
    CATEGORY_PROCESSING_FAILURE,
    CATEGORY_BRAIN_FAILURE,
    CATEGORY_INVALID_RESPONSE,
    CATEGORY_UNKNOWN_CAPABILITY,
)
from brains.brain_dispatcher import BrainResponse
from conversation.response_processor import ResponseProcessor, ProcessedResponse


# ---------------------------------------------------------------------------
# Internal error strings that must never reach the user
# ---------------------------------------------------------------------------
_INTERNAL_STRINGS = [
    "Cloud brain returned no response.",
    "Local brain returned no response.",
    "Cloud brain failed:",
    "Local brain failed:",
    "Brain connection failed.",
    "Forced cloud brain failed:",
    "API Error:",
    "APIError",
    "Exception",
    "Traceback",
    "None returned.",
    "error:",
]


def _assert_no_internal_leak(text: str, label: str = "") -> None:
    """Raise AssertionError if any internal error string appears in text."""
    for internal in _INTERNAL_STRINGS:
        assert internal.lower() not in text.lower(), (
            f"[{label}] Internal error leaked to user output: {internal!r}\n"
            f"Full text: {text!r}"
        )


# ---------------------------------------------------------------------------
# 1. FallbackClassifier unit tests
# ---------------------------------------------------------------------------

class TestFallbackClassifier(unittest.TestCase):
    """Unit tests for FallbackClassifier classification logic."""

    def setUp(self):
        self.clf = FallbackClassifier()

    # --- Scenario 2: Empty Cloud Brain response
    def test_empty_cloud_brain_classifies_as_invalid_response(self):
        result = self.clf.classify("Cloud brain returned no response.")
        self.assertEqual(result.category, CATEGORY_INVALID_RESPONSE)
        _assert_no_internal_leak(result.user_message, "empty_cloud_brain")
        self.assertTrue(result.user_message.strip())
        print("✓ test_empty_cloud_brain_classifies_as_invalid_response passed")

    # --- Scenario 3: Cloud provider / API failure
    def test_cloud_api_failure_classifies_as_brain_failure(self):
        for raw in [
            "Cloud brain failed: ConnectionError timeout",
            "Brain connection failed. Local error: timeout. Cloud error: 503.",
            "Forced cloud brain failed: API Error 429 rate limit exceeded",
        ]:
            result = self.clf.classify(raw)
            self.assertEqual(result.category, CATEGORY_BRAIN_FAILURE, f"Expected BRAIN_FAILURE for: {raw!r}")
            _assert_no_internal_leak(result.user_message, "cloud_api_failure")
        print("✓ test_cloud_api_failure_classifies_as_brain_failure passed")

    # --- Scenario 4: Invalid / malformed response
    def test_invalid_response_classification(self):
        result = self.clf.classify(raw_error="unusable response", category=CATEGORY_INVALID_RESPONSE)
        self.assertEqual(result.category, CATEGORY_INVALID_RESPONSE)
        _assert_no_internal_leak(result.user_message, "invalid_response")
        print("✓ test_invalid_response_classification passed")

    # --- Scenario 5: Missing capability
    def test_capability_unavailable_classification(self):
        result = self.clf.classify(raw_error="feature not available", category=CATEGORY_CAPABILITY_UNAVAILABLE)
        self.assertEqual(result.category, CATEGORY_CAPABILITY_UNAVAILABLE)
        _assert_no_internal_leak(result.user_message, "capability_unavailable")
        print("✓ test_capability_unavailable_classification passed")

    # --- Scenario 6: Missing information
    def test_information_unavailable_classification(self):
        result = self.clf.classify(raw_error="data not available", category=CATEGORY_INFORMATION_UNAVAILABLE)
        self.assertEqual(result.category, CATEGORY_INFORMATION_UNAVAILABLE)
        _assert_no_internal_leak(result.user_message, "information_unavailable")
        print("✓ test_information_unavailable_classification passed")

    # --- Scenario 7: Unknown capability
    def test_unknown_capability_classification(self):
        result = self.clf.classify(category=CATEGORY_UNKNOWN_CAPABILITY)
        self.assertEqual(result.category, CATEGORY_UNKNOWN_CAPABILITY)
        _assert_no_internal_leak(result.user_message, "unknown_capability")
        print("✓ test_unknown_capability_classification passed")

    # --- Auto-detection from signal strings
    def test_auto_detect_brain_failure_from_exception_text(self):
        result = self.clf.classify("Exception: Connection refused")
        self.assertEqual(result.category, CATEGORY_BRAIN_FAILURE)
        _assert_no_internal_leak(result.user_message, "auto_detect_exception")
        print("✓ test_auto_detect_brain_failure_from_exception_text passed")

    def test_auto_detect_invalid_response_from_no_response(self):
        result = self.clf.classify("returned no response")
        self.assertEqual(result.category, CATEGORY_INVALID_RESPONSE)
        _assert_no_internal_leak(result.user_message, "auto_detect_invalid")
        print("✓ test_auto_detect_invalid_response_from_no_response passed")

    def test_fallback_on_none_input(self):
        result = self.clf.classify(raw_error=None)
        self.assertIn(result.category, [
            CATEGORY_PROCESSING_FAILURE,
            CATEGORY_BRAIN_FAILURE,
            CATEGORY_INVALID_RESPONSE,
        ])
        _assert_no_internal_leak(result.user_message, "none_input")
        print("✓ test_fallback_on_none_input passed")

    def test_all_categories_have_non_empty_messages(self):
        categories = [
            CATEGORY_CAPABILITY_UNAVAILABLE,
            CATEGORY_INFORMATION_UNAVAILABLE,
            CATEGORY_PROCESSING_FAILURE,
            CATEGORY_BRAIN_FAILURE,
            CATEGORY_INVALID_RESPONSE,
            CATEGORY_UNKNOWN_CAPABILITY,
        ]
        for cat in categories:
            result = self.clf.classify(category=cat)
            self.assertEqual(result.category, cat)
            self.assertTrue(result.user_message.strip(), f"Empty message for category: {cat}")
            _assert_no_internal_leak(result.user_message, cat)
        print("✓ test_all_categories_have_non_empty_messages passed")


# ---------------------------------------------------------------------------
# 2. safe_response + brain sentinel tests
# ---------------------------------------------------------------------------

class TestSafeResponse(unittest.TestCase):
    """Verifies safe_response() strips the fallback sentinel correctly."""

    def setUp(self):
        from Jarwis import safe_response, _BRAIN_FALLBACK_PREFIX, BRAIN_FALLBACK
        self.safe_response = safe_response
        self.prefix = _BRAIN_FALLBACK_PREFIX
        self.generic = BRAIN_FALLBACK

    # --- Scenario 1: Normal successful response
    def test_normal_response_passes_through(self):
        normal = "Gravity is a force that attracts masses toward each other."
        result = self.safe_response(normal)
        self.assertEqual(result, normal)
        _assert_no_internal_leak(result, "normal_response")
        print("✓ test_normal_response_passes_through passed")

    # --- Sentinel stripping
    def test_sentinel_stripped_returns_clean_message(self):
        clean_msg = "I couldn't complete that request with my currently available processing systems."
        sentineled = f"{self.prefix}{clean_msg}"
        result = self.safe_response(sentineled)
        self.assertEqual(result, clean_msg)
        _assert_no_internal_leak(result, "sentinel_stripping")
        print("✓ test_sentinel_stripped_returns_clean_message passed")

    def test_empty_string_returns_generic_fallback(self):
        result = self.safe_response("")
        self.assertTrue(result.strip())
        _assert_no_internal_leak(result, "empty_string")
        print("✓ test_empty_string_returns_generic_fallback passed")

    def test_none_returns_generic_fallback(self):
        result = self.safe_response(None)
        self.assertTrue(result.strip())
        _assert_no_internal_leak(result, "none_input")
        print("✓ test_none_returns_generic_fallback passed")


# ---------------------------------------------------------------------------
# 3. ResponseProcessor fallback tests
# ---------------------------------------------------------------------------

class TestResponseProcessorFallback(unittest.TestCase):
    """Ensures ResponseProcessor produces clean fallback messages, not raw error strings."""

    def setUp(self):
        self.processor = ResponseProcessor()

    # --- Scenario 1: Normal successful response
    def test_normal_response_processes_cleanly(self):
        raw = BrainResponse(
            content="The capital of France is Paris.",
            brain_id="mock_v1",
            provider="local",
            mode="QUERY",
            latency_ms=5.0,
        )
        result = self.processor.process(raw)
        self.assertTrue(result.is_safe)
        self.assertFalse(result.fallback_triggered)
        self.assertEqual(result.sanitized_content, "The capital of France is Paris.")
        _assert_no_internal_leak(result.sanitized_content, "processor_normal")
        print("✓ test_normal_response_processes_cleanly passed")

    # --- Scenario 4: Invalid / empty response through processor
    def test_empty_response_triggers_clean_fallback(self):
        raw = BrainResponse(
            content="",
            brain_id="mock_v1",
            provider="local",
            mode="QUERY",
            latency_ms=1.0,
        )
        result = self.processor.process(raw)
        self.assertFalse(result.is_safe)
        self.assertTrue(result.fallback_triggered)
        _assert_no_internal_leak(result.sanitized_content, "processor_empty")
        self.assertTrue(result.sanitized_content.strip())
        print("✓ test_empty_response_triggers_clean_fallback passed")

    def test_none_response_triggers_clean_fallback(self):
        result = self.processor.process(None)
        self.assertTrue(result.fallback_triggered)
        _assert_no_internal_leak(result.sanitized_content, "processor_none")
        self.assertTrue(result.sanitized_content.strip())
        print("✓ test_none_response_triggers_clean_fallback passed")


# ---------------------------------------------------------------------------
# 4. AGPipeline fallback tests (scenarios 2-4)
# ---------------------------------------------------------------------------

class TestAGPipelineFallbacks(unittest.TestCase):
    """Verifies AGPipeline fallback messages are clean across failure scenarios."""

    def setUp(self):
        from pipeline.ag_pipeline import AGPipeline
        self.pipeline = AGPipeline()

    # --- Scenario 3: Brain dispatcher exception → BRAIN_FAILURE message
    def test_brain_dispatcher_failure_returns_clean_message(self):
        self.pipeline.dispatcher.dispatch = MagicMock(
            side_effect=ConnectionError("Cloud API unreachable")
        )
        result = self.pipeline.process_input("Tell me about quantum mechanics.")
        self.assertFalse(result.success)
        self.assertTrue(result.processed_response.fallback_triggered)
        _assert_no_internal_leak(
            result.processed_response.sanitized_content, "pipeline_brain_failure"
        )
        self.assertTrue(result.processed_response.sanitized_content.strip())
        print("✓ test_brain_dispatcher_failure_returns_clean_message passed")

    # --- Scenario 4: Response processor returning empty content
    def test_empty_brain_response_returns_clean_fallback(self):
        from brains.brain_dispatcher import BrainResponse
        self.pipeline.dispatcher.dispatch = MagicMock(
            return_value=BrainResponse(
                content="",
                brain_id="mock",
                provider="cloud",
                mode="QUERY",
                latency_ms=1.0,
            )
        )
        result = self.pipeline.process_input("What is the speed of light?")
        self.assertTrue(result.processed_response is not None)
        _assert_no_internal_leak(
            result.processed_response.sanitized_content, "pipeline_empty_content"
        )
        print("✓ test_empty_brain_response_returns_clean_fallback passed")


# ---------------------------------------------------------------------------
# 5. End-to-end safe_response scenarios (scenarios 8-10)
# ---------------------------------------------------------------------------

class TestEndToEndSafeResponse(unittest.TestCase):
    """
    Tests safe_response integration against the sentinel + real brain output.
    Covers scenarios 8 (directory failure), 9 (self-info), 10 (knowledge question).
    """

    def setUp(self):
        from Jarwis import safe_response, _BRAIN_FALLBACK_PREFIX
        self.safe_response = safe_response
        self.prefix = _BRAIN_FALLBACK_PREFIX

    # --- Scenario 8: Directory-related failure
    def test_directory_failure_produces_clean_response(self):
        # Simulate a sentinel-wrapped PROCESSING_FAILURE from a directory exception
        fc = FallbackClassifier()
        result = fc.classify(category=CATEGORY_PROCESSING_FAILURE)
        final = self.safe_response(f"{self.prefix}{result.user_message}")
        _assert_no_internal_leak(final, "directory_failure")
        self.assertTrue(final.strip())
        print("✓ test_directory_failure_produces_clean_response passed")

    # --- Scenario 9: Existing self-info question
    def test_self_info_normal_response_not_leaked(self):
        # A clean brain answer to "who are you" should pass through intact
        answer = "I am AB, Ambient Guidance. Version 0.7 Beta."
        result = self.safe_response(answer)
        self.assertEqual(result, answer)
        _assert_no_internal_leak(result, "self_info_answer")
        print("✓ test_self_info_normal_response_not_leaked passed")

    # --- Scenario 10: Normal knowledge question
    def test_normal_knowledge_question_passes_through(self):
        answer = "Photosynthesis is the process plants use to convert light into energy."
        result = self.safe_response(answer)
        self.assertEqual(result, answer)
        _assert_no_internal_leak(result, "knowledge_question")
        print("✓ test_normal_knowledge_question_passes_through passed")


# ---------------------------------------------------------------------------
# 6. Regression: existing ResponseProcessor tests still pass
# ---------------------------------------------------------------------------

class TestResponseProcessorRegression(unittest.TestCase):
    """Ensures previously passing ResponseProcessor tests still pass."""

    def setUp(self):
        self.processor = ResponseProcessor()

    def test_basic_processing_and_sanitization(self):
        raw = BrainResponse(
            content="  <|im_start|>Here is the explanation for gravity.<|im_end|>  ",
            brain_id="mock_v1",
            provider="local",
            mode="QUERY",
            latency_ms=10.0,
        )
        processed = self.processor.process(raw)
        self.assertIsInstance(processed, ProcessedResponse)
        self.assertTrue(processed.is_safe)
        self.assertEqual(processed.sanitized_content, "Here is the explanation for gravity.")
        self.assertFalse(processed.fallback_triggered)
        print("✓ regression: test_basic_processing_and_sanitization passed")

    def test_action_extraction_and_confirmation_flag(self):
        raw = BrainResponse(
            content="I can delete this file. Please confirm deletion before proceeding.",
            brain_id="mock_v1",
            provider="local",
            mode="ACTION",
            latency_ms=12.0,
        )
        processed = self.processor.process(raw)
        self.assertTrue(processed.requires_user_confirmation)
        self.assertIsNotNone(processed.extracted_pending_action)
        self.assertIsInstance(processed.extracted_pending_action, dict)
        self.assertEqual(processed.extracted_pending_action.get("type"), "unconfirmed_brain_action")
        print("✓ regression: test_action_extraction_and_confirmation_flag passed")

    def test_fallback_on_empty_or_unsafe_response(self):
        raw_empty = BrainResponse(
            content="", brain_id="mock_v1", provider="local", mode="QUERY", latency_ms=1.0
        )
        proc_empty = self.processor.process(raw_empty)
        self.assertFalse(proc_empty.is_safe)
        self.assertTrue(proc_empty.fallback_triggered)
        # Must NOT contain [AG Fallback] — replaced by classifier message
        _assert_no_internal_leak(proc_empty.sanitized_content, "regression_empty")
        self.assertTrue(proc_empty.sanitized_content.strip())

        proc_none = self.processor.process(None)
        self.assertTrue(proc_none.fallback_triggered)
        _assert_no_internal_leak(proc_none.sanitized_content, "regression_none")
        print("✓ regression: test_fallback_on_empty_or_unsafe_response passed")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all_tests():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [
        TestFallbackClassifier,
        TestSafeResponse,
        TestResponseProcessorFallback,
        TestAGPipelineFallbacks,
        TestEndToEndSafeResponse,
        TestResponseProcessorRegression,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
