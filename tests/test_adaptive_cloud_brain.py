"""
test_adaptive_cloud_brain.py
============================
Comprehensive pytest suite for Part A — Adaptive Cloud Brain.

Coverage
--------
TestProviderRegistry        — ProviderRegistry + ProviderProfile
TestTokenEstimator          — TokenEstimator + EstimationResult
TestRequestValidator        — RequestValidator + ValidationReport
TestResponseValidator       — ResponseValidator across all 6 axes
TestRetryManager            — RetryManager + RetryOutcome
TestRequestLogger           — RequestLogger + RequestRecord
TestAdaptiveCloudBrain      — AdaptiveCloudBrain orchestrator (full pipeline)

All tests use stdlib only — no network calls, no providers, no LLM calls.
"""

import pytest
from brains.cloud_brain.provider_registry import (
    ProviderRegistry,
    ProviderProfile,
    ProviderFeature,
)
from brains.cloud_brain.token_estimator import (
    TokenEstimator,
    EstimatorConfig,
    EstimationResult,
    RequestCategory,
)
from brains.cloud_brain.request_validator import (
    RequestValidator,
    RequestValidatorConfig,
    ValidationReport,
    ValidationIssue,
    Severity,
)
from brains.cloud_brain.response_validator import (
    ResponseValidator,
    ResponseValidatorConfig,
    ValidationResult,
    ValidationAxis,
    AxisResult,
)
from brains.cloud_brain.retry_manager import (
    RetryManager,
    RetryConfig,
    RetryOutcome,
    RetryStatus,
)
from brains.cloud_brain.request_logger import (
    RequestLogger,
    LoggerConfig,
    RequestRecord,
    LogLevel,
)
from brains.cloud_brain.adaptive_brain import (
    AdaptiveCloudBrain,
    BrainResult,
    BrainStatus,
    PrepResult,
)


# ============================================================
# Helpers / fixtures
# ============================================================

def make_good_response(content="Hello world", finish_reason="stop"):
    """Return a minimal well-formed provider response dict."""
    return {
        "id": "resp_abc123",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": finish_reason,
            }
        ],
    }


def make_estimation(max_tokens=512, prompt_tokens=50, category=RequestCategory.CONVERSATIONAL):
    """Return a minimal EstimationResult."""
    return EstimationResult(
        category=category,
        max_tokens=max_tokens,
        estimated_prompt_tokens=prompt_tokens,
        complexity_score=0.1,
        provider_ceiling=4096,
        reasoning=["test"],
    )


# ============================================================
# TestProviderRegistry
# ============================================================

class TestProviderRegistry:

    def test_defaults_loaded(self):
        reg = ProviderRegistry()
        assert "openrouter" in reg
        assert "anthropic" in reg
        assert "openai" in reg
        assert "ollama" in reg

    def test_len(self):
        reg = ProviderRegistry()
        assert len(reg) == 4

    def test_get_known_provider(self):
        reg = ProviderRegistry()
        profile = reg.get("openrouter")
        assert profile is not None
        assert profile.provider_id == "openrouter"

    def test_get_unknown_returns_none(self):
        reg = ProviderRegistry()
        assert reg.get("nonexistent") is None

    def test_require_known(self):
        reg = ProviderRegistry()
        profile = reg.require("anthropic")
        assert profile.provider_id == "anthropic"

    def test_require_unknown_raises(self):
        reg = ProviderRegistry()
        with pytest.raises(KeyError):
            reg.require("does_not_exist")

    def test_register_custom(self):
        reg = ProviderRegistry()
        custom = ProviderProfile(
            provider_id="custom_ai",
            display_name="Custom AI",
            supported_features=frozenset({ProviderFeature.STREAMING}),
            context_window=16_000,
            max_output_tokens=2048,
        )
        reg.register(custom)
        assert "custom_ai" in reg
        assert reg.get("custom_ai").display_name == "Custom AI"

    def test_register_replaces_existing(self):
        reg = ProviderRegistry()
        new_profile = ProviderProfile(
            provider_id="openrouter",
            display_name="OpenRouter Override",
        )
        reg.register(new_profile)
        assert reg.get("openrouter").display_name == "OpenRouter Override"

    def test_register_wrong_type_raises(self):
        reg = ProviderRegistry()
        with pytest.raises(TypeError):
            reg.register("not a profile")  # type: ignore

    def test_remove_existing(self):
        reg = ProviderRegistry()
        removed = reg.remove("ollama")
        assert removed is True
        assert "ollama" not in reg

    def test_remove_nonexistent_returns_false(self):
        reg = ProviderRegistry()
        assert reg.remove("ghost") is False

    def test_no_defaults(self):
        reg = ProviderRegistry(load_defaults=False)
        assert len(reg) == 0

    def test_provider_ids(self):
        reg = ProviderRegistry()
        ids = reg.provider_ids()
        assert "openrouter" in ids
        assert "openai" in ids

    def test_all_providers(self):
        reg = ProviderRegistry()
        profiles = reg.all_providers()
        assert len(profiles) == 4
        assert all(isinstance(p, ProviderProfile) for p in profiles)

    def test_supports_feature_true(self):
        reg = ProviderRegistry()
        assert reg.supports_feature("openrouter", ProviderFeature.STREAMING) is True

    def test_supports_feature_false(self):
        reg = ProviderRegistry()
        # ollama does not support FUNCTION_CALLING
        assert reg.supports_feature("ollama", ProviderFeature.FUNCTION_CALLING) is False

    def test_supports_feature_unknown_provider(self):
        reg = ProviderRegistry()
        assert reg.supports_feature("unknown", ProviderFeature.STREAMING) is False

    def test_providers_with_feature(self):
        reg = ProviderRegistry()
        results = reg.providers_with_feature(ProviderFeature.REASONING_BLOCKS)
        ids = [p.provider_id for p in results]
        assert "anthropic" in ids

    def test_find_model(self):
        reg = ProviderRegistry()
        profiles = reg.find_model("gpt-4o")
        ids = [p.provider_id for p in profiles]
        assert "openai" in ids

    def test_find_model_not_found(self):
        reg = ProviderRegistry()
        assert reg.find_model("model_that_does_not_exist") == []

    def test_profile_clamp_output_tokens(self):
        reg = ProviderRegistry()
        profile = reg.require("openrouter")
        # openrouter max_output_tokens = 8192
        clamped = profile.clamp_output_tokens(99999)
        assert clamped == profile.max_output_tokens

    def test_profile_clamp_output_tokens_below_max(self):
        reg = ProviderRegistry()
        profile = reg.require("openrouter")
        assert profile.clamp_output_tokens(512) == 512

    def test_profile_has_model_true(self):
        reg = ProviderRegistry()
        profile = reg.require("openai")
        assert profile.has_model("gpt-4o") is True

    def test_profile_has_model_false(self):
        reg = ProviderRegistry()
        profile = reg.require("openai")
        assert profile.has_model("llama3.1") is False

    def test_profile_to_dict(self):
        reg = ProviderRegistry()
        d = reg.require("openrouter").to_dict()
        assert d["provider_id"] == "openrouter"
        assert "supported_features" in d
        assert isinstance(d["models"], list)


# ============================================================
# TestTokenEstimator
# ============================================================

class TestTokenEstimator:

    def test_greeting_short_prompt(self):
        est = TokenEstimator()
        result = est.estimate("Hi", intent="greeting")
        assert result.category == RequestCategory.GREETING
        assert result.max_tokens <= 128

    def test_short_factual_intent(self):
        est = TokenEstimator()
        result = est.estimate("What version is this?", intent="brain_status")
        assert result.category == RequestCategory.SHORT_FACTUAL
        assert result.max_tokens <= 256

    def test_conversational_length(self):
        est = TokenEstimator()
        # Medium prompt — no keywords, no intent
        prompt = "Tell me about the weather in Paris " * 5  # ~50 tokens
        result = est.estimate(prompt)
        assert result.max_tokens > 0
        assert result.category in (RequestCategory.CONVERSATIONAL, RequestCategory.REASONING)

    def test_report_keyword_detection(self):
        est = TokenEstimator()
        result = est.estimate("Write a comprehensive report on machine learning.")
        assert result.category == RequestCategory.LARGE_REPORT
        assert result.max_tokens == 2048

    def test_output_budget_leaves_room_for_prompt_in_context_window(self):
        est = TokenEstimator()
        result = est.estimate(
            "Explain this briefly. " * 30,
            category="large_report",
            provider_max_output=8192,
            provider_context_window=512,
        )
        assert result.max_tokens <= 512 - result.estimated_prompt_tokens - 128

    def test_complexity_uplift_is_still_clamped_before_provider_dispatch(self):
        config = EstimatorConfig(provider_safety_cap=4096)
        config.category_budgets[RequestCategory.LARGE_REPORT] = 4096
        task = "Write a detailed report, first then finally compare two strategies? Why?"

        result = TokenEstimator(config).estimate(
            task,
            provider_max_output=8192,
            provider_context_window=128000,
            task_text=task,
        )

        assert result.complexity_score == pytest.approx(0.6)
        assert any("4505 tokens" in step for step in result.reasoning)
        assert result.max_tokens == 4096

    def test_reasoning_keyword_detection(self):
        est = TokenEstimator()
        result = est.estimate("Explain the difference between SQL and NoSQL databases.")
        assert result.category == RequestCategory.REASONING

    def test_caller_supplied_category_overrides(self):
        est = TokenEstimator()
        result = est.estimate("Hi", category="large_report")
        assert result.category == RequestCategory.LARGE_REPORT

    def test_invalid_category_falls_through(self):
        est = TokenEstimator()
        result = est.estimate("Hi", category="not_a_real_category")
        # Should fall back to heuristic — result still valid
        assert result.category is not None
        assert result.max_tokens > 0

    def test_provider_ceiling_applied(self):
        est = TokenEstimator()
        result = est.estimate(
            "Write a comprehensive report on everything.",
            provider_max_output=1000,
        )
        assert result.max_tokens <= 1000

    def test_provider_ceiling_in_estimation_result(self):
        est = TokenEstimator()
        result = est.estimate("Hi", provider_max_output=256)
        assert result.provider_ceiling == 256
        assert result.max_tokens <= 256

    def test_safety_cap_used_when_no_provider(self):
        config = EstimatorConfig(provider_safety_cap=2000)
        est = TokenEstimator(config=config)
        result = est.estimate("Write a detailed essay.")
        assert result.max_tokens <= 2000

    def test_complexity_score_range(self):
        est = TokenEstimator()
        result = est.estimate("Why is this? How does that work? Explain everything.")
        assert 0.0 <= result.complexity_score <= 1.0

    def test_empty_prompt_handled(self):
        est = TokenEstimator()
        result = est.estimate("")
        assert result.max_tokens > 0
        assert result.estimated_prompt_tokens >= 1

    def test_very_long_prompt(self):
        est = TokenEstimator()
        long_prompt = "word " * 1000
        result = est.estimate(long_prompt)
        assert result.category in (RequestCategory.REASONING, RequestCategory.LONG_EXPLANATION)
        assert result.estimated_prompt_tokens > 200

    def test_fallback_budget(self):
        est = TokenEstimator()
        fb = est.fallback_budget()
        assert fb > 0
        assert fb <= EstimatorConfig().provider_safety_cap

    def test_fallback_budget_with_provider_max(self):
        est = TokenEstimator()
        fb = est.fallback_budget(provider_max_output=300)
        assert fb <= 300

    def test_estimation_result_to_dict(self):
        est = TokenEstimator()
        result = est.estimate("Hello")
        d = result.to_dict()
        assert "category" in d
        assert "max_tokens" in d
        assert "reasoning" in d
        assert isinstance(d["reasoning"], list)

    def test_reasoning_trace_populated(self):
        est = TokenEstimator()
        result = est.estimate("Explain quantum entanglement in detail.")
        assert len(result.reasoning) >= 2

    def test_config_validation_bad_cap(self):
        with pytest.raises(ValueError):
            EstimatorConfig(provider_safety_cap=0).validate()

    def test_config_validation_bad_fallback(self):
        with pytest.raises(ValueError):
            EstimatorConfig(fallback_budget=0).validate()

    def test_multi_step_complexity_uplift(self):
        est = TokenEstimator()
        prompt = (
            "First explain quantum computing. "
            "Then describe how it differs from classical computing. "
            "Next, give three practical applications. "
            "Finally, summarise the limitations."
        )
        result = est.estimate(prompt)
        assert result.complexity_score > 0.2



# ============================================================
# TestRequestValidator
# ============================================================

class TestRequestValidator:

    def _make_validator(self, config=None):
        registry = ProviderRegistry()
        return RequestValidator(registry, config=config)

    def test_passes_valid_request(self):
        v = self._make_validator()
        est = make_estimation(max_tokens=512, prompt_tokens=50)
        report = v.validate("openrouter", "deepseek/deepseek-v4-flash-0731", "Hello", est)
        assert report.passed is True
        assert len(report.errors) == 0

    def test_unknown_provider_fails(self):
        v = self._make_validator()
        est = make_estimation()
        report = v.validate("ghost_provider", "some-model", "Hello", est)
        assert report.passed is False
        assert any(i.check == "provider_known" for i in report.errors)

    def test_prompt_too_large_error(self):
        registry = ProviderRegistry()
        profile = ProviderProfile(
            provider_id="tiny",
            display_name="Tiny",
            context_window=100,
            max_output_tokens=50,
        )
        registry.register(profile)
        v = RequestValidator(registry)
        # 98 tokens > 95% of 100
        est = make_estimation(max_tokens=50, prompt_tokens=98)
        report = v.validate("tiny", "any-model", "x" * 392, est)
        size_issues = [i for i in report.issues if i.check == "request_size"]
        assert any(i.severity == Severity.ERROR for i in size_issues)

    def test_prompt_large_warning(self):
        registry = ProviderRegistry()
        profile = ProviderProfile(
            provider_id="small",
            display_name="Small",
            context_window=1000,
            max_output_tokens=512,
        )
        registry.register(profile)
        v = RequestValidator(registry)
        # 800 tokens = 80% of 1000 → warning territory
        est = make_estimation(max_tokens=256, prompt_tokens=800)
        report = v.validate("small", "any-model", "x", est)
        size_issues = [i for i in report.issues if i.check == "request_size"]
        assert any(i.severity == Severity.WARNING for i in size_issues)

    def test_unsupported_feature_fails(self):
        v = self._make_validator()
        est = make_estimation()
        # ollama does not support FUNCTION_CALLING
        registry = ProviderRegistry()
        v2 = RequestValidator(registry)
        report = v2.validate(
            "ollama", "llama3.1", "Hello", est,
            required_features={"function_calling"},
        )
        assert report.passed is False
        compat_errors = [i for i in report.errors if i.check == "provider_compatibility"]
        assert len(compat_errors) >= 1

    def test_unknown_feature_string_warning(self):
        v = self._make_validator()
        est = make_estimation()
        report = v.validate(
            "openrouter", "deepseek/deepseek-v4-flash-0731", "Hello", est,
            required_features={"not_a_real_feature"},
        )
        compat_warnings = [
            i for i in report.warnings if i.check == "provider_compatibility"
        ]
        assert len(compat_warnings) >= 1

    def test_unknown_model_warning(self):
        v = self._make_validator()
        est = make_estimation()
        report = v.validate("openrouter", "model-not-in-list", "Hello", est)
        model_issues = [i for i in report.issues if i.check == "model_availability"]
        assert len(model_issues) == 1
        assert model_issues[0].severity == Severity.WARNING

    def test_token_budget_exceeds_provider_max_fails(self):
        registry = ProviderRegistry()
        profile = ProviderProfile(
            provider_id="capped",
            display_name="Capped",
            context_window=10000,
            max_output_tokens=256,
            models=["capped-model"],
        )
        registry.register(profile)
        v = RequestValidator(registry)
        est = make_estimation(max_tokens=512)  # 512 > 256
        report = v.validate("capped", "capped-model", "Hello", est)
        budget_errors = [i for i in report.errors if i.check == "token_budget"]
        assert len(budget_errors) == 1

    def test_zero_max_tokens_fails(self):
        v = self._make_validator()
        est = make_estimation(max_tokens=0)
        report = v.validate("openrouter", "deepseek/deepseek-v4-flash-0731", "Hello", est)
        budget_errors = [i for i in report.errors if i.check == "token_budget"]
        assert len(budget_errors) == 1

    def test_report_to_dict(self):
        v = self._make_validator()
        est = make_estimation()
        report = v.validate("openrouter", "deepseek/deepseek-v4-flash-0731", "Hello", est)
        d = report.to_dict()
        assert "passed" in d
        assert "issues" in d
        assert "provider_id" in d

    def test_validation_issue_to_dict(self):
        issue = ValidationIssue(
            check="test_check",
            severity=Severity.ERROR,
            message="Something failed.",
            detail="Detail here.",
        )
        d = issue.to_dict()
        assert d["check"] == "test_check"
        assert d["severity"] == "error"



# ============================================================
# TestResponseValidator
# ============================================================

class TestResponseValidator:

    def test_valid_response_passes(self):
        v = ResponseValidator()
        result = v.validate(make_good_response())
        assert result.passed is True
        assert len(result.failures) == 0

    def test_none_response_fails_empty(self):
        v = ResponseValidator()
        result = v.validate(None)
        axis = result.axis_result(ValidationAxis.EMPTY_RESPONSE)
        assert axis is not None
        assert axis.passed is False

    def test_empty_dict_fails_empty(self):
        v = ResponseValidator()
        result = v.validate({})
        axis = result.axis_result(ValidationAxis.EMPTY_RESPONSE)
        assert axis is not None
        assert axis.passed is False

    def test_empty_string_fails_empty(self):
        v = ResponseValidator()
        result = v.validate("   ")
        axis = result.axis_result(ValidationAxis.EMPTY_RESPONSE)
        assert axis.passed is False

    def test_non_dict_fails_malformed(self):
        v = ResponseValidator()
        result = v.validate("not a dict")
        axis = result.axis_result(ValidationAxis.MALFORMED)
        assert axis.passed is False

    def test_missing_choices_fails_malformed(self):
        v = ResponseValidator()
        result = v.validate({"id": "x"})
        axis = result.axis_result(ValidationAxis.MALFORMED)
        assert axis.passed is False

    def test_empty_choices_fails_malformed(self):
        v = ResponseValidator()
        result = v.validate({"choices": []})
        axis = result.axis_result(ValidationAxis.MALFORMED)
        assert axis.passed is False

    def test_choices_missing_message_fails_malformed(self):
        v = ResponseValidator()
        result = v.validate({"choices": [{"finish_reason": "stop"}]})
        axis = result.axis_result(ValidationAxis.MALFORMED)
        assert axis.passed is False

    def test_valid_tool_calls_pass(self):
        v = ResponseValidator()
        resp = {
            "id": "r1",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": "{}"},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }],
        }
        result = v.validate(resp)
        axis = result.axis_result(ValidationAxis.TOOL_OUTPUT)
        assert axis.passed is True
        assert not axis.skipped

    def test_invalid_tool_calls_fail(self):
        v = ResponseValidator()
        resp = {
            "id": "r1",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"bad": "structure"}],
                },
                "finish_reason": "tool_calls",
            }],
        }
        result = v.validate(resp)
        axis = result.axis_result(ValidationAxis.TOOL_OUTPUT)
        assert axis.passed is False

    def test_no_tool_calls_skips_axis(self):
        v = ResponseValidator()
        result = v.validate(make_good_response())
        axis = result.axis_result(ValidationAxis.TOOL_OUTPUT)
        assert axis.skipped is True

    def test_valid_finish_reason_passes(self):
        v = ResponseValidator()
        result = v.validate(make_good_response(finish_reason="stop"))
        axis = result.axis_result(ValidationAxis.FINISH_REASON)
        assert axis.passed is True

    def test_truncated_finish_reason_fails_validation(self):
        v = ResponseValidator()
        result = v.validate(make_good_response(finish_reason="length"))
        axis = result.axis_result(ValidationAxis.FINISH_REASON)
        assert axis.passed is False

    def test_unknown_finish_reason_fails(self):
        v = ResponseValidator()
        resp = make_good_response()
        resp["choices"][0]["finish_reason"] = "alien_reason"
        result = v.validate(resp)
        axis = result.axis_result(ValidationAxis.FINISH_REASON)
        assert axis.passed is False

    def test_reasoning_blocks_string_passes(self):
        v = ResponseValidator()
        resp = make_good_response()
        resp["choices"][0]["message"]["reasoning_content"] = "I thought about it carefully."
        result = v.validate(resp)
        axis = result.axis_result(ValidationAxis.REASONING_BLOCKS)
        assert axis.passed is True

    def test_reasoning_blocks_valid_list_passes(self):
        v = ResponseValidator()
        resp = make_good_response()
        resp["choices"][0]["message"]["thinking"] = [
            {"type": "thinking", "thinking": "step 1"},
            {"type": "thinking", "thinking": "step 2"},
        ]
        result = v.validate(resp)
        axis = result.axis_result(ValidationAxis.REASONING_BLOCKS)
        assert axis.passed is True

    def test_reasoning_blocks_invalid_type_fails(self):
        v = ResponseValidator()
        resp = make_good_response()
        resp["choices"][0]["message"]["reasoning_content"] = 12345
        result = v.validate(resp)
        axis = result.axis_result(ValidationAxis.REASONING_BLOCKS)
        assert axis.passed is False

    def test_empty_content_fails_message_content(self):
        v = ResponseValidator()
        resp = make_good_response(content="   ")
        result = v.validate(resp)
        axis = result.axis_result(ValidationAxis.MESSAGE_CONTENT)
        assert axis.passed is False

    def test_multimodal_content_list_passes(self):
        v = ResponseValidator()
        resp = {
            "id": "r1",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Hi"}],
                },
                "finish_reason": "stop",
            }],
        }
        result = v.validate(resp)
        axis = result.axis_result(ValidationAxis.MESSAGE_CONTENT)
        assert axis.passed is True

    def test_result_to_dict(self):
        v = ResponseValidator()
        result = v.validate(make_good_response())
        d = result.to_dict()
        assert "passed" in d
        assert "axes" in d
        assert isinstance(d["axes"], list)

    def test_response_id_extracted(self):
        v = ResponseValidator()
        result = v.validate(make_good_response())
        assert result.response_id == "resp_abc123"



# ============================================================
# TestRetryManager
# ============================================================

class TestRetryManager:

    def _make_failed_report(self, check="token_budget", severity=Severity.ERROR):
        return ValidationReport(
            passed=False,
            issues=[ValidationIssue(check=check, severity=severity, message="fail")],
            provider_id="openrouter",
            model_id="model",
            estimated_max_tokens=512,
        )

    def _make_passed_report(self):
        return ValidationReport(
            passed=True,
            issues=[],
            provider_id="openrouter",
            model_id="model",
            estimated_max_tokens=512,
        )

    def _make_failed_result(self, axis=ValidationAxis.EMPTY_RESPONSE):
        axes = [
            AxisResult(axis=axis, passed=False, message="failed"),
        ]
        return ValidationResult(passed=False, axes=axes)

    def _make_passed_result(self):
        axes = [
            AxisResult(axis=ax, passed=True)
            for ax in ValidationAxis
        ]
        return ValidationResult(passed=True, axes=axes)

    def test_passed_report_returns_success(self):
        rm = RetryManager()
        outcome = rm.evaluate_request_failure(self._make_passed_report(), 512, 1)
        assert outcome.status == RetryStatus.SUCCESS

    def test_token_failure_retries_with_reduced_budget(self):
        rm = RetryManager(RetryConfig(max_attempts=3, token_reduction_factor=0.5))
        report = self._make_failed_report(check="token_budget")
        outcome = rm.evaluate_request_failure(report, 512, 1)
        assert outcome.should_retry is True
        assert outcome.adjusted_max_tokens == 256

    def test_budget_floor_respected(self):
        rm = RetryManager(RetryConfig(max_attempts=3, token_reduction_factor=0.5, min_tokens=300))
        report = self._make_failed_report(check="token_budget")
        outcome = rm.evaluate_request_failure(report, 512, 1)
        assert outcome.adjusted_max_tokens >= 300

    def test_max_attempts_reached_gives_up(self):
        rm = RetryManager(RetryConfig(max_attempts=2))
        report = self._make_failed_report(check="token_budget")
        outcome = rm.evaluate_request_failure(report, 512, 2)
        assert outcome.status == RetryStatus.GIVE_UP

    def test_provider_failure_no_fallback_no_retry(self):
        rm = RetryManager(RetryConfig(fallback_provider_id=None))
        report = self._make_failed_report(check="provider_known")
        outcome = rm.evaluate_request_failure(report, 512, 1)
        assert outcome.status == RetryStatus.NO_RETRY

    def test_provider_failure_with_fallback_retries(self):
        rm = RetryManager(RetryConfig(fallback_provider_id="anthropic"))
        report = self._make_failed_report(check="provider_known")
        outcome = rm.evaluate_request_failure(report, 512, 1)
        assert outcome.should_retry is True
        assert outcome.adjusted_provider_id == "anthropic"

    def test_model_failure_with_fallback_retries(self):
        rm = RetryManager(RetryConfig(fallback_model_id="gpt-4o-mini"))
        report = self._make_failed_report(check="model_availability", severity=Severity.ERROR)
        # model_availability is WARNING by default, so we need ERROR for it to block
        outcome = rm.evaluate_request_failure(report, 512, 1)
        # token budget is fine; model check produces an adjustment note
        assert outcome.status in (RetryStatus.RETRY, RetryStatus.NO_RETRY)

    def test_passed_response_returns_success(self):
        rm = RetryManager()
        outcome = rm.evaluate_response_failure(self._make_passed_result(), 512, 1)
        assert outcome.status == RetryStatus.SUCCESS

    def test_empty_response_retries(self):
        rm = RetryManager(RetryConfig(max_attempts=3))
        result = self._make_failed_result(axis=ValidationAxis.EMPTY_RESPONSE)
        outcome = rm.evaluate_response_failure(result, 512, 1)
        assert outcome.should_retry is True

    def test_malformed_response_no_retry(self):
        rm = RetryManager()
        result = self._make_failed_result(axis=ValidationAxis.MALFORMED)
        outcome = rm.evaluate_response_failure(result, 512, 1)
        assert outcome.status == RetryStatus.NO_RETRY

    def test_response_max_attempts_gives_up(self):
        rm = RetryManager(RetryConfig(max_attempts=2))
        result = self._make_failed_result(axis=ValidationAxis.EMPTY_RESPONSE)
        outcome = rm.evaluate_response_failure(result, 512, 2)
        assert outcome.status == RetryStatus.GIVE_UP

    def test_retry_outcome_to_dict(self):
        rm = RetryManager()
        report = self._make_failed_report()
        outcome = rm.evaluate_request_failure(report, 512, 1)
        d = outcome.to_dict()
        assert "status" in d
        assert "attempt_number" in d
        assert "adjustments" in d

    def test_config_validation_bad_max_attempts(self):
        with pytest.raises(ValueError):
            RetryConfig(max_attempts=0).validate()

    def test_config_validation_bad_reduction_factor(self):
        with pytest.raises(ValueError):
            RetryConfig(token_reduction_factor=1.5).validate()



# ============================================================
# TestRequestLogger
# ============================================================

class TestRequestLogger:

    def test_record_created_and_stored(self):
        log = RequestLogger()
        rec = log.record(
            provider="openrouter", model="m1",
            token_estimate=512, actual_tokens=512,
            attempt_number=1, request_passed=True,
            response_passed=True, retry_status="success",
        )
        assert isinstance(rec, RequestRecord)
        assert len(log.all_records()) == 1

    def test_record_id_is_unique(self):
        log = RequestLogger()
        r1 = log.record("p", "m", 100, 100, 1, True, True, "success")
        r2 = log.record("p", "m", 100, 100, 2, True, True, "success")
        assert r1.record_id != r2.record_id

    def test_recent_returns_n(self):
        log = RequestLogger()
        for i in range(10):
            log.record("p", "m", 100, 100, i+1, True, True, "success")
        recent = log.recent(5)
        assert len(recent) == 5

    def test_all_records_order(self):
        log = RequestLogger()
        log.record("p", "m", 100, 100, 1, True, True, "success")
        log.record("p", "m", 200, 200, 2, False, False, "retry")
        records = log.all_records()
        assert records[0].actual_tokens == 100
        assert records[1].actual_tokens == 200

    def test_clear_empties_buffer(self):
        log = RequestLogger()
        log.record("p", "m", 100, 100, 1, True, True, "success")
        log.clear()
        assert len(log.all_records()) == 0

    def test_max_records_bounded(self):
        config = LoggerConfig(max_records=5)
        log = RequestLogger(config=config)
        for i in range(10):
            log.record("p", "m", 100, 100, i+1, True, True, "success")
        assert len(log.all_records()) == 5

    def test_summary_stats_empty(self):
        log = RequestLogger()
        stats = log.summary_stats()
        assert stats["total_attempts"] == 0
        assert stats["avg_latency_ms"] is None

    def test_summary_stats_counts(self):
        log = RequestLogger()
        log.record("p", "m", 100, 100, 1, True, True, "success", latency_ms=100.0)
        log.record("p", "m", 100, 100, 2, True, False, "retry", latency_ms=200.0)
        stats = log.summary_stats()
        assert stats["total_attempts"] == 2
        assert stats["successful"] == 1
        assert stats["failed"] == 1
        assert stats["retried"] == 1
        assert stats["avg_latency_ms"] == 150.0

    def test_summary_stats_providers(self):
        log = RequestLogger()
        log.record("openrouter", "m", 100, 100, 1, True, True, "success")
        log.record("anthropic", "m", 100, 100, 1, True, True, "success")
        log.record("openrouter", "m", 100, 100, 2, True, True, "success")
        stats = log.summary_stats()
        assert stats["providers"]["openrouter"] == 2
        assert stats["providers"]["anthropic"] == 1

    def test_overall_success_property(self):
        rec = RequestRecord(
            record_id="abc", provider="p", model="m",
            token_estimate=100, actual_tokens=100,
            attempt_number=1, request_passed=True,
            response_passed=True, retry_status="success",
        )
        assert rec.overall_success is True

    def test_overall_success_false_when_response_failed(self):
        rec = RequestRecord(
            record_id="abc", provider="p", model="m",
            token_estimate=100, actual_tokens=100,
            attempt_number=1, request_passed=True,
            response_passed=False, retry_status="retry",
        )
        assert rec.overall_success is False

    def test_record_to_dict(self):
        log = RequestLogger()
        rec = log.record("p", "m", 100, 100, 1, True, True, "success")
        d = rec.to_dict()
        assert "record_id" in d
        assert "provider" in d
        assert "timestamp_utc" in d


# ============================================================
# TestAdaptiveCloudBrain
# ============================================================

class TestAdaptiveCloudBrain:

    def _make_brain(self):
        return AdaptiveCloudBrain()

    def test_prepare_returns_prep_result(self):
        brain = self._make_brain()
        prep = brain.prepare("openrouter", "deepseek/deepseek-v4-flash-0731", "Hello")
        assert isinstance(prep, PrepResult)

    def test_prepare_known_provider_is_ready(self):
        brain = self._make_brain()
        prep = brain.prepare("openrouter", "deepseek/deepseek-v4-flash-0731", "Hello")
        assert prep.ready is True

    def test_prepare_unknown_provider_not_ready(self):
        brain = self._make_brain()
        prep = brain.prepare("ghost", "model", "Hello")
        assert prep.ready is False

    def test_prepare_max_tokens_positive(self):
        brain = self._make_brain()
        prep = brain.prepare("openrouter", "deepseek/deepseek-v4-flash-0731", "Hello")
        assert prep.max_tokens > 0

    def test_prepare_to_dict(self):
        brain = self._make_brain()
        prep = brain.prepare("openrouter", "deepseek/deepseek-v4-flash-0731", "Hello")
        d = prep.to_dict()
        assert "provider_id" in d
        assert "max_tokens" in d
        assert "ready" in d

    def test_full_pipeline_passed(self):
        brain = self._make_brain()
        result = brain.prepare_and_validate(
            "openrouter", "deepseek/deepseek-v4-flash-0731",
            "Hello", raw_response=make_good_response(),
        )
        assert result.status == BrainStatus.PASSED
        assert result.passed is True

    def test_response_validation_reuses_actual_request_budget(self):
        brain = self._make_brain()
        prep = brain.prepare(
            "openrouter",
            "deepseek/deepseek-v4-flash-0731",
            "System prompt and recent context.\nUser: brainstorm this.",
            task_text="Brainstorm practical options.",
        )

        result = brain.validate_response(make_good_response(), prep)

        assert result.status == BrainStatus.PASSED
        assert result.max_tokens_used == prep.max_tokens
        assert result.estimation == prep.estimation

    def test_full_pipeline_no_response_returns_ready(self):
        brain = self._make_brain()
        result = brain.prepare_and_validate(
            "openrouter", "deepseek/deepseek-v4-flash-0731", "Hello",
        )
        assert result.status == BrainStatus.READY

    def test_full_pipeline_bad_provider_returns_request_failed_or_retrying(self):
        brain = self._make_brain()
        result = brain.prepare_and_validate(
            "ghost", "model", "Hello", raw_response=make_good_response(),
        )
        assert result.status in (
            BrainStatus.REQUEST_FAILED,
            BrainStatus.RETRYING,
            BrainStatus.GAVE_UP,
        )
        assert result.passed is False

    def test_full_pipeline_bad_response_returns_response_failed_or_retrying(self):
        brain = self._make_brain()
        result = brain.prepare_and_validate(
            "openrouter", "deepseek/deepseek-v4-flash-0731",
            "Hello", raw_response=None.__class__,
        )
        assert result.status in (
            BrainStatus.RESPONSE_FAILED,
            BrainStatus.RETRYING,
            BrainStatus.GAVE_UP,
        )

    def test_log_record_attached(self):
        brain = self._make_brain()
        result = brain.prepare_and_validate(
            "openrouter", "deepseek/deepseek-v4-flash-0731",
            "Hello", raw_response=make_good_response(),
        )
        assert result.log_record is not None
        assert result.log_record.provider == "openrouter"

    def test_logger_stats_after_call(self):
        brain = self._make_brain()
        brain.prepare_and_validate(
            "openrouter", "deepseek/deepseek-v4-flash-0731",
            "Hello", raw_response=make_good_response(),
        )
        stats = brain.logger_stats()
        assert stats["total_attempts"] == 1
        assert stats["successful"] == 1

    def test_result_to_dict(self):
        brain = self._make_brain()
        result = brain.prepare_and_validate(
            "openrouter", "deepseek/deepseek-v4-flash-0731",
            "Hello", raw_response=make_good_response(),
        )
        d = result.to_dict()
        assert "status" in d
        assert "passed" in d
        assert "estimation" in d
        assert "request_report" in d

    def test_component_accessors(self):
        brain = self._make_brain()
        assert brain.provider_registry is not None
        assert brain.token_estimator is not None
        assert brain.request_validator is not None
        assert brain.response_validator is not None
        assert brain.retry_manager is not None
        assert brain.request_logger is not None

    def test_custom_injection(self):
        registry = ProviderRegistry(load_defaults=False)
        brain = AdaptiveCloudBrain(provider_registry=registry)
        prep = brain.prepare("openrouter", "model", "Hello")
        assert prep.ready is False  # openrouter not in empty registry

    def test_intent_passed_to_estimator(self):
        brain = self._make_brain()
        prep = brain.prepare(
            "openrouter", "deepseek/deepseek-v4-flash-0731",
            "Hi", intent="greeting",
        )
        assert prep.estimation.category.value == "greeting"

    def test_category_passed_to_estimator(self):
        brain = self._make_brain()
        prep = brain.prepare(
            "openrouter", "deepseek/deepseek-v4-flash-0731",
            "Hello", category="reasoning",
        )
        assert prep.estimation.category.value == "reasoning"

    def test_attempt_number_in_result(self):
        brain = self._make_brain()
        result = brain.prepare_and_validate(
            "openrouter", "deepseek/deepseek-v4-flash-0731",
            "Hello", raw_response=make_good_response(),
            attempt_number=2,
        )
        assert result.attempt_number == 2
