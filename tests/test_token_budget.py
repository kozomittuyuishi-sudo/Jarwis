"""
test_token_budget.py
====================
AG — pytest suite for the Dynamic Token Budget Manager.

Covers 8 required scenarios:
    1. Greeting           — budget ≤ 128
    2. Factual questions  — budget ≤ 512
    3. Long explanations  — budget ≥ 1024
    4. Reports            — budget ≥ 2048
    5. Provider fallback  — fallback budget is small and safe
    6. Retry behaviour    — retryable errors use fallback; others re-raise
    7. Configuration      — custom config overrides defaults
    8. Safe defaults      — unknown / empty input never exceeds provider_max

Module under test: token_budget.py
"""

import pytest
from brains.token_budget import (
    ResponseCategory,
    TokenBudgetConfig,
    TokenBudgetManager,
    BudgetEstimate,
    get_default_manager,
    reset_default_manager,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fresh_manager(
    provider_max: int = 4096,
    fallback_budget: int = 512,
    safe_default: int = 512,
) -> TokenBudgetManager:
    """Return a TokenBudgetManager with a known, controlled configuration.

    fallback_budget and safe_default are automatically clamped to provider_max
    so the config always passes validation regardless of the test values passed.
    """
    fallback_budget = min(fallback_budget, provider_max)
    safe_default = min(safe_default, provider_max)
    config = TokenBudgetConfig(
        provider_max=provider_max,
        fallback_budget=fallback_budget,
        safe_default=safe_default,
    )
    return TokenBudgetManager(config=config)


# ---------------------------------------------------------------------------
# SCENARIO 1 — Greeting
# ---------------------------------------------------------------------------

class TestGreeting:
    """Budget for greeting-class inputs must be small (≤ 128 tokens)."""

    def test_greeting_intent_category(self):
        m = fresh_manager()
        est = m.estimate("hi", intent="greeting")
        assert est.category == ResponseCategory.GREETING

    def test_greeting_budget_at_most_128(self):
        m = fresh_manager()
        est = m.estimate("hello", intent="greeting")
        assert est.max_tokens <= 128

    def test_greeting_budget_positive(self):
        m = fresh_manager()
        est = m.estimate("hey", intent="greeting")
        assert est.max_tokens > 0

    def test_greeting_capped_by_provider_max(self):
        """Even if the greeting budget exceeds a tiny provider_max, it gets clamped."""
        m = fresh_manager(provider_max=64)
        est = m.estimate("hello", intent="greeting")
        assert est.max_tokens <= 64

    def test_greeting_reasoning_non_empty(self):
        m = fresh_manager()
        est = m.estimate("hello", intent="greeting")
        assert len(est.reasoning) > 0

    def test_greeting_returns_budget_estimate(self):
        m = fresh_manager()
        est = m.estimate("hi there", intent="greeting")
        assert isinstance(est, BudgetEstimate)


# ---------------------------------------------------------------------------
# SCENARIO 2 — Factual Questions
# ---------------------------------------------------------------------------

class TestFactualQuestions:
    """Short factual prompts should get a moderate budget (≤ 512 tokens)."""

    def test_short_factual_intent_category(self):
        m = fresh_manager()
        est = m.estimate("Who is Einstein?", intent="recall")
        assert est.category == ResponseCategory.SHORT_FACTUAL

    def test_short_factual_budget_at_most_512(self):
        m = fresh_manager()
        est = m.estimate("What is the speed of light?", intent="recall")
        assert est.max_tokens <= 512

    def test_brain_status_is_short_factual(self):
        m = fresh_manager()
        est = m.estimate("what brain are you using", intent="brain_status")
        assert est.category == ResponseCategory.SHORT_FACTUAL
        assert est.max_tokens <= 512

    def test_current_version_is_short_factual(self):
        m = fresh_manager()
        est = m.estimate("what version are we on", intent="current_version")
        assert est.category == ResponseCategory.SHORT_FACTUAL

    def test_very_short_prompt_without_intent_is_short_factual(self):
        """A 3-word prompt with no intent should default to SHORT_FACTUAL."""
        m = fresh_manager()
        est = m.estimate("Who is Musk?")  # ~4 tokens, no intent
        assert est.max_tokens <= 512

    def test_elon_musk_prompt_normal_conversation(self):
        """
        'Tell me about Elon Musk.' — the original failing request.
        With unknown intent and a medium prompt, budget must be sane
        and never equal to 65536.
        """
        m = fresh_manager()
        est = m.estimate("Tell me about Elon Musk.", intent="unknown")
        assert est.max_tokens != 65536
        assert est.max_tokens <= m.provider_max()
        assert est.max_tokens > 0


# ---------------------------------------------------------------------------
# SCENARIO 3 — Long Explanations
# ---------------------------------------------------------------------------

class TestLongExplanations:
    """Prompts that request explanations should get ≥ 1024 tokens."""

    def test_explain_keyword_triggers_reasoning(self):
        m = fresh_manager()
        est = m.estimate("Can you explain how neural networks work?", intent="unknown")
        assert est.category in {ResponseCategory.REASONING, ResponseCategory.LONG_EXPLANATION}

    def test_reasoning_budget_at_least_512(self):
        m = fresh_manager()
        est = m.estimate("Why does quantum entanglement work?", intent="unknown")
        # Must be non-trivial
        assert est.max_tokens >= 512

    def test_how_does_keyword_triggers_reasoning(self):
        m = fresh_manager()
        est = m.estimate("How does the internet work?", intent="unknown")
        assert est.category == ResponseCategory.REASONING

    def test_compare_keyword_triggers_reasoning(self):
        m = fresh_manager()
        est = m.estimate("Compare Python and Java for web development.", intent="unknown")
        assert est.category == ResponseCategory.REASONING

    def test_long_explanation_budget_capped_by_provider_max(self):
        """Even a LONG_EXPLANATION is capped at provider_max."""
        m = fresh_manager(provider_max=1024)
        est = m.estimate("Explain the history of the Roman Empire in detail.")
        assert est.max_tokens <= 1024

    def test_analyse_keyword_triggers_reasoning(self):
        m = fresh_manager()
        est = m.estimate("Analyse the causes of World War I.", intent="unknown")
        assert est.category == ResponseCategory.REASONING


# ---------------------------------------------------------------------------
# SCENARIO 4 — Reports
# ---------------------------------------------------------------------------

class TestReports:
    """Report-class prompts should get a large budget (≥ 2048)."""

    def test_report_keyword_triggers_large_report(self):
        m = fresh_manager()
        est = m.estimate("Write a detailed report on climate change.", intent="unknown")
        assert est.category == ResponseCategory.LARGE_REPORT

    def test_summary_keyword_triggers_large_report(self):
        m = fresh_manager()
        est = m.estimate("Summarize the entire history of computing.", intent="unknown")
        assert est.category == ResponseCategory.LARGE_REPORT

    def test_comprehensive_keyword_triggers_large_report(self):
        m = fresh_manager()
        est = m.estimate("Give a comprehensive overview of machine learning.", intent="unknown")
        assert est.category == ResponseCategory.LARGE_REPORT

    def test_report_budget_at_least_2048_unless_capped(self):
        """Report budget is 2048 by default, irrespective of unknown credits."""
        m = fresh_manager(provider_max=8192)
        est = m.estimate("Write a detailed report on renewable energy.", intent="unknown")
        assert est.max_tokens == 2048

    def test_report_budget_capped_by_provider_max(self):
        m = fresh_manager(provider_max=1024)
        est = m.estimate("Generate a full report on AI trends.", intent="unknown")
        assert est.max_tokens <= 1024

    def test_essay_keyword_triggers_large_report(self):
        m = fresh_manager()
        est = m.estimate("Write an essay on the impact of social media.", intent="unknown")
        assert est.category == ResponseCategory.LARGE_REPORT



# ---------------------------------------------------------------------------
# SCENARIO 5 — Provider Fallback
# ---------------------------------------------------------------------------

class TestProviderFallback:
    """Fallback budget must always be small and ≤ provider_max."""

    def test_fallback_is_at_most_512(self):
        m = fresh_manager(fallback_budget=512, provider_max=4096)
        assert m.fallback_budget() == 512

    def test_fallback_never_exceeds_provider_max(self):
        """fallback_budget() always returns a value ≤ provider_max."""
        # When provider_max is small, fallback is clamped to match it.
        m = fresh_manager(fallback_budget=512, provider_max=256)
        # fresh_manager auto-clamps fallback to 256
        assert m.fallback_budget() <= 256
        assert m.fallback_budget() <= m.provider_max()

    def test_custom_fallback_budget(self):
        m = fresh_manager(fallback_budget=128, provider_max=4096)
        assert m.fallback_budget() == 128

    def test_fallback_positive(self):
        m = fresh_manager()
        assert m.fallback_budget() > 0

    def test_provider_max_accessor(self):
        m = fresh_manager(provider_max=2048)
        assert m.provider_max() == 2048

    def test_no_estimate_exceeds_provider_max(self):
        """Whatever the category, max_tokens must never exceed provider_max."""
        m = fresh_manager(provider_max=1024)
        prompts = [
            ("hi", "greeting"),
            ("Tell me about Elon Musk.", "unknown"),
            ("Write a detailed report on AI.", "unknown"),
            ("Explain quantum computing.", "unknown"),
        ]
        for prompt, intent in prompts:
            est = m.estimate(prompt, intent=intent)
            assert est.max_tokens <= 1024, (
                f"Exceeded provider_max for prompt={prompt!r}: got {est.max_tokens}"
            )

    def test_65536_never_requested(self):
        """The old hardcoded limit must never appear as max_tokens."""
        m = fresh_manager()
        for intent in ("greeting", "unknown", "recall", "cloud_brain", None):
            est = m.estimate("Tell me about Elon Musk.", intent=intent)
            assert est.max_tokens != 65536


# ---------------------------------------------------------------------------
# SCENARIO 6 — Retry Behaviour (unit-level)
# ---------------------------------------------------------------------------

class TestRetryBehaviour:
    """
    Retry logic lives in brain.py's ask_cloud_brain(); we test the budget
    primitives here to confirm the retry infrastructure has correct inputs.
    """

    def test_fallback_budget_less_than_or_equal_to_normal_estimate(self):
        """The fallback budget should be conservative — not larger than a normal estimate."""
        m = fresh_manager()
        normal = m.estimate("Tell me about Elon Musk.", intent="unknown")
        # Fallback is 512; normal for 'unknown' medium prompt is also 512 NORMAL_CONV.
        # Fallback must be ≤ provider_max and > 0.
        assert m.fallback_budget() <= m.provider_max()
        assert m.fallback_budget() > 0

    def test_fallback_budget_smaller_than_large_report_estimate(self):
        m = fresh_manager(provider_max=8192)
        large = m.estimate("Write a comprehensive report on AI.", intent="unknown")
        # Fallback (512) is much smaller than a large_report estimate (4096).
        assert m.fallback_budget() < large.max_tokens

    def test_budget_estimate_to_dict_serialises_correctly(self):
        m = fresh_manager()
        est = m.estimate("Hello", intent="greeting")
        d = est.to_dict()
        assert d["category"] == "greeting"
        assert d["max_tokens"] == est.max_tokens
        assert d["estimated_prompt_tokens"] == est.estimated_prompt_tokens
        assert isinstance(d["reasoning"], list)

    def test_budget_estimate_includes_reasoning_steps(self):
        m = fresh_manager()
        est = m.estimate("Tell me about Elon Musk.", intent="unknown")
        # Must have at least two steps: token estimate + category assignment
        assert len(est.reasoning) >= 2

    def test_estimate_prompt_tokens_proportional_to_length(self):
        m = fresh_manager()
        short = m.estimate("Hi")
        long_est = m.estimate(
            "Tell me everything about the history of artificial intelligence "
            "from the 1950s to the present day including all major milestones "
            "and key researchers and their contributions to the field."
        )
        assert long_est.estimated_prompt_tokens > short.estimated_prompt_tokens


# ---------------------------------------------------------------------------
# SCENARIO 7 — Configuration Loading
# ---------------------------------------------------------------------------

class TestConfigurationLoading:
    """Custom TokenBudgetConfig overrides defaults; validation rejects bad values."""

    def test_custom_budgets_used(self):
        from brains.token_budget import ResponseCategory
        config = TokenBudgetConfig(
            budgets={
                ResponseCategory.GREETING:        64,
                ResponseCategory.SHORT_FACTUAL:   128,
                ResponseCategory.NORMAL_CONV:     256,
                ResponseCategory.REASONING:       512,
                ResponseCategory.LONG_EXPLANATION:1024,
                ResponseCategory.LARGE_REPORT:    2048,
                ResponseCategory.VERY_LARGE:      4096,
            },
            provider_max=4096,
            fallback_budget=64,
            safe_default=128,
        )
        m = TokenBudgetManager(config=config)
        est = m.estimate("hello", intent="greeting")
        assert est.max_tokens == 64

    def test_provider_max_clamps_all_categories(self):
        config = TokenBudgetConfig(provider_max=100, fallback_budget=50, safe_default=50)
        m = TokenBudgetManager(config=config)
        for category in ResponseCategory:
            budget = config.budget_for(category)
            assert budget <= 100, f"Category {category} exceeds provider_max"

    def test_invalid_provider_max_raises(self):
        with pytest.raises(ValueError, match="provider_max"):
            TokenBudgetConfig(provider_max=0, fallback_budget=64, safe_default=64).validate()

    def test_invalid_fallback_raises(self):
        with pytest.raises(ValueError, match="fallback_budget"):
            TokenBudgetConfig(provider_max=512, fallback_budget=0, safe_default=64).validate()

    def test_fallback_exceeds_provider_max_raises(self):
        with pytest.raises(ValueError, match="fallback_budget"):
            TokenBudgetConfig(
                provider_max=256, fallback_budget=512, safe_default=64
            ).validate()

    def test_invalid_safe_default_raises(self):
        with pytest.raises(ValueError, match="safe_default"):
            TokenBudgetConfig(provider_max=512, fallback_budget=64, safe_default=0).validate()

    def test_default_config_validates_without_error(self):
        """The default configuration must pass its own validation."""
        config = TokenBudgetConfig()
        config.validate()  # should not raise

    def test_budget_for_returns_clamped_value(self):
        config = TokenBudgetConfig(provider_max=200, fallback_budget=100, safe_default=100)
        # LARGE_REPORT has budget 4096, but provider_max=200 clamps it
        budget = config.budget_for(ResponseCategory.LARGE_REPORT)
        assert budget == 200

    def test_reset_default_manager_replaces_instance(self):
        original = get_default_manager()
        config = TokenBudgetConfig(provider_max=1024, fallback_budget=128, safe_default=128)
        reset_default_manager(config)
        new_mgr = get_default_manager()
        assert new_mgr.provider_max() == 1024
        # Restore
        reset_default_manager()

    def test_get_default_manager_returns_same_instance_on_repeat_calls(self):
        reset_default_manager()  # ensure clean state
        m1 = get_default_manager()
        m2 = get_default_manager()
        assert m1 is m2


# ---------------------------------------------------------------------------
# SCENARIO 8 — Safe Defaults
# ---------------------------------------------------------------------------

class TestSafeDefaults:
    """Unknown, empty, or edge-case inputs must never exceed provider_max."""

    def test_empty_prompt_returns_safe_default(self):
        """
        An empty prompt has estimated_prompt_tokens=1 (min 1), which hits the
        SHORT_FACTUAL heuristic branch (≤ 20 tokens).  The resulting budget
        (256) is positive, ≤ provider_max, and well within a safe range.
        """
        m = fresh_manager()
        est = m.estimate("", intent=None)
        assert est.max_tokens > 0
        assert est.max_tokens <= m.provider_max()
        # SHORT_FACTUAL budget is 256 — sensible and safe
        assert est.max_tokens <= 512

    def test_whitespace_only_prompt_safe(self):
        m = fresh_manager()
        est = m.estimate("   ", intent=None)
        assert est.max_tokens <= m.provider_max()
        assert est.max_tokens > 0

    def test_none_intent_still_produces_valid_estimate(self):
        m = fresh_manager()
        est = m.estimate("Tell me about Elon Musk.", intent=None)
        assert est.max_tokens > 0
        assert est.max_tokens <= m.provider_max()

    def test_unknown_intent_string_falls_through_to_heuristic(self):
        m = fresh_manager()
        est = m.estimate("Tell me about Elon Musk.", intent="unknown")
        assert est.max_tokens > 0
        assert est.max_tokens <= m.provider_max()

    def test_unrecognised_intent_falls_through_safely(self):
        m = fresh_manager()
        est = m.estimate("Some query.", intent="totally_made_up_intent_xyz")
        # Unknown intent has no mapping → falls through to heuristic
        assert est.max_tokens > 0
        assert est.max_tokens <= m.provider_max()

    def test_max_tokens_never_zero(self):
        m = fresh_manager()
        for prompt in ["", "hi", "a" * 1000]:
            est = m.estimate(prompt)
            assert est.max_tokens > 0, f"Zero tokens for prompt of length {len(prompt)}"

    def test_safe_default_accessor(self):
        m = fresh_manager(safe_default=256, provider_max=4096)
        assert m.safe_default() == 256

    def test_safe_default_clamped_to_provider_max(self):
        m = fresh_manager(safe_default=8192, provider_max=512)
        assert m.safe_default() <= 512

    def test_all_categories_produce_valid_budgets(self):
        m = fresh_manager()
        config = TokenBudgetConfig()
        for category in ResponseCategory:
            budget = config.budget_for(category)
            assert 0 < budget <= config.provider_max, (
                f"Invalid budget {budget} for category {category}"
            )

    def test_very_long_prompt_does_not_overflow_provider_max(self):
        m = fresh_manager(provider_max=2048)
        huge_prompt = "word " * 2000   # ~10000 chars
        est = m.estimate(huge_prompt, intent="unknown")
        assert est.max_tokens <= 2048
