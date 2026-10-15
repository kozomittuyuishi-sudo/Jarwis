"""
token_budget.py
===============
AG — Dynamic Token Budget Manager for the Cloud Brain.

Problem solved
--------------
The previous implementation sent cloud requests with no ``max_tokens``
parameter, allowing OpenRouter to apply its model default (65536 tokens).
That default exceeds typical account credit limits and causes 402 errors
before the model has reasoned about anything.

Solution
--------
Classify every incoming request into a ``ResponseCategory`` based on
prompt length and intent, then look up a sensible upper-bound token budget
from ``TokenBudgetConfig``.  Never request more than the configured
provider safety cap.  If the provider still rejects the budget, the caller
can retry with the fallback budget.

Design principles
-----------------
- Deterministic, stdlib-only. No LLM calls, no I/O.
- All limits are in ``TokenBudgetConfig`` — nothing hardcoded in logic.
- ``TokenBudgetManager.estimate()`` returns the SMALLEST practical budget,
  not the largest allowed one.
- The provider_max acts as an absolute ceiling: no request will ever exceed
  it regardless of what the estimator computes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response categories
# ---------------------------------------------------------------------------

class ResponseCategory(str, Enum):
    """
    Coarse classification of the expected response size.

    The string value is used in log messages and serialisation.
    """
    GREETING          = "greeting"           # 128  tokens
    SHORT_FACTUAL     = "short_factual"      # 256  tokens
    NORMAL_CONV       = "normal_conversation"# 512  tokens
    REASONING         = "reasoning"          # 1024 tokens
    LONG_EXPLANATION  = "long_explanation"   # 2048 tokens
    LARGE_REPORT      = "large_report"       # 2048 tokens
    VERY_LARGE        = "very_large"         # 2048 tokens


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TokenBudgetConfig:
    """
    All token limits in one place.  Adjust here; never touch business logic.

    Attributes
    ----------
    budgets : Dict[ResponseCategory, int]
        Maximum output tokens per category.
    provider_max : int
        Hard ceiling enforced regardless of category budget.
        Set to a conservative value that fits within typical credit tiers.
    fallback_budget : int
        Budget to retry with after a provider rejection.
    safe_default : int
        Used when the category cannot be determined.
    """
    budgets: Dict[ResponseCategory, int] = field(default_factory=lambda: {
        ResponseCategory.GREETING:         128,
        ResponseCategory.SHORT_FACTUAL:    256,
        ResponseCategory.NORMAL_CONV:      384,
        ResponseCategory.REASONING:       1024,
        ResponseCategory.LONG_EXPLANATION:1536,
        ResponseCategory.LARGE_REPORT:    2048,
        ResponseCategory.VERY_LARGE:      2048,
    })
    provider_max: int   = 2048   # output cap; account credit availability is unknown
    fallback_budget: int = 512   # retry budget after a retryable request rejection
    safe_default: int   = 512    # used when category is unknown

    def budget_for(self, category: ResponseCategory) -> int:
        """Return the token budget for ``category``, clamped to provider_max."""
        raw = self.budgets.get(category, self.safe_default)
        return min(raw, self.provider_max)

    def validate(self) -> None:
        """Raise ValueError if configuration is internally inconsistent."""
        if self.provider_max <= 0:
            raise ValueError(f"provider_max must be > 0, got {self.provider_max}")
        if self.fallback_budget <= 0:
            raise ValueError(f"fallback_budget must be > 0, got {self.fallback_budget}")
        if self.fallback_budget > self.provider_max:
            raise ValueError(
                f"fallback_budget ({self.fallback_budget}) must not exceed "
                f"provider_max ({self.provider_max})"
            )
        if self.safe_default <= 0:
            raise ValueError(f"safe_default must be > 0, got {self.safe_default}")


# ---------------------------------------------------------------------------
# Budget estimation result
# ---------------------------------------------------------------------------

@dataclass
class BudgetEstimate:
    """
    The result of a single budget estimation call.

    Attributes
    ----------
    category : ResponseCategory
        The detected response category.
    max_tokens : int
        The recommended ``max_tokens`` value to send to the provider.
    estimated_prompt_tokens : int
        Rough estimate of the prompt's token count (4 chars ≈ 1 token).
    reasoning : List[str]
        Step-by-step explanation of how the category was chosen.
    """
    category: ResponseCategory
    max_tokens: int
    estimated_prompt_tokens: int
    reasoning: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "max_tokens": self.max_tokens,
            "estimated_prompt_tokens": self.estimated_prompt_tokens,
            "reasoning": list(self.reasoning),
        }


# ---------------------------------------------------------------------------
# Budget manager
# ---------------------------------------------------------------------------

class TokenBudgetManager:
    """
    Estimates the appropriate ``max_tokens`` for a cloud brain request.

    Usage
    -----
    ::

        manager = TokenBudgetManager()
        estimate = manager.estimate(prompt="Tell me about Elon Musk.", intent="unknown")
        # use estimate.max_tokens in the API call

        # On provider rejection:
        fallback = manager.fallback_budget()
    """

    # Intent labels that map to specific categories regardless of prompt length
    _INTENT_CATEGORY_MAP: Dict[str, ResponseCategory] = {
        "greeting":     ResponseCategory.GREETING,
        "brain_status": ResponseCategory.SHORT_FACTUAL,
        "project_name": ResponseCategory.SHORT_FACTUAL,
        "current_version": ResponseCategory.SHORT_FACTUAL,
        "next_step":    ResponseCategory.SHORT_FACTUAL,
        "recall":       ResponseCategory.SHORT_FACTUAL,
        "show_memory":  ResponseCategory.SHORT_FACTUAL,
        "project_status":       ResponseCategory.NORMAL_CONV,
        "completed_milestones": ResponseCategory.NORMAL_CONV,
        "show_tasks":           ResponseCategory.NORMAL_CONV,
        "show_completed_tasks": ResponseCategory.NORMAL_CONV,
        "cloud_brain":  ResponseCategory.NORMAL_CONV,
        "unknown":      None,   # None = fall through to prompt-length heuristic
    }

    # Keywords that signal a long or complex response is expected
    _REPORT_KEYWORDS = frozenset({
        "report", "summary", "summarize", "summarise", "essay", "document",
        "write a", "generate a", "create a", "detailed", "comprehensive",
        "in depth", "in-depth", "explain everything", "full explanation",
        "all about", "complete guide", "step by step", "step-by-step",
    })
    _REASONING_KEYWORDS = frozenset({
        "why", "how does", "explain", "analyse", "analyze", "compare",
        "difference between", "pros and cons", "advantages", "disadvantages",
        "design", "architecture", "implement", "algorithm",
    })

    def __init__(self, config: Optional[TokenBudgetConfig] = None) -> None:
        self._config = config or TokenBudgetConfig()
        self._config.validate()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(self, prompt: str, intent: Optional[str] = None) -> BudgetEstimate:
        """
        Estimate the token budget for a single cloud brain request.

        :param prompt: The full prompt text that will be sent.
        :param intent: Optional intent label from detect_intent() in Jarwis.py.
        :returns: BudgetEstimate with the recommended max_tokens.
        """
        reasoning: List[str] = []
        text = str(prompt).strip()

        # Step 1: rough token estimate (4 chars ≈ 1 token, common rule of thumb)
        estimated_prompt_tokens = max(1, len(text) // 4)
        reasoning.append(
            f"Prompt length: {len(text)} chars → ~{estimated_prompt_tokens} prompt tokens."
        )

        # Step 2: try intent-based category first
        intent_key = (intent or "").lower().strip()
        category = self._INTENT_CATEGORY_MAP.get(intent_key, None)

        if category is not None:
            reasoning.append(f"Intent '{intent_key}' maps directly to category '{category.value}'.")
        else:
            # Step 3: keyword detection
            text_lower = text.lower()
            if any(kw in text_lower for kw in self._REPORT_KEYWORDS):
                category = ResponseCategory.LARGE_REPORT
                matched = [kw for kw in self._REPORT_KEYWORDS if kw in text_lower]
                reasoning.append(f"Report keywords detected: {matched[:3]} → LARGE_REPORT.")
            elif any(kw in text_lower for kw in self._REASONING_KEYWORDS):
                category = ResponseCategory.REASONING
                matched = [kw for kw in self._REASONING_KEYWORDS if kw in text_lower]
                reasoning.append(f"Reasoning keywords detected: {matched[:3]} → REASONING.")
            else:
                # Step 4: fall back to prompt-length heuristic
                category = self._category_from_prompt_length(estimated_prompt_tokens, reasoning)

        # Step 5: look up budget and clamp to provider_max
        max_tokens = self._config.budget_for(category)
        reasoning.append(
            f"Category '{category.value}' → budget {max_tokens} tokens "
            f"(provider_max={self._config.provider_max})."
        )

        logger.debug(
            "TokenBudget estimated: category=%s max_tokens=%d prompt_tokens=%d",
            category.value, max_tokens, estimated_prompt_tokens,
        )

        return BudgetEstimate(
            category=category,
            max_tokens=max_tokens,
            estimated_prompt_tokens=estimated_prompt_tokens,
            reasoning=reasoning,
        )

    def fallback_budget(self) -> int:
        """
        Return the safe fallback budget to use after a provider rejection.

        Always clamped to provider_max.
        """
        return min(self._config.fallback_budget, self._config.provider_max)

    def provider_max(self) -> int:
        """Return the absolute provider ceiling."""
        return self._config.provider_max

    def safe_default(self) -> int:
        """Return the safe-default budget."""
        return min(self._config.safe_default, self._config.provider_max)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _category_from_prompt_length(
        self,
        estimated_prompt_tokens: int,
        reasoning: List[str],
    ) -> ResponseCategory:
        """
        Choose a category based purely on how large the prompt is.

        Longer prompts usually need longer answers (context-continuation
        pattern); shorter prompts are more likely simple factual queries.
        """
        if estimated_prompt_tokens <= 20:
            reasoning.append(
                f"Short prompt ({estimated_prompt_tokens} tokens) → SHORT_FACTUAL."
            )
            return ResponseCategory.SHORT_FACTUAL

        if estimated_prompt_tokens <= 60:
            reasoning.append(
                f"Medium prompt ({estimated_prompt_tokens} tokens) → NORMAL_CONVERSATION."
            )
            return ResponseCategory.NORMAL_CONV

        if estimated_prompt_tokens <= 200:
            reasoning.append(
                f"Long prompt ({estimated_prompt_tokens} tokens) → REASONING."
            )
            return ResponseCategory.REASONING

        reasoning.append(
            f"Very long prompt ({estimated_prompt_tokens} tokens) → LONG_EXPLANATION."
        )
        return ResponseCategory.LONG_EXPLANATION


# ---------------------------------------------------------------------------
# Module-level convenience: a shared default manager instance
# ---------------------------------------------------------------------------

_default_manager: Optional[TokenBudgetManager] = None


def get_default_manager() -> TokenBudgetManager:
    """Return the module-level default TokenBudgetManager (lazy-initialised)."""
    global _default_manager
    if _default_manager is None:
        _default_manager = TokenBudgetManager()
    return _default_manager


def reset_default_manager(config: Optional[TokenBudgetConfig] = None) -> None:
    """
    Replace the module-level default manager.
    Useful in tests and when reloading configuration.
    """
    global _default_manager
    _default_manager = TokenBudgetManager(config)
