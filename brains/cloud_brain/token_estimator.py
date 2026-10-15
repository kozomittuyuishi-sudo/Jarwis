"""
cloud_brain/token_estimator.py
================================
Token Budget Manager

Estimates an appropriate max_tokens value for every cloud request.
Never uses hardcoded values — all limits are driven by:

    1. Prompt size     (character count → rough token estimate)
    2. Reasoning complexity  (detected from keywords and structure)
    3. Expected response size (inferred from request category + prompt)
    4. Request category  (caller-supplied or auto-detected)
    5. Provider capability  (provider's max_output_tokens is the ceiling)

Design principles
-----------------
- All budget constants live in EstimatorConfig, never in logic.
- Returns the SMALLEST practical budget, not the largest allowed one.
- Provider capability profile supplies the hard ceiling.
- Stdlib only. No I/O. No LLM calls.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request categories
# ---------------------------------------------------------------------------

class RequestCategory(str, Enum):
    """
    Coarse classification of the expected response size.
    The string value is used in logs and serialisation.
    """
    GREETING         = "greeting"           # ≤ 128 output tokens
    SHORT_FACTUAL    = "short_factual"      # ≤ 256 output tokens
    CONVERSATIONAL   = "conversational"     # ≤ 384 output tokens
    REASONING        = "reasoning"          # ≤ 1 024 output tokens
    LONG_EXPLANATION = "long_explanation"  # ≤ 1 536 output tokens
    LARGE_REPORT     = "large_report"       # ≤ 2 048 output tokens
    VERY_LARGE       = "very_large"         # ≤ 2 048 output tokens


# ---------------------------------------------------------------------------
# Estimator configuration — all constants here, never in logic
# ---------------------------------------------------------------------------

@dataclass
class EstimatorConfig:
    """
    All token-budget constants in one place.

    Attributes
    ----------
    category_budgets : Dict[RequestCategory, int]
        Maximum output tokens per category.
    provider_safety_cap : int
        Absolute ceiling when no provider profile is available.
    fallback_budget : int
        Safe retry value after a provider rejection.
    safe_default : int
        Used when the category cannot be determined.
    chars_per_token : float
        Rough chars-per-token ratio for prompt size estimation.
    """
    category_budgets: Dict[RequestCategory, int] = field(
        default_factory=lambda: {
            RequestCategory.GREETING:         128,
            RequestCategory.SHORT_FACTUAL:    256,
            RequestCategory.CONVERSATIONAL:   384,
            RequestCategory.REASONING:       1024,
            RequestCategory.LONG_EXPLANATION:1536,
            RequestCategory.LARGE_REPORT:    2048,
            RequestCategory.VERY_LARGE:      2048,
        }
    )
    provider_safety_cap: int = 2048
    context_safety_reserve: int = 128
    fallback_budget: int     = 512
    safe_default: int        = 512
    chars_per_token: float   = 4.0

    def validate(self) -> None:
        if self.provider_safety_cap <= 0:
            raise ValueError(f"provider_safety_cap must be > 0, got {self.provider_safety_cap}")
        if self.fallback_budget <= 0:
            raise ValueError(f"fallback_budget must be > 0, got {self.fallback_budget}")
        if self.safe_default <= 0:
            raise ValueError(f"safe_default must be > 0, got {self.safe_default}")
        if self.chars_per_token <= 0:
            raise ValueError(f"chars_per_token must be > 0, got {self.chars_per_token}")
        if self.context_safety_reserve < 0:
            raise ValueError(
                "context_safety_reserve must be >= 0, "
                f"got {self.context_safety_reserve}"
            )


# ---------------------------------------------------------------------------
# Estimation result
# ---------------------------------------------------------------------------

@dataclass
class EstimationResult:
    """
    Output of a single estimation call.

    Attributes
    ----------
    category : RequestCategory
        The detected or supplied request category.
    max_tokens : int
        Recommended max_tokens to send to the provider.
    estimated_prompt_tokens : int
        Rough estimate of the prompt's token count.
    complexity_score : float
        0.0–1.0 measure of reasoning complexity detected.
    provider_ceiling : int
        The ceiling actually applied (provider profile or safety cap).
    reasoning : List[str]
        Step-by-step explanation of how the estimate was reached.
    """
    category: RequestCategory
    max_tokens: int
    estimated_prompt_tokens: int
    complexity_score: float
    provider_ceiling: int
    reasoning: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "max_tokens": self.max_tokens,
            "estimated_prompt_tokens": self.estimated_prompt_tokens,
            "complexity_score": self.complexity_score,
            "provider_ceiling": self.provider_ceiling,
            "reasoning": list(self.reasoning),
        }


# ---------------------------------------------------------------------------
# Token Estimator
# ---------------------------------------------------------------------------

class TokenEstimator:
    """
    Estimates the appropriate max_tokens for a cloud brain request.

    Estimation pipeline
    -------------------
    1. Compute estimated prompt token count from character length.
    2. Detect reasoning complexity via structural signals.
    3. Resolve category (caller-supplied → keyword detection → length heuristic).
    4. Look up budget from config and clamp to provider output and remaining
       context-window limits.
    5. Return EstimationResult with full reasoning trace.

    Usage
    -----
    ::

        estimator = TokenEstimator()
        result = estimator.estimate(
            prompt="Explain the CAP theorem in detail.",
            category="reasoning",
            provider_max_output=8192,
        )
        # result.max_tokens is safe to pass directly to the provider
    """

    # -------------------------------------------------------------------
    # Intent label → category (fast-path; intent supplied by caller)
    # -------------------------------------------------------------------
    _INTENT_MAP: Dict[str, RequestCategory] = {
        "greeting":               RequestCategory.GREETING,
        "brain_status":           RequestCategory.SHORT_FACTUAL,
        "project_name":           RequestCategory.SHORT_FACTUAL,
        "current_version":        RequestCategory.SHORT_FACTUAL,
        "next_step":              RequestCategory.SHORT_FACTUAL,
        "recall":                 RequestCategory.SHORT_FACTUAL,
        "show_memory":            RequestCategory.SHORT_FACTUAL,
        "project_status":         RequestCategory.CONVERSATIONAL,
        "completed_milestones":   RequestCategory.CONVERSATIONAL,
        "show_tasks":             RequestCategory.CONVERSATIONAL,
        "show_completed_tasks":   RequestCategory.CONVERSATIONAL,
        "cloud_brain":            RequestCategory.CONVERSATIONAL,
    }

    # -------------------------------------------------------------------
    # Keyword sets for auto-detection (larger responses expected)
    # -------------------------------------------------------------------
    _REPORT_KEYWORDS = frozenset({
        "report", "summary", "summarize", "summarise", "essay", "document",
        "write a", "generate a", "create a", "detailed", "comprehensive",
        "in depth", "in-depth", "explain everything", "full explanation",
        "all about", "complete guide", "step by step", "step-by-step",
        "write me", "draft", "outline",
    })
    _REASONING_KEYWORDS = frozenset({
        "why", "how does", "explain", "analyse", "analyze", "compare",
        "difference between", "pros and cons", "advantages", "disadvantages",
        "design", "architecture", "implement", "algorithm", "reason",
        "brainstorm",
        "evaluate", "assess", "critique", "review", "trace", "debug",
        "refactor", "optimize", "optimise",
    })

    # -------------------------------------------------------------------
    # Complexity signal patterns
    # -------------------------------------------------------------------
    _MULTI_STEP_RE = re.compile(
        r"\b(first|then|next|finally|step \d|additionally|furthermore|"
        r"after that|following that|subsequently)\b",
        re.IGNORECASE,
    )
    _QUESTION_CLUSTER_RE = re.compile(r"\?")

    def __init__(self, config: Optional[EstimatorConfig] = None) -> None:
        self._config = config or EstimatorConfig()
        self._config.validate()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(
        self,
        prompt: str,
        *,
        category: Optional[str] = None,
        intent: Optional[str] = None,
        provider_max_output: Optional[int] = None,
        provider_context_window: Optional[int] = None,
        task_text: Optional[str] = None,
    ) -> EstimationResult:
        """
        Estimate the max_tokens budget for a cloud request.

        Parameters
        ----------
        prompt : str
            The full prompt text that will be sent to the provider.
        category : str, optional
            Caller-supplied RequestCategory value (e.g. "reasoning").
            Overrides auto-detection when provided and valid.
        intent : str, optional
            Intent label (e.g. from detect_intent()).  Used as a fast-path
            category mapping before keyword analysis.
        provider_max_output : int, optional
            The provider's max_output_tokens.  Acts as the hard ceiling.
            Falls back to config.provider_safety_cap when not supplied.
        provider_context_window : int, optional
            Total prompt-plus-output context window reported by the provider.
        task_text : str, optional
            User request text for category detection, kept separate from
            system instructions and conversation context in ``prompt``.

        Returns
        -------
        EstimationResult
        """
        reasoning: List[str] = []
        text = str(prompt).strip()
        classification_text = str(task_text).strip() if task_text is not None else text
        classification_tokens = max(
            1,
            int(len(classification_text) / self._config.chars_per_token),
        )

        # Step 1 — Prompt token estimate
        estimated_prompt_tokens = max(1, int(len(text) / self._config.chars_per_token))
        reasoning.append(
            f"Prompt: {len(text)} chars → ~{estimated_prompt_tokens} prompt tokens "
            f"(ratio {self._config.chars_per_token} chars/token)."
        )

        # Step 2 — Complexity score
        complexity = self._compute_complexity(classification_text, reasoning)

        # Step 3 — Resolve category
        resolved_category = self._resolve_category(
            text=classification_text,
            estimated_prompt_tokens=classification_tokens,
            category=category,
            intent=intent,
            complexity=complexity,
            reasoning=reasoning,
        )

        # Step 4 — Look up budget
        raw_budget = self._config.category_budgets.get(
            resolved_category, self._config.safe_default
        )

        # Step 5 — Apply complexity uplift (up to 50% extra for high complexity)
        if complexity > 0.5:
            uplift = 1.0 + (complexity - 0.5)  # 0.5 complexity → 1.0x, 1.0 → 1.5x
            uplifted = int(raw_budget * uplift)
            reasoning.append(
                f"Complexity uplift: {complexity:.2f} → ×{uplift:.2f} → {uplifted} tokens."
            )
            raw_budget = uplifted

        # Output allowance is distinct from estimated input size. Reserve
        # context space for both the complete prompt and provider output.
        output_ceiling = (
            provider_max_output
            if provider_max_output and provider_max_output > 0
            else self._config.provider_safety_cap
        )
        ceiling = min(output_ceiling, self._config.provider_safety_cap)
        if provider_context_window and provider_context_window > 0:
            context_headroom = max(
                1,
                provider_context_window
                - estimated_prompt_tokens
                - self._config.context_safety_reserve,
            )
            ceiling = min(ceiling, context_headroom)
            reasoning.append(
                f"Context window {provider_context_window} minus estimated prompt "
                f"{estimated_prompt_tokens} and reserve "
                f"{self._config.context_safety_reserve} leaves {context_headroom} "
                "output tokens."
            )
        max_tokens = min(raw_budget, ceiling)
        reasoning.append(
            f"Category '{resolved_category.value}' → budget {raw_budget} tokens; "
            f"ceiling {ceiling} → final max_tokens={max_tokens}."
        )

        logger.debug(
            "TokenEstimator: category=%s complexity=%.2f max_tokens=%d prompt_tokens=%d",
            resolved_category.value, complexity, max_tokens, estimated_prompt_tokens,
        )

        return EstimationResult(
            category=resolved_category,
            max_tokens=max_tokens,
            estimated_prompt_tokens=estimated_prompt_tokens,
            complexity_score=complexity,
            provider_ceiling=ceiling,
            reasoning=reasoning,
        )

    def fallback_budget(
        self, provider_max_output: Optional[int] = None
    ) -> int:
        """Return the safe fallback budget, clamped to the provider ceiling."""
        ceiling = provider_max_output or self._config.provider_safety_cap
        return min(self._config.fallback_budget, ceiling)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_complexity(self, text: str, reasoning: List[str]) -> float:
        """
        Score reasoning complexity on a 0.0–1.0 scale.

        Signals
        -------
        - Presence of multi-step connectives (first/then/next/…)
        - Multiple question marks in one prompt
        - Prompt length above 200 tokens (long context implies complex task)
        - Reasoning/analysis keywords
        """
        score = 0.0
        lower = text.lower()

        multi_step_hits = len(self._MULTI_STEP_RE.findall(text))
        if multi_step_hits >= 2:
            score += min(0.3, multi_step_hits * 0.1)
            reasoning.append(f"Multi-step connectives detected ({multi_step_hits}) → +{min(0.3, multi_step_hits*0.1):.2f} complexity.")

        question_count = len(self._QUESTION_CLUSTER_RE.findall(text))
        if question_count >= 2:
            score += min(0.2, question_count * 0.05)
            reasoning.append(f"Multiple questions ({question_count}) → +{min(0.2, question_count*0.05):.2f} complexity.")

        reasoning_hits = sum(1 for kw in self._REASONING_KEYWORDS if kw in lower)
        if reasoning_hits:
            score += min(0.3, reasoning_hits * 0.1)
            reasoning.append(f"Reasoning keywords ({reasoning_hits}) → +{min(0.3, reasoning_hits*0.1):.2f} complexity.")

        est_tokens = max(1, int(len(text) / self._config.chars_per_token))
        if est_tokens > 200:
            score += 0.2
            reasoning.append(f"Long prompt (~{est_tokens} tokens) → +0.20 complexity.")

        complexity = min(1.0, score)
        reasoning.append(f"Complexity score: {complexity:.2f}.")
        return complexity

    def _resolve_category(
        self,
        text: str,
        estimated_prompt_tokens: int,
        category: Optional[str],
        intent: Optional[str],
        complexity: float,
        reasoning: List[str],
    ) -> RequestCategory:
        """
        Resolve the request category using a priority chain:

        1. Explicit caller-supplied category (if valid)
        2. Intent label mapping
        3. Report keyword detection
        4. Reasoning keyword detection
        5. High-complexity upgrade
        6. Prompt length heuristic
        """
        # Priority 1: explicit caller category
        if category:
            for rc in RequestCategory:
                if rc.value == category.lower().strip():
                    reasoning.append(f"Caller-supplied category '{rc.value}' accepted.")
                    return rc
            reasoning.append(
                f"Caller-supplied category '{category}' not recognised; falling through."
            )

        # Priority 2: intent label
        intent_key = (intent or "").lower().strip()
        if intent_key in self._INTENT_MAP:
            mapped = self._INTENT_MAP[intent_key]
            reasoning.append(f"Intent '{intent_key}' → category '{mapped.value}'.")
            return mapped

        lower = text.lower()

        # Priority 3: report keywords
        if any(kw in lower for kw in self._REPORT_KEYWORDS):
            matched = [kw for kw in self._REPORT_KEYWORDS if kw in lower][:3]
            reasoning.append(f"Report keywords {matched} → LARGE_REPORT.")
            return RequestCategory.LARGE_REPORT

        # Priority 4: reasoning keywords
        if any(kw in lower for kw in self._REASONING_KEYWORDS):
            matched = [kw for kw in self._REASONING_KEYWORDS if kw in lower][:3]
            reasoning.append(f"Reasoning keywords {matched} → REASONING.")
            return RequestCategory.REASONING

        # Priority 5: high complexity upgrade
        if complexity >= 0.6:
            reasoning.append(f"Complexity {complexity:.2f} ≥ 0.60 → REASONING.")
            return RequestCategory.REASONING

        # Priority 6: prompt length heuristic
        return self._category_from_length(estimated_prompt_tokens, reasoning)

    def _category_from_length(
        self, estimated_prompt_tokens: int, reasoning: List[str]
    ) -> RequestCategory:
        if estimated_prompt_tokens <= 20:
            reasoning.append(f"Short prompt (~{estimated_prompt_tokens} tokens) → SHORT_FACTUAL.")
            return RequestCategory.SHORT_FACTUAL
        if estimated_prompt_tokens <= 60:
            reasoning.append(f"Medium prompt (~{estimated_prompt_tokens} tokens) → CONVERSATIONAL.")
            return RequestCategory.CONVERSATIONAL
        if estimated_prompt_tokens <= 200:
            reasoning.append(f"Long prompt (~{estimated_prompt_tokens} tokens) → REASONING.")
            return RequestCategory.REASONING
        reasoning.append(f"Very long prompt (~{estimated_prompt_tokens} tokens) → LONG_EXPLANATION.")
        return RequestCategory.LONG_EXPLANATION
