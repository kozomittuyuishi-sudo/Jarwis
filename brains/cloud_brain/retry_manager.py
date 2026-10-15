"""
cloud_brain/retry_manager.py
==============================
Retry Manager

When request or response validation fails, the Retry Manager decides
whether a retry is warranted and prepares the adjusted configuration
for the next attempt.

Retry strategy
--------------
On a failed validation attempt the manager:

    1. Checks whether we have reached the retry cap (never infinite).
    2. Classifies the failure to determine the appropriate adjustment:
       - Token/budget failures → reduce budget to fallback value.
       - Provider/model failures → swap to fallback provider config if available.
       - Response failures → reduce budget and flag as soft retry.
    3. Returns a RetryOutcome describing whether to retry and with what config.

Design principles
-----------------
- Stdlib only. No I/O. No LLM calls.
- Maximum retries enforced via RetryConfig.max_attempts.
- The manager never calls the provider itself — it only decides whether
  and how to retry; the caller performs the actual dispatch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from brains.cloud_brain.request_validator import ValidationReport, Severity
from brains.cloud_brain.response_validator import ValidationResult, ValidationAxis

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Retry status
# ---------------------------------------------------------------------------

class RetryStatus(str, Enum):
    RETRY          = "retry"           # Retry with adjusted config
    GIVE_UP        = "give_up"         # Max attempts reached
    NO_RETRY       = "no_retry"        # Failure is not retryable
    SUCCESS        = "success"         # Nothing to retry — previous call passed


# ---------------------------------------------------------------------------
# Retry outcome
# ---------------------------------------------------------------------------

@dataclass
class RetryOutcome:
    """
    Decision output from the Retry Manager.

    Attributes
    ----------
    status : RetryStatus
        Whether and how to retry.
    attempt_number : int
        The attempt that just completed (1-based).
    max_attempts : int
        The configured ceiling.
    adjusted_max_tokens : Optional[int]
        Reduced token budget to use on the next attempt.
    adjusted_provider_id : Optional[str]
        Fallback provider to use on the next attempt (if provider swap needed).
    adjusted_model_id : Optional[str]
        Fallback model to use on the next attempt.
    reason : str
        Human-readable explanation of the decision.
    adjustments : List[str]
        List of specific changes made for the next attempt.
    """
    status: RetryStatus
    attempt_number: int
    max_attempts: int
    adjusted_max_tokens: Optional[int] = None
    adjusted_provider_id: Optional[str] = None
    adjusted_model_id: Optional[str] = None
    reason: str = ""
    adjustments: List[str] = field(default_factory=list)

    @property
    def should_retry(self) -> bool:
        return self.status == RetryStatus.RETRY

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "attempt_number": self.attempt_number,
            "max_attempts": self.max_attempts,
            "adjusted_max_tokens": self.adjusted_max_tokens,
            "adjusted_provider_id": self.adjusted_provider_id,
            "adjusted_model_id": self.adjusted_model_id,
            "reason": self.reason,
            "adjustments": list(self.adjustments),
        }


# ---------------------------------------------------------------------------
# Retry configuration
# ---------------------------------------------------------------------------

@dataclass
class RetryConfig:
    """
    All retry policy constants.

    Attributes
    ----------
    max_attempts : int
        Total number of attempts allowed (first attempt + retries).
        Must be >= 1.  Set to 1 to disable retries.
    token_reduction_factor : float
        Multiply current max_tokens by this factor on each budget retry.
        E.g. 0.5 halves the budget each time.
    min_tokens : int
        Floor for the adjusted token budget; never go below this.
    fallback_provider_id : Optional[str]
        Provider to swap to when the primary provider itself is the problem.
    fallback_model_id : Optional[str]
        Model to swap to when the primary model is unavailable.
    """
    max_attempts: int            = 3
    token_reduction_factor: float = 0.5
    min_tokens: int               = 128
    fallback_provider_id: Optional[str] = None
    fallback_model_id: Optional[str]    = None

    def validate(self) -> None:
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {self.max_attempts}")
        if not (0.0 < self.token_reduction_factor < 1.0):
            raise ValueError(
                f"token_reduction_factor must be in (0, 1), "
                f"got {self.token_reduction_factor}"
            )
        if self.min_tokens < 1:
            raise ValueError(f"min_tokens must be >= 1, got {self.min_tokens}")


# ---------------------------------------------------------------------------
# Retry Manager
# ---------------------------------------------------------------------------

class RetryManager:
    """
    Decides whether to retry a failed dispatch and how to adjust the config.

    Usage
    -----
    ::

        manager = RetryManager(RetryConfig(max_attempts=3))

        # After a failed RequestValidator report:
        outcome = manager.evaluate_request_failure(
            report=report,
            current_max_tokens=512,
            attempt_number=1,
        )

        # After a failed ResponseValidator result:
        outcome = manager.evaluate_response_failure(
            result=result,
            current_max_tokens=1024,
            attempt_number=1,
        )

        if outcome.should_retry:
            dispatch(max_tokens=outcome.adjusted_max_tokens, ...)
    """

    # Failure check names that indicate budget/token problems
    _TOKEN_CHECKS = frozenset({"token_budget", "request_size"})

    # Failure check names that indicate provider/model problems
    _PROVIDER_CHECKS = frozenset({"provider_known", "provider_compatibility"})
    _MODEL_CHECKS    = frozenset({"model_availability"})

    # Response axes that are retryable via budget reduction
    _BUDGET_AXES = frozenset({
        ValidationAxis.EMPTY_RESPONSE,
        ValidationAxis.MESSAGE_CONTENT,
    })

    def __init__(self, config: Optional[RetryConfig] = None) -> None:
        self._config = config or RetryConfig()
        self._config.validate()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_request_failure(
        self,
        report: ValidationReport,
        current_max_tokens: int,
        attempt_number: int,
    ) -> RetryOutcome:
        """
        Decide whether to retry after a RequestValidator failure.

        Parameters
        ----------
        report : ValidationReport
            The failed validation report.
        current_max_tokens : int
            The max_tokens that was used in the failed attempt.
        attempt_number : int
            The attempt number that just failed (1-based).

        Returns
        -------
        RetryOutcome
        """
        if report.passed:
            return RetryOutcome(
                status=RetryStatus.SUCCESS,
                attempt_number=attempt_number,
                max_attempts=self._config.max_attempts,
                reason="Request validation passed — no retry needed.",
            )

        if attempt_number >= self._config.max_attempts:
            return RetryOutcome(
                status=RetryStatus.GIVE_UP,
                attempt_number=attempt_number,
                max_attempts=self._config.max_attempts,
                reason=(
                    f"Max attempts ({self._config.max_attempts}) reached after "
                    f"request validation failure."
                ),
            )

        error_checks = {i.check for i in report.errors}
        adjustments: List[str] = []
        adjusted_tokens = current_max_tokens
        adjusted_provider = None
        adjusted_model    = None

        # Token/budget failure → reduce budget
        if error_checks & self._TOKEN_CHECKS:
            adjusted_tokens = self._reduce_tokens(current_max_tokens)
            adjustments.append(
                f"Reduced max_tokens: {current_max_tokens} → {adjusted_tokens}."
            )

        # Provider failure → swap provider
        if error_checks & self._PROVIDER_CHECKS:
            if self._config.fallback_provider_id:
                adjusted_provider = self._config.fallback_provider_id
                adjustments.append(
                    f"Switched provider to fallback: {adjusted_provider}."
                )
            else:
                return RetryOutcome(
                    status=RetryStatus.NO_RETRY,
                    attempt_number=attempt_number,
                    max_attempts=self._config.max_attempts,
                    reason=(
                        "Provider compatibility failure and no fallback provider configured."
                    ),
                )

        # Model failure → swap model
        if error_checks & self._MODEL_CHECKS:
            if self._config.fallback_model_id:
                adjusted_model = self._config.fallback_model_id
                adjustments.append(
                    f"Switched model to fallback: {adjusted_model}."
                )
            else:
                # Model not in known list is WARNING by default; don't block retry
                adjustments.append(
                    "Model not in known list — proceeding with same model."
                )

        if not adjustments:
            return RetryOutcome(
                status=RetryStatus.NO_RETRY,
                attempt_number=attempt_number,
                max_attempts=self._config.max_attempts,
                reason=f"Request failed on checks {error_checks} — not retryable.",
            )

        logger.info(
            "RetryManager: attempt=%d/%d → RETRY. adjustments=%s",
            attempt_number, self._config.max_attempts, adjustments,
        )

        return RetryOutcome(
            status=RetryStatus.RETRY,
            attempt_number=attempt_number,
            max_attempts=self._config.max_attempts,
            adjusted_max_tokens=adjusted_tokens,
            adjusted_provider_id=adjusted_provider,
            adjusted_model_id=adjusted_model,
            reason=f"Request validation failed on: {sorted(error_checks)}.",
            adjustments=adjustments,
        )

    def evaluate_response_failure(
        self,
        result: ValidationResult,
        current_max_tokens: int,
        attempt_number: int,
    ) -> RetryOutcome:
        """
        Decide whether to retry after a ResponseValidator failure.

        Parameters
        ----------
        result : ValidationResult
            The failed validation result.
        current_max_tokens : int
            The max_tokens that was used in the failed attempt.
        attempt_number : int
            The attempt number that just failed (1-based).

        Returns
        -------
        RetryOutcome
        """
        if result.passed:
            return RetryOutcome(
                status=RetryStatus.SUCCESS,
                attempt_number=attempt_number,
                max_attempts=self._config.max_attempts,
                reason="Response validation passed — no retry needed.",
            )

        if attempt_number >= self._config.max_attempts:
            return RetryOutcome(
                status=RetryStatus.GIVE_UP,
                attempt_number=attempt_number,
                max_attempts=self._config.max_attempts,
                reason=(
                    f"Max attempts ({self._config.max_attempts}) reached after "
                    f"response validation failure."
                ),
            )

        failed_axes = {r.axis for r in result.failures}
        adjustments: List[str] = []
        adjusted_tokens = current_max_tokens

        # Malformed / structural failures are rarely fixable by token reduction
        if ValidationAxis.MALFORMED in failed_axes:
            return RetryOutcome(
                status=RetryStatus.NO_RETRY,
                attempt_number=attempt_number,
                max_attempts=self._config.max_attempts,
                reason="Malformed response — structural failure; retry unlikely to help.",
            )

        # Empty response or invalid content can receive a bounded retry.
        if failed_axes & self._BUDGET_AXES:
            # This is not a billing recovery path; HTTP 402 is handled before
            # response validation and must never be retried here.
            adjusted_tokens = self._reduce_tokens(current_max_tokens)
            adjustments.append(
                f"Adjusted max_tokens for retry: {current_max_tokens} → {adjusted_tokens}."
            )

        if not adjustments:
            adjustments.append("Retrying with same token budget.")
            adjusted_tokens = current_max_tokens

        logger.info(
            "RetryManager: response attempt=%d/%d → RETRY. adjustments=%s",
            attempt_number, self._config.max_attempts, adjustments,
        )

        return RetryOutcome(
            status=RetryStatus.RETRY,
            attempt_number=attempt_number,
            max_attempts=self._config.max_attempts,
            adjusted_max_tokens=adjusted_tokens,
            reason=f"Response validation failed on axes: {[a.value for a in sorted(failed_axes)]}.",
            adjustments=adjustments,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reduce_tokens(self, current: int) -> int:
        """Apply token_reduction_factor, clamped to min_tokens."""
        reduced = max(
            self._config.min_tokens,
            int(current * self._config.token_reduction_factor),
        )
        return reduced
