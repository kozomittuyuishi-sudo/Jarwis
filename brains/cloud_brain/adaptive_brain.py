"""
cloud_brain/adaptive_brain.py
================================
Adaptive Cloud Brain — Orchestrator

The AdaptiveCloudBrain is the top-level entry point for Part A.
It ties together all five subsystem components into one coherent
middleware pipeline:

    1. TokenEstimator         — estimate max_tokens
    2. RequestValidator       — validate before dispatch
    3. [Caller dispatches]    — the orchestrator does NOT call the provider
    4. ResponseValidator      — validate after dispatch
    5. RetryManager           — decide whether/how to retry
    6. RequestLogger          — record every attempt

The orchestrator does NOT make HTTP calls.
Its job is to prepare, validate, and wrap a request so that the
caller always receives a structured BrainResult regardless of
provider behaviour.

Two usage patterns are supported:

    Pattern A — Full pipeline (prepare + validate both sides):
    ::

        brain  = AdaptiveCloudBrain(...)
        result = brain.prepare_and_validate(
            provider="openrouter",
            model="deepseek/deepseek-v4-flash-0731",
            prompt="Explain quantum computing.",
            raw_response=actual_response_from_provider,
        )

    Pattern B — Prepare only (caller validates the response separately):
    ::

        prep  = brain.prepare(provider, model, prompt)
        # caller dispatches using prep.max_tokens
        result = brain.validate_response(raw_response, prep)

Design principles
-----------------
- Stdlib only. No I/O. No LLM calls.
- Never raises on provider failures — encapsulates into BrainResult.
- All component defaults can be overridden via constructor injection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from brains.cloud_brain.provider_registry import ProviderRegistry
from brains.cloud_brain.token_estimator import TokenEstimator, EstimationResult
from brains.cloud_brain.request_validator import RequestValidator, ValidationReport
from brains.cloud_brain.response_validator import ResponseValidator, ValidationResult
from brains.cloud_brain.retry_manager import RetryManager, RetryOutcome, RetryStatus
from brains.cloud_brain.request_logger import RequestLogger, RequestRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Brain status
# ---------------------------------------------------------------------------

class BrainStatus(str, Enum):
    READY            = "ready"           # Prepared and validated; ready to dispatch
    PASSED           = "passed"          # Full pipeline: both sides passed
    REQUEST_FAILED   = "request_failed"  # Pre-dispatch validation failed
    RESPONSE_FAILED  = "response_failed" # Post-dispatch validation failed
    RETRYING         = "retrying"        # Retry recommended
    GAVE_UP          = "gave_up"         # Max retries reached


# ---------------------------------------------------------------------------
# Preparation result (output of the pre-dispatch phase)
# ---------------------------------------------------------------------------

@dataclass
class PrepResult:
    """
    The output of the pre-dispatch preparation phase.

    Attributes
    ----------
    provider_id : str
    model_id : str
    max_tokens : int
        Token budget to use in the actual provider call.
    estimation : EstimationResult
        Full estimation detail.
    validation_report : ValidationReport
        Pre-dispatch validation result.
    ready : bool
        True when the request passed validation and is safe to dispatch.
    """
    provider_id: str
    model_id: str
    max_tokens: int
    estimation: EstimationResult
    validation_report: ValidationReport
    ready: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "max_tokens": self.max_tokens,
            "ready": self.ready,
            "estimation": self.estimation.to_dict(),
            "validation_report": self.validation_report.to_dict(),
        }


# ---------------------------------------------------------------------------
# Brain result (final output of the full pipeline)
# ---------------------------------------------------------------------------

@dataclass
class BrainResult:
    """
    Structured output of the AdaptiveCloudBrain pipeline.

    Attributes
    ----------
    status : BrainStatus
        Overall pipeline outcome.
    provider_id : str
    model_id : str
    max_tokens_used : int
        Token budget that was actually used in the final attempt.
    estimation : EstimationResult
        Token estimate from the last preparation.
    request_report : ValidationReport
        Pre-dispatch validation report from the last attempt.
    response_result : Optional[ValidationResult]
        Post-dispatch validation result (None if response was not validated).
    retry_outcome : Optional[RetryOutcome]
        Retry decision if a retry was evaluated.
    log_record : Optional[RequestRecord]
        The log record produced for this attempt.
    attempt_number : int
        Which attempt produced this result (1-based).
    """
    status: BrainStatus
    provider_id: str
    model_id: str
    max_tokens_used: int
    estimation: EstimationResult
    request_report: ValidationReport
    response_result: Optional[ValidationResult] = None
    retry_outcome: Optional[RetryOutcome]       = None
    log_record: Optional[RequestRecord]         = None
    attempt_number: int                          = 1

    @property
    def passed(self) -> bool:
        return self.status == BrainStatus.PASSED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "max_tokens_used": self.max_tokens_used,
            "attempt_number": self.attempt_number,
            "passed": self.passed,
            "estimation": self.estimation.to_dict(),
            "request_report": self.request_report.to_dict(),
            "response_result": (
                self.response_result.to_dict()
                if self.response_result else None
            ),
            "retry_outcome": (
                self.retry_outcome.to_dict()
                if self.retry_outcome else None
            ),
            "log_record_id": (
                self.log_record.record_id if self.log_record else None
            ),
        }


# ---------------------------------------------------------------------------
# Adaptive Cloud Brain — Orchestrator
# ---------------------------------------------------------------------------

class AdaptiveCloudBrain:
    """
    Intelligent middleware that prepares, validates, and logs every
    cloud request before and after provider dispatch.

    Parameters
    ----------
    provider_registry : ProviderRegistry, optional
        Custom provider registry.  Defaults to a registry with built-in profiles.
    token_estimator : TokenEstimator, optional
        Custom token estimator.
    request_validator : RequestValidator, optional
        Custom request validator.
    response_validator : ResponseValidator, optional
        Custom response validator.
    retry_manager : RetryManager, optional
        Custom retry manager.
    request_logger : RequestLogger, optional
        Custom request logger.
    """

    def __init__(
        self,
        provider_registry: Optional[ProviderRegistry]   = None,
        token_estimator:   Optional[TokenEstimator]     = None,
        request_validator: Optional[RequestValidator]   = None,
        response_validator:Optional[ResponseValidator]  = None,
        retry_manager:     Optional[RetryManager]       = None,
        request_logger:    Optional[RequestLogger]      = None,
    ) -> None:
        self._registry  = provider_registry  if provider_registry  is not None else ProviderRegistry()
        self._estimator = token_estimator    if token_estimator    is not None else TokenEstimator()
        self._req_val   = request_validator  if request_validator  is not None else RequestValidator(self._registry)
        self._res_val   = response_validator if response_validator is not None else ResponseValidator()
        self._retry     = retry_manager      if retry_manager      is not None else RetryManager()
        self._log       = request_logger     if request_logger     is not None else RequestLogger()

    # ------------------------------------------------------------------
    # Pattern A — Full pipeline
    # ------------------------------------------------------------------

    def prepare_and_validate(
        self,
        provider: str,
        model: str,
        prompt: str,
        raw_response: Optional[Any] = None,
        *,
        category: Optional[str] = None,
        intent: Optional[str] = None,
        required_features: Optional[Set[str]] = None,
        task_text: Optional[str] = None,
        prepared_request: Optional[PrepResult] = None,
        attempt_number: int = 1,
        latency_ms: Optional[float] = None,
    ) -> BrainResult:
        """
        Run the full pre- and post-dispatch pipeline.

        Parameters
        ----------
        provider : str
            Provider ID.
        model : str
            Model ID.
        prompt : str
            The full prompt text.
        raw_response : Any, optional
            The raw response dict returned by the provider.
            When None, only the pre-dispatch phase is run.
        category : str, optional
            Override for RequestCategory auto-detection.
        intent : str, optional
            Intent label for token estimation.
        task_text : str, optional
            User request text used for category/complexity classification when
            prompt also contains system instructions or conversation context.
        prepared_request : PrepResult, optional
            Reuse the exact prepared request when validating its response.
        required_features : set of str, optional
            Features the request requires (checked against provider profile).
        attempt_number : int
            Which attempt this is (1-based).
        latency_ms : float, optional
            Round-trip latency if the caller measured it.

        Returns
        -------
        BrainResult
        """
        # --- Phase 1: Prepare (estimate + pre-validate) ---
        prep = prepared_request or self.prepare(
            provider=provider,
            model=model,
            prompt=prompt,
            category=category,
            intent=intent,
            required_features=required_features,
            task_text=task_text,
        )
        if prep.provider_id != provider or prep.model_id != model:
            raise ValueError(
                "Prepared request provider/model do not match the response "
                "being validated."
            )

        if not prep.ready:
            retry_outcome = self._retry.evaluate_request_failure(
                report=prep.validation_report,
                current_max_tokens=prep.max_tokens,
                attempt_number=attempt_number,
            )
            issues = self._collect_request_issues(prep.validation_report)
            log_rec = self._log.record(
                provider=provider,
                model=model,
                token_estimate=prep.estimation.max_tokens,
                actual_tokens=prep.max_tokens,
                attempt_number=attempt_number,
                request_passed=False,
                response_passed=False,
                retry_status=retry_outcome.status.value,
                validation_issues=issues,
                latency_ms=latency_ms,
            )
            status = (
                BrainStatus.RETRYING
                if retry_outcome.should_retry
                else (
                    BrainStatus.GAVE_UP
                    if retry_outcome.status == RetryStatus.GIVE_UP
                    else BrainStatus.REQUEST_FAILED
                )
            )
            return BrainResult(
                status=status,
                provider_id=provider,
                model_id=model,
                max_tokens_used=prep.max_tokens,
                estimation=prep.estimation,
                request_report=prep.validation_report,
                retry_outcome=retry_outcome,
                log_record=log_rec,
                attempt_number=attempt_number,
            )

        # --- Phase 2: No raw_response yet — return READY ---
        if raw_response is None:
            log_rec = self._log.record(
                provider=provider,
                model=model,
                token_estimate=prep.estimation.max_tokens,
                actual_tokens=prep.max_tokens,
                attempt_number=attempt_number,
                request_passed=True,
                response_passed=False,
                retry_status="n/a",
                latency_ms=latency_ms,
            )
            return BrainResult(
                status=BrainStatus.READY,
                provider_id=provider,
                model_id=model,
                max_tokens_used=prep.max_tokens,
                estimation=prep.estimation,
                request_report=prep.validation_report,
                log_record=log_rec,
                attempt_number=attempt_number,
            )

        # --- Phase 3: Validate response ---
        res_result = self._res_val.validate(raw_response)

        if res_result.passed:
            log_rec = self._log.record(
                provider=provider,
                model=model,
                token_estimate=prep.estimation.max_tokens,
                actual_tokens=prep.max_tokens,
                attempt_number=attempt_number,
                request_passed=True,
                response_passed=True,
                retry_status="success",
                latency_ms=latency_ms,
            )
            return BrainResult(
                status=BrainStatus.PASSED,
                provider_id=provider,
                model_id=model,
                max_tokens_used=prep.max_tokens,
                estimation=prep.estimation,
                request_report=prep.validation_report,
                response_result=res_result,
                log_record=log_rec,
                attempt_number=attempt_number,
            )

        # Response failed
        retry_outcome = self._retry.evaluate_response_failure(
            result=res_result,
            current_max_tokens=prep.max_tokens,
            attempt_number=attempt_number,
        )
        issues = self._collect_response_issues(res_result)
        log_rec = self._log.record(
            provider=provider,
            model=model,
            token_estimate=prep.estimation.max_tokens,
            actual_tokens=prep.max_tokens,
            attempt_number=attempt_number,
            request_passed=True,
            response_passed=False,
            retry_status=retry_outcome.status.value,
            validation_issues=issues,
            latency_ms=latency_ms,
        )
        status = (
            BrainStatus.RETRYING
            if retry_outcome.should_retry
            else (
                BrainStatus.GAVE_UP
                if retry_outcome.status == RetryStatus.GIVE_UP
                else BrainStatus.RESPONSE_FAILED
            )
        )
        return BrainResult(
            status=status,
            provider_id=provider,
            model_id=model,
            max_tokens_used=prep.max_tokens,
            estimation=prep.estimation,
            request_report=prep.validation_report,
            response_result=res_result,
            retry_outcome=retry_outcome,
            log_record=log_rec,
            attempt_number=attempt_number,
        )

    # ------------------------------------------------------------------
    # Pattern B — Prepare only
    # ------------------------------------------------------------------

    def prepare(
        self,
        provider: str,
        model: str,
        prompt: str,
        *,
        category: Optional[str] = None,
        intent: Optional[str] = None,
        required_features: Optional[Set[str]] = None,
        task_text: Optional[str] = None,
    ) -> PrepResult:
        """
        Run only the pre-dispatch phase (estimate + validate).

        Returns a PrepResult with the recommended max_tokens and a flag
        indicating whether the request is safe to dispatch.
        """
        profile = self._registry.get(provider)
        provider_max = profile.max_output_tokens if profile else None
        provider_context = profile.context_window if profile else None

        estimation = self._estimator.estimate(
            prompt=prompt,
            category=category,
            intent=intent,
            provider_max_output=provider_max,
            provider_context_window=provider_context,
            task_text=task_text,
        )

        report = self._req_val.validate(
            provider_id=provider,
            model_id=model,
            prompt=prompt,
            estimation=estimation,
            required_features=required_features or set(),
        )

        return PrepResult(
            provider_id=provider,
            model_id=model,
            max_tokens=estimation.max_tokens,
            estimation=estimation,
            validation_report=report,
            ready=report.passed,
        )

    def validate_response(
        self,
        raw_response: Any,
        prep: PrepResult,
        *,
        attempt_number: int = 1,
        latency_ms: Optional[float] = None,
    ) -> BrainResult:
        """
        Run only the post-dispatch phase (response validation + retry decision).

        Intended for use after Pattern B prepare() + caller dispatch.
        """
        return self.prepare_and_validate(
            provider=prep.provider_id,
            model=prep.model_id,
            prompt="",          # prompt already used in prepare()
            raw_response=raw_response,
            prepared_request=prep,
            attempt_number=attempt_number,
            latency_ms=latency_ms,
        )

    # ------------------------------------------------------------------
    # Component accessors (useful for testing and introspection)
    # ------------------------------------------------------------------

    @property
    def provider_registry(self) -> ProviderRegistry:
        return self._registry

    @property
    def token_estimator(self) -> TokenEstimator:
        return self._estimator

    @property
    def request_validator(self) -> RequestValidator:
        return self._req_val

    @property
    def response_validator(self) -> ResponseValidator:
        return self._res_val

    @property
    def retry_manager(self) -> RetryManager:
        return self._retry

    @property
    def request_logger(self) -> RequestLogger:
        return self._log

    def logger_stats(self) -> Dict[str, Any]:
        """Return summary statistics from the request logger."""
        return self._log.summary_stats()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_request_issues(self, report: ValidationReport) -> List[str]:
        return [
            f"{i.check}[{i.severity.value}]: {i.message}"
            for i in report.issues
        ]

    def _collect_response_issues(self, result: ValidationResult) -> List[str]:
        return [
            f"{r.axis.value}: {r.message}"
            for r in result.failures
        ]
