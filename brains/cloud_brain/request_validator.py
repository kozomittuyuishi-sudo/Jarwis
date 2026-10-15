"""
cloud_brain/request_validator.py
=================================
Request Validator

Verifies a cloud request before it is dispatched to any provider.
Four checks are performed in order:

    1. Request size      — prompt does not exceed the provider's context window
    2. Provider compat   — provider supports the features the request requires
    3. Model availability — model is known for this provider
    4. Token budget      — estimated max_tokens is within the provider's ceiling

Returns a structured ValidationReport with one ValidationIssue per
failing check.  A report with no issues is a pass.

Design principles
-----------------
- Stdlib only. No I/O. No LLM calls.
- All severity thresholds are in RequestValidatorConfig, not in logic.
- Does NOT raise on failure — the caller decides what to do.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from brains.cloud_brain.provider_registry import ProviderFeature, ProviderProfile, ProviderRegistry
from brains.cloud_brain.token_estimator import EstimationResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Severity levels
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    INFO    = "info"
    WARNING = "warning"
    ERROR   = "error"


# ---------------------------------------------------------------------------
# Individual issue
# ---------------------------------------------------------------------------

@dataclass
class ValidationIssue:
    """One problem found during request validation."""
    check: str            # Which check raised this issue
    severity: Severity
    message: str
    detail: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check": self.check,
            "severity": self.severity.value,
            "message": self.message,
            "detail": self.detail,
        }


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    """
    Aggregated result of all pre-dispatch checks.

    Attributes
    ----------
    passed : bool
        True when no ERROR-severity issues were found.
    issues : List[ValidationIssue]
        All issues found (INFO, WARNING, ERROR).
    provider_id : str
        Provider that was validated against.
    model_id : str
        Model that was validated.
    estimated_max_tokens : int
        Token budget used in validation.
    """
    passed: bool
    issues: List[ValidationIssue]
    provider_id: str
    model_id: str
    estimated_max_tokens: int

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.ERROR]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.WARNING]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "estimated_max_tokens": self.estimated_max_tokens,
            "issues": [i.to_dict() for i in self.issues],
        }


# ---------------------------------------------------------------------------
# Validator configuration
# ---------------------------------------------------------------------------

@dataclass
class RequestValidatorConfig:
    """
    All threshold constants for request validation.

    Attributes
    ----------
    prompt_size_warning_fraction : float
        Warn when prompt tokens exceed this fraction of the context window.
    prompt_size_error_fraction : float
        Error when prompt tokens exceed this fraction of the context window.
    unknown_model_severity : Severity
        Severity to assign when the model is not in the provider's known list.
        INFO = don't block (provider may support unlisted models),
        WARNING = soft warning,
        ERROR = hard block.
    """
    prompt_size_warning_fraction: float = 0.75
    prompt_size_error_fraction: float   = 0.95
    unknown_model_severity: Severity    = Severity.WARNING


# ---------------------------------------------------------------------------
# Request Validator
# ---------------------------------------------------------------------------

class RequestValidator:
    """
    Validates a cloud request against a provider's capability profile
    before dispatch.

    Usage
    -----
    ::

        registry  = ProviderRegistry()
        estimator = TokenEstimator()
        validator = RequestValidator(registry)

        estimate = estimator.estimate(prompt, provider_max_output=profile.max_output_tokens)
        report   = validator.validate(
            provider_id="openrouter",
            model_id="deepseek/deepseek-v4-flash-0731",
            prompt=prompt,
            estimation=estimate,
            required_features={"streaming"},
        )
        if report.passed:
            dispatch(...)
    """

    def __init__(
        self,
        provider_registry: ProviderRegistry,
        config: Optional[RequestValidatorConfig] = None,
    ) -> None:
        self._registry = provider_registry
        self._config   = config or RequestValidatorConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        provider_id: str,
        model_id: str,
        prompt: str,
        estimation: EstimationResult,
        required_features: Optional[Set[str]] = None,
    ) -> ValidationReport:
        """
        Run all four validation checks and return a ValidationReport.

        Parameters
        ----------
        provider_id : str
            Target provider key (e.g. "openrouter").
        model_id : str
            Model identifier (e.g. "deepseek/deepseek-v4-flash-0731").
        prompt : str
            The full prompt text.
        estimation : EstimationResult
            Result from TokenEstimator.estimate().
        required_features : set of str, optional
            Feature names (ProviderFeature values) the request requires.

        Returns
        -------
        ValidationReport
        """
        issues: List[ValidationIssue] = []
        profile: Optional[ProviderProfile] = self._registry.get(provider_id)

        if profile is None:
            issues.append(ValidationIssue(
                check="provider_known",
                severity=Severity.ERROR,
                message=f"Provider '{provider_id}' is not registered.",
                detail=f"Known providers: {self._registry.provider_ids()}",
            ))
            return ValidationReport(
                passed=False,
                issues=issues,
                provider_id=provider_id,
                model_id=model_id,
                estimated_max_tokens=estimation.max_tokens,
            )

        # Check 1: Request size
        self._check_request_size(profile, estimation, issues)

        # Check 2: Provider compatibility
        self._check_provider_compat(
            profile, required_features or set(), issues
        )

        # Check 3: Model availability
        self._check_model_availability(profile, model_id, issues)

        # Check 4: Token budget
        self._check_token_budget(profile, estimation, issues)

        has_errors = any(i.severity == Severity.ERROR for i in issues)
        passed = not has_errors

        report = ValidationReport(
            passed=passed,
            issues=issues,
            provider_id=provider_id,
            model_id=model_id,
            estimated_max_tokens=estimation.max_tokens,
        )

        if not passed:
            logger.warning(
                "RequestValidator: FAILED for provider=%s model=%s — %d error(s).",
                provider_id, model_id, len(report.errors),
            )
        else:
            logger.debug(
                "RequestValidator: PASSED for provider=%s model=%s.",
                provider_id, model_id,
            )

        return report

    # ------------------------------------------------------------------
    # Internal checks
    # ------------------------------------------------------------------

    def _check_request_size(
        self,
        profile: ProviderProfile,
        estimation: EstimationResult,
        issues: List[ValidationIssue],
    ) -> None:
        """Check that the prompt fits within the provider's context window."""
        prompt_tokens = estimation.estimated_prompt_tokens
        context_window = profile.context_window
        request_tokens = prompt_tokens + estimation.max_tokens

        usage_fraction = prompt_tokens / context_window if context_window > 0 else 1.0

        if request_tokens > context_window:
            issues.append(ValidationIssue(
                check="request_size",
                severity=Severity.ERROR,
                message=(
                    f"Estimated prompt ({prompt_tokens} tokens) plus output "
                    f"allowance ({estimation.max_tokens} tokens) exceeds the "
                    f"context window ({context_window} tokens)."
                ),
                detail=f"Estimated total: {request_tokens} tokens",
            ))

        if usage_fraction >= self._config.prompt_size_error_fraction:
            issues.append(ValidationIssue(
                check="request_size",
                severity=Severity.ERROR,
                message=(
                    f"Prompt (~{prompt_tokens} tokens) exceeds "
                    f"{int(self._config.prompt_size_error_fraction * 100)}% of "
                    f"the context window ({context_window} tokens)."
                ),
                detail=f"Usage: {usage_fraction:.1%}",
            ))
        elif usage_fraction >= self._config.prompt_size_warning_fraction:
            issues.append(ValidationIssue(
                check="request_size",
                severity=Severity.WARNING,
                message=(
                    f"Prompt (~{prompt_tokens} tokens) uses "
                    f"{usage_fraction:.1%} of the context window."
                ),
                detail="Consider summarising or truncating the prompt.",
            ))

    def _check_provider_compat(
        self,
        profile: ProviderProfile,
        required_features: Set[str],
        issues: List[ValidationIssue],
    ) -> None:
        """Check that the provider supports all required features."""
        for feature_str in required_features:
            # Accept both raw strings and ProviderFeature values
            try:
                feature = ProviderFeature(feature_str)
            except ValueError:
                issues.append(ValidationIssue(
                    check="provider_compatibility",
                    severity=Severity.WARNING,
                    message=f"Unknown feature requested: '{feature_str}'.",
                    detail="Check ProviderFeature enum for valid values.",
                ))
                continue

            if not profile.supports(feature):
                issues.append(ValidationIssue(
                    check="provider_compatibility",
                    severity=Severity.ERROR,
                    message=(
                        f"Provider '{profile.provider_id}' does not support "
                        f"feature '{feature_str}'."
                    ),
                    detail=(
                        f"Supported features: "
                        f"{[f.value for f in sorted(profile.supported_features)]}"
                    ),
                ))

    def _check_model_availability(
        self,
        profile: ProviderProfile,
        model_id: str,
        issues: List[ValidationIssue],
    ) -> None:
        """Check that the model is listed for this provider."""
        if not profile.has_model(model_id):
            issues.append(ValidationIssue(
                check="model_availability",
                severity=self._config.unknown_model_severity,
                message=(
                    f"Model '{model_id}' is not in the known models list "
                    f"for provider '{profile.provider_id}'."
                ),
                detail=(
                    f"Known models: {profile.models[:5]}"
                    + (" …" if len(profile.models) > 5 else "")
                ),
            ))

    def _check_token_budget(
        self,
        profile: ProviderProfile,
        estimation: EstimationResult,
        issues: List[ValidationIssue],
    ) -> None:
        """Check that max_tokens fits within the provider's output limit."""
        if estimation.max_tokens > profile.max_output_tokens:
            issues.append(ValidationIssue(
                check="token_budget",
                severity=Severity.ERROR,
                message=(
                    f"Requested max_tokens ({estimation.max_tokens}) exceeds "
                    f"provider maximum ({profile.max_output_tokens})."
                ),
                detail=(
                    f"Reduce max_tokens to at most {profile.max_output_tokens}."
                ),
            ))
        elif estimation.max_tokens <= 0:
            issues.append(ValidationIssue(
                check="token_budget",
                severity=Severity.ERROR,
                message=f"max_tokens must be > 0, got {estimation.max_tokens}.",
            ))
