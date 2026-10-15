"""
cloud_brain/
============
Adaptive Cloud Brain — Part A

Intelligent middleware that prepares every cloud request before it
reaches any provider.  This package is a standalone subsystem.

It does NOT integrate with Jarwis.py and does NOT modify runtime behaviour.

Public API
----------
AdaptiveCloudBrain      Top-level orchestrator.
ProviderRegistry        Provider capability detector.
TokenEstimator          Token budget manager.
RequestValidator        Pre-dispatch request validation.
ResponseValidator       Post-dispatch response validation.
RetryManager            Retry-on-failure with reduced budget.
RequestLogger           Structured request/response logging.

Usage
-----
::

    from cloud_brain import AdaptiveCloudBrain, ProviderRegistry

    registry = ProviderRegistry()
    brain    = AdaptiveCloudBrain(provider_registry=registry)

    result = brain.prepare_and_validate(
        provider="openrouter",
        model="deepseek/deepseek-v4-flash-0731",
        prompt="Explain quantum entanglement.",
        category="reasoning",
    )
"""

from brains.cloud_brain.provider_registry import (
    ProviderRegistry,
    ProviderProfile,
    ProviderFeature,
)
from brains.cloud_brain.token_estimator import (
    TokenEstimator,
    EstimationResult,
    RequestCategory,
)
from brains.cloud_brain.request_validator import (
    RequestValidator,
    ValidationReport,
    ValidationIssue,
    Severity,
)
from brains.cloud_brain.response_validator import (
    ResponseValidator,
    ValidationResult,
    ValidationAxis,
    AxisResult,
)
from brains.cloud_brain.retry_manager import (
    RetryManager,
    RetryOutcome,
    RetryStatus,
)
from brains.cloud_brain.request_logger import (
    RequestLogger,
    RequestRecord,
    LogLevel,
)
from brains.cloud_brain.adaptive_brain import (
    AdaptiveCloudBrain,
    BrainResult,
    BrainStatus,
)

__all__ = [
    # Orchestrator
    "AdaptiveCloudBrain",
    "BrainResult",
    "BrainStatus",
    # Provider registry
    "ProviderRegistry",
    "ProviderProfile",
    "ProviderFeature",
    # Token estimator
    "TokenEstimator",
    "EstimationResult",
    "RequestCategory",
    # Request validator
    "RequestValidator",
    "ValidationReport",
    "ValidationIssue",
    "Severity",
    # Response validator
    "ResponseValidator",
    "ValidationResult",
    "ValidationAxis",
    "AxisResult",
    # Retry manager
    "RetryManager",
    "RetryOutcome",
    "RetryStatus",
    # Logger
    "RequestLogger",
    "RequestRecord",
    "LogLevel",
]
