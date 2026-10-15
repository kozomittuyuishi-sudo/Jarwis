"""
cloud_brain/response_validator.py
===================================
Response Validator

Validates a raw provider response across six axes after dispatch:

    1. empty_response   — response has non-empty, non-whitespace content
    2. malformed        — response has the expected structural fields
    3. tool_output      — tool call outputs are structurally valid (if present)
    4. finish_reason    — finish_reason is an accepted value
    5. reasoning_blocks — reasoning block structure is sound (if present)
    6. message_content  — message content passes basic sanity checks

Returns a ValidationResult containing one AxisResult per axis.
A result is passing when all non-skipped axes pass.

Design principles
-----------------
- Stdlib only. No I/O. No LLM calls.
- All accepted finish reasons live in config, not in logic.
- Never raises on unexpected response shape — returns a structured failure.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_INTERNAL_PLANNING_PATTERNS = (
    re.compile(
        r"\b(?:let me|i(?:'ll| will)|we need to)\s+(?:now\s+)?"
        r"(?:craft|compose|draft)\s+(?:a\s+)?(?:response|reply|answer)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bappropriately\s*:\s*(?:acknowledge|explain|answer|respond|"
        r"clarify|summarize)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:analysis|internal reasoning|chain of thought)\s*[:\-]",
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(r"<\s*think(?:ing)?\s*>", re.IGNORECASE),
)


def _contains_internal_planning(text: str) -> bool:
    return any(pattern.search(text) for pattern in _INTERNAL_PLANNING_PATTERNS)


# ---------------------------------------------------------------------------
# Validation axes
# ---------------------------------------------------------------------------

class ValidationAxis(str, Enum):
    EMPTY_RESPONSE   = "empty_response"
    MALFORMED        = "malformed"
    TOOL_OUTPUT      = "tool_output"
    FINISH_REASON    = "finish_reason"
    REASONING_BLOCKS = "reasoning_blocks"
    MESSAGE_CONTENT  = "message_content"


# ---------------------------------------------------------------------------
# Per-axis result
# ---------------------------------------------------------------------------

@dataclass
class AxisResult:
    """Result of a single validation axis."""
    axis: ValidationAxis
    passed: bool
    skipped: bool = False
    message: Optional[str] = None
    detail: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "axis": self.axis.value,
            "passed": self.passed,
            "skipped": self.skipped,
            "message": self.message,
            "detail": self.detail,
        }


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """
    Aggregated result of all six response-validation axes.

    Attributes
    ----------
    passed : bool
        True when every non-skipped axis passed.
    axes : List[AxisResult]
        Per-axis detail.
    response_id : Optional[str]
        Provider-assigned response ID if present in the raw response.
    """
    passed: bool
    axes: List[AxisResult]
    response_id: Optional[str] = None

    @property
    def failures(self) -> List[AxisResult]:
        return [r for r in self.axes if not r.passed and not r.skipped]

    def axis_result(self, axis: ValidationAxis) -> Optional[AxisResult]:
        for r in self.axes:
            if r.axis == axis:
                return r
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "response_id": self.response_id,
            "axes": [r.to_dict() for r in self.axes],
        }


# ---------------------------------------------------------------------------
# Validator configuration
# ---------------------------------------------------------------------------

@dataclass
class ResponseValidatorConfig:
    """
    All thresholds and accepted values for response validation.

    Attributes
    ----------
    accepted_finish_reasons : List[str]
        Finish reasons that are considered normal completions.
    warn_finish_reasons : List[str]
        Finish reasons that pass validation but trigger a warning.
    min_content_length : int
        Minimum non-whitespace characters for message content to be non-empty.
    max_content_length : int
        Maximum characters before issuing a length warning (0 = no limit).
    """
    accepted_finish_reasons: List[str] = field(default_factory=lambda: [
        "stop", "end_turn", "tool_calls", "function_call", "content_filter",
    ])
    warn_finish_reasons: List[str] = field(default_factory=list)
    min_content_length: int = 1
    max_content_length: int = 0      # 0 = no maximum enforced


# ---------------------------------------------------------------------------
# Response Validator
# ---------------------------------------------------------------------------

class ResponseValidator:
    """
    Validates a raw provider response dict across six axes.

    Expected response structure (OpenAI-compatible schema):
    ::

        {
            "id": "...",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "...",
                        "tool_calls": [...],   # optional
                        "reasoning_content": "..."  # optional
                    },
                    "finish_reason": "stop"
                }
            ]
        }

    Usage
    -----
    ::

        validator = ResponseValidator()
        result    = validator.validate(raw_response)
        if not result.passed:
            for failure in result.failures:
                print(failure.axis.value, failure.message)
    """

    def __init__(self, config: Optional[ResponseValidatorConfig] = None) -> None:
        self._config = config or ResponseValidatorConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, raw_response: Any) -> ValidationResult:
        """
        Validate ``raw_response`` across all six axes.

        ``raw_response`` should be a dict (or dict-like object) returned
        directly by the provider SDK.  The validator is tolerant of
        missing fields — it reports them as failures rather than raising.

        Returns
        -------
        ValidationResult
        """
        axes: List[AxisResult] = []

        # --- Axis 1: empty response ---
        axes.append(self._check_empty_response(raw_response))

        # --- Axis 2: malformed structure ---
        axes.append(self._check_malformed(raw_response))

        # For the remaining axes we need to reach into choices[0].message;
        # only attempt if the structure check passed.
        structure_ok = axes[1].passed
        message: Optional[Dict[str, Any]] = None
        choice:  Optional[Dict[str, Any]] = None

        if structure_ok and isinstance(raw_response, dict):
            choices = raw_response.get("choices") or []
            if choices and isinstance(choices[0], dict):
                choice  = choices[0]
                message = choice.get("message") or {}

        # --- Axis 3: tool output ---
        axes.append(self._check_tool_output(message))

        # --- Axis 4: finish reason ---
        axes.append(self._check_finish_reason(choice))

        # --- Axis 5: reasoning blocks ---
        axes.append(self._check_reasoning_blocks(message))

        # --- Axis 6: message content ---
        axes.append(self._check_message_content(message))

        passed = all(r.passed or r.skipped for r in axes)

        response_id: Optional[str] = None
        if isinstance(raw_response, dict):
            response_id = raw_response.get("id")

        result = ValidationResult(passed=passed, axes=axes, response_id=response_id)

        if not passed:
            failed_names = [r.axis.value for r in result.failures]
            logger.warning(
                "ResponseValidator: FAILED on axes=%s response_id=%s",
                failed_names, response_id,
            )
        else:
            logger.debug(
                "ResponseValidator: PASSED response_id=%s", response_id
            )

        return result

    # ------------------------------------------------------------------
    # Axis implementations
    # ------------------------------------------------------------------

    def _check_empty_response(self, raw_response: Any) -> AxisResult:
        """Axis 1: Reject completely empty or None responses."""
        if raw_response is None:
            return AxisResult(
                axis=ValidationAxis.EMPTY_RESPONSE,
                passed=False,
                message="Response is None.",
            )
        if isinstance(raw_response, str) and not raw_response.strip():
            return AxisResult(
                axis=ValidationAxis.EMPTY_RESPONSE,
                passed=False,
                message="Response is an empty string.",
            )
        if isinstance(raw_response, dict) and not raw_response:
            return AxisResult(
                axis=ValidationAxis.EMPTY_RESPONSE,
                passed=False,
                message="Response is an empty dict.",
            )
        return AxisResult(axis=ValidationAxis.EMPTY_RESPONSE, passed=True)

    def _check_malformed(self, raw_response: Any) -> AxisResult:
        """Axis 2: Verify the response has the expected structural fields."""
        if not isinstance(raw_response, dict):
            return AxisResult(
                axis=ValidationAxis.MALFORMED,
                passed=False,
                message=f"Response is not a dict (got {type(raw_response).__name__}).",
            )

        choices = raw_response.get("choices")
        if choices is None:
            return AxisResult(
                axis=ValidationAxis.MALFORMED,
                passed=False,
                message="Response missing 'choices' field.",
            )

        if not isinstance(choices, list) or len(choices) == 0:
            return AxisResult(
                axis=ValidationAxis.MALFORMED,
                passed=False,
                message="'choices' must be a non-empty list.",
                detail=f"Got: {type(choices).__name__}",
            )

        first = choices[0]
        if not isinstance(first, dict):
            return AxisResult(
                axis=ValidationAxis.MALFORMED,
                passed=False,
                message="choices[0] is not a dict.",
            )

        if "message" not in first and "text" not in first:
            return AxisResult(
                axis=ValidationAxis.MALFORMED,
                passed=False,
                message="choices[0] has neither 'message' nor 'text' field.",
            )

        return AxisResult(axis=ValidationAxis.MALFORMED, passed=True)

    def _check_tool_output(self, message: Optional[Dict[str, Any]]) -> AxisResult:
        """Axis 3: Validate tool_calls structure if present."""
        if message is None:
            return AxisResult(
                axis=ValidationAxis.TOOL_OUTPUT,
                passed=True,
                skipped=True,
                message="No message object; axis skipped.",
            )

        tool_calls = message.get("tool_calls")
        if tool_calls is None:
            # No tool calls — axis not applicable
            return AxisResult(
                axis=ValidationAxis.TOOL_OUTPUT,
                passed=True,
                skipped=True,
                message="No tool_calls present; axis skipped.",
            )

        if not isinstance(tool_calls, list):
            return AxisResult(
                axis=ValidationAxis.TOOL_OUTPUT,
                passed=False,
                message=f"'tool_calls' must be a list, got {type(tool_calls).__name__}.",
            )

        for idx, tc in enumerate(tool_calls):
            if not isinstance(tc, dict):
                return AxisResult(
                    axis=ValidationAxis.TOOL_OUTPUT,
                    passed=False,
                    message=f"tool_calls[{idx}] is not a dict.",
                )
            # Expect 'id', 'type', 'function' keys
            missing = [k for k in ("id", "type", "function") if k not in tc]
            if missing:
                return AxisResult(
                    axis=ValidationAxis.TOOL_OUTPUT,
                    passed=False,
                    message=f"tool_calls[{idx}] missing keys: {missing}.",
                )
            fn = tc.get("function")
            if not isinstance(fn, dict) or "name" not in fn:
                return AxisResult(
                    axis=ValidationAxis.TOOL_OUTPUT,
                    passed=False,
                    message=f"tool_calls[{idx}].function must be a dict with 'name'.",
                )

        return AxisResult(
            axis=ValidationAxis.TOOL_OUTPUT,
            passed=True,
            message=f"{len(tool_calls)} tool call(s) structurally valid.",
        )

    def _check_finish_reason(self, choice: Optional[Dict[str, Any]]) -> AxisResult:
        """Axis 4: Verify finish_reason is an accepted value."""
        if choice is None:
            return AxisResult(
                axis=ValidationAxis.FINISH_REASON,
                passed=True,
                skipped=True,
                message="No choice object; axis skipped.",
            )

        finish_reason = choice.get("finish_reason")

        if finish_reason is None:
            # Some providers omit finish_reason; treat as acceptable
            return AxisResult(
                axis=ValidationAxis.FINISH_REASON,
                passed=True,
                message="finish_reason absent; treating as acceptable.",
            )

        if finish_reason not in self._config.accepted_finish_reasons:
            return AxisResult(
                axis=ValidationAxis.FINISH_REASON,
                passed=False,
                message=f"Unexpected finish_reason: '{finish_reason}'.",
                detail=(
                    f"Accepted: {self._config.accepted_finish_reasons}"
                ),
            )

        if finish_reason in self._config.warn_finish_reasons:
            return AxisResult(
                axis=ValidationAxis.FINISH_REASON,
                passed=True,
                message=f"finish_reason '{finish_reason}' indicates truncation.",
            )

        return AxisResult(
            axis=ValidationAxis.FINISH_REASON,
            passed=True,
            message=f"finish_reason '{finish_reason}' is acceptable.",
        )

    def _check_reasoning_blocks(self, message: Optional[Dict[str, Any]]) -> AxisResult:
        """Axis 5: Validate reasoning block structure if present."""
        if message is None:
            return AxisResult(
                axis=ValidationAxis.REASONING_BLOCKS,
                passed=True,
                skipped=True,
                message="No message object; axis skipped.",
            )

        # Reasoning blocks appear under various keys depending on provider
        reasoning = (
            message.get("reasoning_content")
            or message.get("thinking")
            or message.get("reasoning")
        )

        if reasoning is None:
            return AxisResult(
                axis=ValidationAxis.REASONING_BLOCKS,
                passed=True,
                skipped=True,
                message="No reasoning blocks; axis skipped.",
            )

        # If it's a string, that's fine
        if isinstance(reasoning, str):
            return AxisResult(
                axis=ValidationAxis.REASONING_BLOCKS,
                passed=True,
                message=f"Reasoning block present ({len(reasoning)} chars).",
            )

        # If it's a list, each entry must be a dict with a 'type' key
        if isinstance(reasoning, list):
            for idx, block in enumerate(reasoning):
                if not isinstance(block, dict):
                    return AxisResult(
                        axis=ValidationAxis.REASONING_BLOCKS,
                        passed=False,
                        message=f"reasoning[{idx}] is not a dict.",
                    )
                if "type" not in block:
                    return AxisResult(
                        axis=ValidationAxis.REASONING_BLOCKS,
                        passed=False,
                        message=f"reasoning[{idx}] missing 'type' key.",
                    )
            return AxisResult(
                axis=ValidationAxis.REASONING_BLOCKS,
                passed=True,
                message=f"{len(reasoning)} reasoning block(s) structurally valid.",
            )

        return AxisResult(
            axis=ValidationAxis.REASONING_BLOCKS,
            passed=False,
            message=(
                f"Unexpected reasoning block type: {type(reasoning).__name__}."
            ),
        )

    def _check_message_content(self, message: Optional[Dict[str, Any]]) -> AxisResult:
        """Axis 6: Basic sanity check on message content."""
        if message is None:
            return AxisResult(
                axis=ValidationAxis.MESSAGE_CONTENT,
                passed=True,
                skipped=True,
                message="No message object; axis skipped.",
            )

        content = message.get("content")

        # content is None when tool_calls are present — that's allowed
        tool_calls = message.get("tool_calls")
        if content is None and tool_calls:
            return AxisResult(
                axis=ValidationAxis.MESSAGE_CONTENT,
                passed=True,
                message="Content is None but tool_calls present; acceptable.",
            )

        if content is None:
            return AxisResult(
                axis=ValidationAxis.MESSAGE_CONTENT,
                passed=False,
                message="Message content is None and no tool_calls present.",
            )

        if isinstance(content, str):
            stripped = content.strip()
            if len(stripped) < self._config.min_content_length:
                return AxisResult(
                    axis=ValidationAxis.MESSAGE_CONTENT,
                    passed=False,
                    message=(
                        f"Message content too short "
                        f"({len(stripped)} chars, min={self._config.min_content_length})."
                    ),
                )
            if _contains_internal_planning(stripped):
                return AxisResult(
                    axis=ValidationAxis.MESSAGE_CONTENT,
                    passed=False,
                    message="Message contains internal response-planning text.",
                )
            if self._config.max_content_length > 0 and len(stripped) > self._config.max_content_length:
                return AxisResult(
                    axis=ValidationAxis.MESSAGE_CONTENT,
                    passed=True,
                    message=(
                        f"Content length {len(stripped)} exceeds soft maximum "
                        f"{self._config.max_content_length}."
                    ),
                )
            return AxisResult(
                axis=ValidationAxis.MESSAGE_CONTENT,
                passed=True,
                message=f"Content: {len(stripped)} chars.",
            )

        # Multimodal content (list of parts)
        if isinstance(content, list):
            if len(content) == 0:
                return AxisResult(
                    axis=ValidationAxis.MESSAGE_CONTENT,
                    passed=False,
                    message="Message content is an empty list.",
                )
            text_count = 0
            for index, part in enumerate(content):
                if not isinstance(part, dict):
                    return AxisResult(
                        axis=ValidationAxis.MESSAGE_CONTENT,
                        passed=False,
                        message=f"Content block {index} is not an object.",
                    )
                if part.get("type") not in {"text", "output_text"}:
                    return AxisResult(
                        axis=ValidationAxis.MESSAGE_CONTENT,
                        passed=False,
                        message=f"Unsupported content block type at index {index}.",
                    )
                text = part.get("text")
                if not isinstance(text, str) or not text.strip():
                    return AxisResult(
                        axis=ValidationAxis.MESSAGE_CONTENT,
                        passed=False,
                        message=f"Content block {index} has no usable text.",
                    )
                if _contains_internal_planning(text):
                    return AxisResult(
                        axis=ValidationAxis.MESSAGE_CONTENT,
                        passed=False,
                        message="Message contains internal response-planning text.",
                    )
                text_count += 1
            return AxisResult(
                axis=ValidationAxis.MESSAGE_CONTENT,
                passed=text_count > 0,
                message=f"Text content: {text_count} block(s).",
            )

        return AxisResult(
            axis=ValidationAxis.MESSAGE_CONTENT,
            passed=False,
            message=f"Unexpected content type: {type(content).__name__}.",
        )
