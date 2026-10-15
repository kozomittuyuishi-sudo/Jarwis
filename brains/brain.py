from openai import OpenAI
from dotenv import load_dotenv

from brains.local_brain import ask_local_brain
from brains.cloud_brain import AdaptiveCloudBrain
from brains.cloud_brain.adaptive_brain import BrainStatus
from pipeline.fallback_classifier import classify_failure
from runtime.persona_controller import get_persona_controller

from typing import Optional

import time
import os
import json
import logging
import re
from collections.abc import Mapping
from dataclasses import replace


logger = logging.getLogger(__name__)

_brain_diag_logger = logging.getLogger(
    "ag.brain.diagnostics"
)


# ---------------------------------------------------------------------------
# FALLBACK SENTINEL
# ---------------------------------------------------------------------------

_FALLBACK_PREFIX = "__BRAIN_FALLBACK__:"


# ---------------------------------------------------------------------------
# ENVIRONMENT
# ---------------------------------------------------------------------------

_dotenv_path = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    ),
    ".venv",
    ".env",
)

if os.path.exists(_dotenv_path):
    load_dotenv(dotenv_path=_dotenv_path)
else:
    load_dotenv()


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

BRAIN_CONFIG_FILE = (
    "configuration/brain_config.json"
)


# ---------------------------------------------------------------------------
# SYSTEM PROMPT
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are Jarwis, a personal AI assistant.

You are a context-aware personal operating assistant.
You are not a chatbot, not a smart speaker, and not an emotional companion.

Jarwis identity and operating requirements:
- Identify yourself as Jarwis when asked; AG is the internal project name.
- Persona changes affect presentation only, never capabilities or permissions.
- Do not claim an operation succeeded unless the supplied execution result verifies it.
- Ground factual and self-capability claims in the supplied context.
- Do not let persona instructions override system requirements, safety rules, or results.
- Return only the final user-facing answer. Never expose internal reasoning,
  analysis, response-planning notes, draft instructions, or self-talk.
- Use explicitly supplied recent conversation as current-session context. Do not
  claim that visible current-session context is unavailable, and do not imply
  access to previous sessions or persistent memory unless supplied.

Architecture awareness:
- Local brain = offline Ollama model running on the user's computer.
- Cloud brain = OpenRouter model accessed through internet.
- Auto brain = local first, cloud fallback.
- If the user asks about your local brain, answer in the context of Jarwis itself.

Your role:
Assist the user with projects, memory, productivity, technical explanation, and workflow guidance.
"""


def build_system_prompt(include_persona: bool = True) -> str:
    """Add active style guidance to shared, persona-independent requirements."""
    if not include_persona:
        return SYSTEM_PROMPT
    controller = get_persona_controller()
    return (
        f"{SYSTEM_PROMPT.rstrip()}\n\n"
        f"Active response persona: {controller.mode}.\n"
        f"Persona style instructions: {controller.instructions}\n"
        f"User addressing guidance: {controller.addressing_guidance}"
    )


def _response_field(value, name, default=None):
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _has_response_field(value, name: str) -> bool:
    if isinstance(value, Mapping):
        return name in value
    return hasattr(value, name)


def _response_metadata(response, provider: str, model: str, attempt: int, max_tokens: int) -> dict:
    http_response = _response_field(response, "_response")
    status_code = _response_field(http_response, "status_code")
    if status_code is None:
        status_code = _response_field(response, "status_code")

    choices = _response_field(response, "choices", []) or []
    choice = choices[0] if choices else None
    message = _response_field(choice, "message") if choice is not None else None
    content = _response_field(message, "content") if message is not None else None
    usage = _response_field(response, "usage")
    message_fields = [
        name for name in (
            "role", "content", "refusal", "tool_calls", "reasoning_content",
            "reasoning", "thinking", "analysis",
        )
        if _has_response_field(message, name)
    ]
    content_block_types = []
    if isinstance(content, list):
        content_block_types = [
            _response_field(block, "type", type(block).__name__)
            for block in content[:8]
        ]
    reasoning_lengths = {
        name: len(value)
        for name in ("reasoning_content", "reasoning", "thinking", "analysis")
        if isinstance((value := _response_field(message, name)), str)
    }
    return {
        "provider": provider,
        "model": model,
        "status": status_code if status_code is not None else "unknown",
        "attempt": attempt,
        "max_tokens": max_tokens,
        "response_id": _response_field(response, "id", "unknown"),
        "finish_reason": _response_field(choice, "finish_reason", "unknown"),
        "content_type": type(content).__name__ if content is not None else "null",
        "content_chars": len(content) if isinstance(content, str) else 0,
        "content_is_null": content is None,
        "refusal_present": bool(_response_field(message, "refusal")),
        "tool_call_count": len(_response_field(message, "tool_calls", []) or []),
        "message_fields": message_fields,
        "content_block_types": content_block_types,
        "reasoning_field_chars": reasoning_lengths,
        "prompt_tokens": _response_field(usage, "prompt_tokens", "unknown"),
        "completion_tokens": _response_field(usage, "completion_tokens", "unknown"),
        "total_tokens": _response_field(usage, "total_tokens", "unknown"),
    }


def _extract_response_text(content) -> Optional[str]:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if _response_field(part, "type") not in {"text", "output_text"}:
                continue
            text = _response_field(part, "text")
            if isinstance(text, str):
                text_parts.append(text)
        if text_parts:
            return "".join(text_parts)
    return None


def _safe_error_detail(error: Exception) -> str:
    """Keep provider diagnostics while redacting common credential formats."""
    detail = str(error)
    detail = re.sub(
        r"(?i)(authorization\s*:\s*bearer\s+|api[_ -]?key\s*[=:]\s*)\S+",
        r"\1[REDACTED]",
        detail,
    )
    detail = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "[REDACTED]", detail)
    detail = re.sub(r"\bsk-or-v1-[A-Za-z0-9_-]+\b", "[REDACTED]", detail)
    return detail[:1000]


def _exception_status(error: Exception):
    status = getattr(error, "status_code", None)
    if status is None:
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
    return status if status is not None else "unknown"


# ---------------------------------------------------------------------------
# RUNTIME STATE
# ---------------------------------------------------------------------------

client = None

_adaptive_brain: Optional[
    AdaptiveCloudBrain
] = None


def get_adaptive_brain() -> AdaptiveCloudBrain:
    """Return or create the AdaptiveCloudBrain instance."""

    global _adaptive_brain

    if _adaptive_brain is None:
        _adaptive_brain = AdaptiveCloudBrain()

    return _adaptive_brain


# ---------------------------------------------------------------------------
# OPENROUTER CLIENT WRAPPER
# ---------------------------------------------------------------------------

class ClientWrapper:
    """
    Wrapper around the OpenAI client used for OpenRouter.
    """

    def __init__(self, inner):
        self._inner = inner

    @property
    def chat(self):

        inner = self._inner

        class CompletionsProxy:

            def __init__(self, inner):
                self._inner = inner

            def create(self, *args, **kwargs):

                target = getattr(
                    self._inner,
                    "chat",
                    None
                )

                if target is None:
                    raise RuntimeError(
                        "Underlying client has no "
                        "'chat' attribute"
                    )

                completions = getattr(
                    target,
                    "completions",
                    None
                )

                if completions is None:
                    raise RuntimeError(
                        "Underlying client has no "
                        "'chat.completions' attribute"
                    )

                create_fn = getattr(
                    completions,
                    "create",
                    None
                )

                if create_fn is None:
                    raise RuntimeError(
                        "Underlying client has no "
                        "'chat.completions.create' method"
                    )

                raw_response_factory = getattr(
                    completions,
                    "with_raw_response",
                    None,
                )
                raw_create_fn = getattr(raw_response_factory, "create", None)
                if raw_create_fn is not None:
                    raw_response = raw_create_fn(*args, **kwargs)
                    status_code = getattr(raw_response, "status_code", None)
                    parsed_response = raw_response.parse()
                    return _ProviderResponse(parsed_response, status_code)

                return create_fn(
                    *args,
                    **kwargs
                )

        class ChatProxy:
            completions = CompletionsProxy(
                inner
            )

        return ChatProxy()


class _ProviderResponse:
    """Keep safe transport status metadata beside the parsed provider response."""

    def __init__(self, parsed_response, status_code) -> None:
        self._parsed_response = parsed_response
        self.status_code = status_code

    def __getattr__(self, name):
        return getattr(self._parsed_response, name)


def get_client():

    global client

    if client is None:

        underlying = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv(
                "OPENROUTER_API_KEY"
            ),
            timeout=30,
        )

        client = ClientWrapper(
            underlying
        )

    return client


# ---------------------------------------------------------------------------
# BRAIN MODE
# ---------------------------------------------------------------------------

def load_brain_mode() -> str:

    if not os.path.exists(
        BRAIN_CONFIG_FILE
    ):

        save_brain_mode("auto")
        return "auto"

    try:

        with open(
            BRAIN_CONFIG_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            config = json.load(file)

        mode = config.get(
            "brain_mode",
            "auto"
        )

        if mode not in [
            "local",
            "cloud",
            "auto"
        ]:

            mode = "auto"
            save_brain_mode(mode)

        return mode

    except Exception:

        save_brain_mode("auto")
        return "auto"


def save_brain_mode(mode) -> None:

    directory = os.path.dirname(
        BRAIN_CONFIG_FILE
    )

    if directory:
        os.makedirs(
            directory,
            exist_ok=True
        )

    with open(
        BRAIN_CONFIG_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {
                "brain_mode": mode
            },
            file,
            indent=4
        )


def set_brain_mode(mode) -> str:

    mode = mode.strip().lower()

    if mode not in [
        "local",
        "cloud",
        "auto"
    ]:

        return (
            "Jarwis: Invalid brain mode. "
            "Use local, cloud, or auto."
        )

    save_brain_mode(mode)

    if mode == "local":

        return (
            "Jarwis: Brain mode changed to LOCAL. "
            "Internet dependence reduced. "
            "Civilization improves slightly."
        )

    if mode == "cloud":

        return (
            "Jarwis: Brain mode changed to CLOUD. "
            "Maximum reasoning. Maximum dependence. "
            "Classic bargain."
        )

    return (
        "Jarwis: Brain mode changed to AUTO. "
        "Local first, cloud fallback. "
        "Sensible, annoyingly rare."
    )


def get_brain_status() -> str:

    mode = load_brain_mode()

    return (
        f"Jarwis: Current brain mode: "
        f"{mode.upper()}."
    )


# ---------------------------------------------------------------------------
# CLOUD BRAIN
# ---------------------------------------------------------------------------

def ask_cloud_brain(
    message: str,
    intent: Optional[str] = None,
    include_persona: bool = True,
) -> str:

    adaptive = get_adaptive_brain()

    provider = "openrouter"
    model = (
        "deepseek/deepseek-v4-flash-0731"
    )
    system_prompt = build_system_prompt(include_persona)

    last_response_payload = None
    last_response_details = {}

    def _http_call(
        max_tokens: int
    ) -> str:

        nonlocal last_response_payload, last_response_details
        response = (
            get_client()
            .chat
            .completions
            .create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": message
                    },
                ],
                max_tokens=max_tokens,
            )
        )

        choices = _response_field(response, "choices", []) or []
        choice = choices[0] if choices else None
        message_object = (
            _response_field(choice, "message")
            if choice is not None else None
        )
        raw_content = _response_field(message_object, "content")
        refusal = _response_field(message_object, "refusal")
        content = _extract_response_text(raw_content)
        if content is None and isinstance(refusal, str) and refusal.strip():
            content = refusal
        finish_reason = (
            _response_field(choice, "finish_reason")
            if choice is not None else None
        )

        last_response_details = _response_metadata(
            response, provider, model, attempt, max_tokens,
        )
        last_response_details["content_chars"] = len(content or "")
        last_response_details["extracted_text_is_null"] = content is None
        if finish_reason in {"length", "max_tokens"}:
            _brain_diag_logger.warning(
                "[BRAIN_DIAG] domain=BRAIN code=TRUNCATED_RESPONSE "
                "failure_category=truncated_output metadata=%s",
                last_response_details,
            )
        if choice is None or message_object is None:
            last_response_payload = {"choices": [], "model": model}
            _brain_diag_logger.warning(
                "[BRAIN_DIAG] domain=BRAIN code=INVALID_RESPONSE "
                "failure_category=malformed_response metadata=%s",
                last_response_details,
            )
            return ""

        # Tool calls are not executed by this response-only path.
        last_response_payload = {
            "choices": [{
                "finish_reason": finish_reason,
                "message": {
                    "content": content if content is not None else "",
                    "refusal": refusal,
                },
            }],
            "model": model,
        }
        if content is None or not content.strip():
            _brain_diag_logger.warning(
                "[BRAIN_DIAG] domain=BRAIN code=EMPTY_RESPONSE "
                "failure_category=empty_response metadata=%s",
                last_response_details,
            )
            return ""
        return content


    # ------------------------------------------------------------------
    # PREPARE
    # ------------------------------------------------------------------

    prep = adaptive.prepare(
        provider=provider,
        model=model,
        prompt=f"{system_prompt}\n\nUser message:\n{message}",
        intent=intent,
        task_text=message,
    )
    estimated_prompt_tokens = getattr(
        prep.estimation, "estimated_prompt_tokens", 0,
    )
    estimate_category = getattr(prep.estimation, "category", "unknown")
    estimate_category = getattr(
        estimate_category, "value", estimate_category,
    )
    provider_output_ceiling = getattr(
        prep.estimation, "provider_ceiling", 0,
    )

    logger.info(
        "AdaptiveCloudBrain.prepare: "
        "ready=%s category=%s estimated_prompt_tokens=%d max_tokens=%d "
        "provider_output_ceiling=%d",
        prep.ready,
        estimate_category,
        estimated_prompt_tokens,
        prep.max_tokens,
        provider_output_ceiling,
    )


    if not prep.ready:

        issues = "; ".join(
            f"{i.check}: {i.message}"
            for i in (
                prep
                .validation_report
                .issues
            )
        )

        logger.warning(
            "AdaptiveCloudBrain: "
            "request validation failed: %s",
            issues,
        )

        _brain_diag_logger.warning(
            "[BRAIN_DIAG] domain=BRAIN code=REQUEST_REJECTED "
            "failure_category=request_validation provider=%s model=%s "
            "status=not_dispatched issues=%s",
            provider,
            model,
            [issue.check for issue in prep.validation_report.issues],
        )
        return ""


    # ------------------------------------------------------------------
    # DISPATCH / RETRY LOOP
    # ------------------------------------------------------------------

    attempt = 1

    while True:

        start = time.perf_counter()

        try:

            raw_text = _http_call(
                prep.max_tokens
            )


        except Exception as http_error:

            err_str = str(
                http_error
            ).lower()
            status_code = _exception_status(http_error)

            is_credit_error = (
                str(status_code) == "402"
                or re.search(r"\b402\b", err_str) is not None
            )
            if is_credit_error:
                _brain_diag_logger.error(
                    "[BRAIN_DIAG] domain=BRAIN code=CREDIT_LIMIT "
                    "failure_category=billing_or_credits provider=%s model=%s "
                    "attempt=%d status=402 category=%s prompt_tokens_estimate=%d "
                    "estimated_request_tokens=%d max_tokens=%d "
                    "provider_output_ceiling=%d error_type=%s detail=%r",
                    provider,
                    model,
                    attempt,
                    estimate_category,
                    estimated_prompt_tokens,
                    estimated_prompt_tokens + prep.max_tokens,
                    prep.max_tokens,
                    provider_output_ceiling,
                    type(http_error).__name__,
                    _safe_error_detail(http_error),
                )
                return (
                    "OpenRouter declined the request with HTTP 402, indicating "
                    "a billing or credit limitation. No retry or provider "
                    "switch was attempted. Check the account's billing and "
                    "available credits, or select another configured brain."
                )


            if (
                "timeout" in err_str
                or "timed out" in err_str
            ):

                _brain_diag_logger.warning(
                    "[BRAIN_DIAG] "
                    "domain=BRAIN "
                    "code=TIMEOUT "
                    "provider=%s "
                    "model=%s "
                    "attempt=%d status=%s error_type=%s "
                    "detail=%r",
                    provider,
                    model,
                    attempt,
                    _exception_status(http_error),
                    type(http_error).__name__,
                    _safe_error_detail(http_error),
                )

            else:

                _brain_diag_logger.warning(
                    "[BRAIN_DIAG] "
                    "domain=BRAIN "
                    "code=PROVIDER_ERROR "
                    "provider=%s "
                    "model=%s "
                    "attempt=%d status=%s error_type=%s detail=%r",
                    provider,
                    model,
                    attempt,
                    _exception_status(http_error),
                    type(http_error).__name__,
                    _safe_error_detail(http_error),
                )


            retryable_signals = (
                "insufficient",
                "credits",
                "quota",
                "rate limit",
                "billing",
                "payment",
                "capacity",
                "context length",
                "maximum context",
                "timeout",
                "timed out",
                "connection reset",
                "connection error",
                "temporarily unavailable",
                "service unavailable",
                "internal server error",
            )


            is_retryable = any(
                signal in err_str
                for signal in retryable_signals
            )
            is_retryable = is_retryable or (
                isinstance(status_code, int)
                and (status_code in {408, 409, 429} or 500 <= status_code <= 599)
            )


            if (
                not is_retryable
                or attempt >= 3
            ):

                raise


            raw_response_dict = {
                "error_type": type(http_error).__name__,
                "choices": [],
            }


            latency_ms = (
                time.perf_counter()
                - start
            ) * 1000.0


            result = (
                adaptive
                .validate_response(
                    raw_response=(
                        raw_response_dict
                    ),
                    prep=prep,
                    attempt_number=attempt,
                    latency_ms=latency_ms,
                )
            )


            if (
                result.retry_outcome
                and
                result.retry_outcome.should_retry
            ):

                adjusted_tokens = (
                    result
                    .retry_outcome
                    .adjusted_max_tokens
                )
                if adjusted_tokens is not None:
                    prep = replace(prep, max_tokens=adjusted_tokens)

                attempt += 1

                logger.warning(
                    "AdaptiveCloudBrain: "
                    "retrying attempt %d "
                    "with max_tokens=%d",
                    attempt,
                    prep.max_tokens,
                )

                continue


            raise


        # ------------------------------------------------------------------
        # RESPONSE VALIDATION
        # ------------------------------------------------------------------

        latency_ms = (
            time.perf_counter()
            - start
        ) * 1000.0


        raw_response_dict = last_response_payload or {
            "choices": [{"message": {"content": raw_text}}],
            "model": model,
        }


        result = adaptive.validate_response(
            raw_response=raw_response_dict,
            prep=prep,
            attempt_number=attempt,
            latency_ms=latency_ms,
        )

        response_result = getattr(result, "response_result", None)
        response_failures = (
            getattr(response_result, "failures", [])
            if response_result is not None else []
        )
        if any(
            "internal response-planning text" in str(issue.message).lower()
            for issue in response_failures
        ):
            _brain_diag_logger.warning(
                "[BRAIN_DIAG] domain=BRAIN code=INTERNAL_PLANNING_LEAK "
                "failure_category=invalid_user_facing_content metadata=%s",
                last_response_details,
            )


        logger.info(
            "AdaptiveCloudBrain."
            "validate_response: "
            "status=%s attempt=%d",
            result.status.value,
            attempt,
        )


        if result.status == BrainStatus.PASSED and raw_text and raw_text.strip():

            return raw_text


        # ------------------------------------------------------------------
        # VALIDATION RETRY
        # ------------------------------------------------------------------

        if (
            result.status
            == BrainStatus.RETRYING
            and attempt < 3
        ):

            truncated_response = last_response_details.get("finish_reason") in {
                "length", "max_tokens",
            }
            can_grow_budget = False
            if result.retry_outcome and result.retry_outcome.adjusted_max_tokens is not None:
                retry_tokens = result.retry_outcome.adjusted_max_tokens
                if truncated_response:
                    ceiling = prep.estimation.provider_ceiling
                    retry_tokens = min(
                        ceiling,
                        max(prep.max_tokens + 1, prep.max_tokens * 2),
                    )
                    can_grow_budget = retry_tokens > prep.max_tokens
                elif not raw_text:
                    retry_tokens = prep.max_tokens
                if last_response_details.get("tool_call_count", 0):
                    retry_tokens = prep.max_tokens
                prep = replace(prep, max_tokens=retry_tokens)

            # An empty result without an explicit truncation signal receives
            # one same-budget retry; repeated empty completions stop promptly.
            max_attempts = (
                3 if can_grow_budget else 1 if truncated_response else 2
            )
            if not raw_text and last_response_details.get("tool_call_count", 0):
                max_attempts = 1
            if attempt >= max_attempts:
                _brain_diag_logger.error(
                    "[BRAIN_DIAG] domain=BRAIN code=EMPTY_RESPONSE "
                    "failure_category=empty_response attempts=%d metadata=%s",
                    attempt,
                    last_response_details,
                )
                return ""
            attempt += 1

            logger.warning(
                "AdaptiveCloudBrain: "
                "response failed validation, "
                "retrying attempt %d "
                "with max_tokens=%d",
                attempt,
                prep.max_tokens,
            )

            continue


        # ------------------------------------------------------------------
        # FINAL VALIDATION FAILURE
        # ------------------------------------------------------------------

        logger.warning(
            "AdaptiveCloudBrain: "
            "pipeline status=%s, "
            "response failed validation; returning fallback.",
            result.status.value,
        )


        _brain_diag_logger.warning(
            "[BRAIN_DIAG] "
            "domain=BRAIN "
            "code=%s "
            "failure_category=invalid_response "
            "status=%s metadata=%s",
            "EMPTY_RESPONSE" if not raw_text else "INVALID_RESPONSE",
            result.status.value,
            last_response_details,
        )

        return ""


# ---------------------------------------------------------------------------
# LOCAL BRAIN
# ---------------------------------------------------------------------------

def ask_local(message, include_persona: bool = True) -> str:

    local_prompt = f"""
{build_system_prompt(include_persona)}

User message:
{message}
"""

    try:

        result = ask_local_brain(
            local_prompt
        )

        return (
            result
            if result is not None
            else ""
        )

    except Exception as error:

        _brain_diag_logger.warning(
            "[BRAIN_DIAG] "
            "domain=BRAIN "
            "code=LOCAL_PROVIDER_ERROR "
            "detail=%r",
            _safe_error_detail(error),
        )

        raise


# ---------------------------------------------------------------------------
# MAIN BRAIN ENTRY POINT
# ---------------------------------------------------------------------------

def ask_brain(message, include_persona: bool = True) -> str:

    start = time.time()

    mode = load_brain_mode()


    # ======================================================================
    # LOCAL MODE
    # ======================================================================

    if mode == "local":

        try:

            reply = ask_local(
                message,
                include_persona=include_persona,
            )

            print(
                "[AG DEBUG] "
                "Local Brain Response Time: "
                f"{time.time() - start:.2f}s"
            )


            if (
                reply
                and reply.strip()
            ):

                return reply


            raw_err = (
                "Local brain returned "
                "no response."
            )


            result = classify_failure(
                raw_err
            )


            return (
                f"{_FALLBACK_PREFIX}"
                f"{result.user_message}"
            )


        except Exception as error:

            raw_err = (
                f"Local brain failed: "
                f"{_safe_error_detail(error)}"
            )


            logger.debug(
                "[AG DEBUG] %s",
                _safe_error_detail(error)
            )


            result = classify_failure(
                raw_err
            )


            return (
                f"{_FALLBACK_PREFIX}"
                f"{result.user_message}"
            )


    # ======================================================================
    # CLOUD MODE
    # ======================================================================

    if mode == "cloud":

        try:

            reply = ask_cloud_brain(
                message,
                include_persona=include_persona,
            )

            print(
                "[AG DEBUG] "
                "Cloud Brain Response Time: "
                f"{time.time() - start:.2f}s"
            )


            if (
                reply
                and reply.strip()
            ):

                return reply


            raw_err = (
                "Cloud brain returned "
                "no response."
            )


            result = classify_failure(
                raw_err
            )


            return (
                f"{_FALLBACK_PREFIX}"
                f"{result.user_message}"
            )


        except Exception as error:

            raw_err = (
                f"Cloud brain failed: "
                f"{_safe_error_detail(error)}"
            )


            logger.debug(
                "[AG DEBUG] %s",
                _safe_error_detail(error)
            )


            result = classify_failure(
                raw_err
            )


            return (
                f"{_FALLBACK_PREFIX}"
                f"{result.user_message}"
            )


    # ======================================================================
    # AUTO MODE
    # Local first -> Cloud fallback
    # ======================================================================

    try:

        reply = ask_local(
            message,
            include_persona=include_persona,
        )

        print(
            "[AG DEBUG] "
            "Local Brain Response Time: "
            f"{time.time() - start:.2f}s"
        )


        if (
            reply
            and reply.strip()
        ):

            return reply


        local_error = (
            "Local brain returned "
            "no response."
        )


    except Exception as local_err:

        local_error = (
            f"Local brain failed: "
            f"{_safe_error_detail(local_err)}"
        )


    logger.debug(
        "[AG DEBUG] %s",
        _safe_error_detail(Exception(local_error))
    )


    print(
        "[AG DEBUG] "
        "Switching to cloud brain."
    )


    _brain_diag_logger.info(
        "[BRAIN_DIAG] "
        "local_failed=%r "
        "switching_to_cloud=True",
        local_error,
    )


    # ------------------------------------------------------------------
    # CLOUD FALLBACK
    # ------------------------------------------------------------------

    try:

        reply = ask_cloud_brain(
            message,
            include_persona=include_persona,
        )

        print(
            "[AG DEBUG] "
            "Cloud Brain Response Time: "
            f"{time.time() - start:.2f}s"
        )


        if (
            reply
            and reply.strip()
        ):

            return reply


        cloud_error = (
            "Cloud brain returned "
            "no response."
        )


    except Exception as cloud_err:

        cloud_error = (
            f"Cloud brain failed: "
            f"{_safe_error_detail(cloud_err)}"
        )


    logger.debug(
        "[AG DEBUG] %s",
        _safe_error_detail(Exception(cloud_error))
    )


    # ------------------------------------------------------------------
    # BOTH BRAINS FAILED
    # ------------------------------------------------------------------

    combined = (
        "Brain connection failed. "
        f"Local error: {local_error}. "
        f"Cloud error: {cloud_error}."
    )


    logger.debug(
        "[AG DEBUG] %s",
        _safe_error_detail(Exception(combined))
    )


    result = classify_failure(
        combined
    )


    return (
        f"{_FALLBACK_PREFIX}"
        f"{result.user_message}"
    )


# ---------------------------------------------------------------------------
# FORCED CLOUD
# ---------------------------------------------------------------------------

def ask_cloud_direct(
    message
) -> str:

    start = time.time()


    try:

        reply = ask_cloud_brain(
            message
        )

        print(
            "[AG DEBUG] "
            "Forced Cloud Brain Response Time: "
            f"{time.time() - start:.2f}s"
        )


        if (
            reply
            and reply.strip()
        ):

            return reply


        raw_err = (
            "Cloud brain returned "
            "no response."
        )


        result = classify_failure(
            raw_err
        )


        return (
            f"{_FALLBACK_PREFIX}"
            f"{result.user_message}"
        )


    except Exception as error:

        raw_err = (
            f"Forced cloud brain failed: "
            f"{_safe_error_detail(error)}"
        )


        logger.debug(
            "[AG DEBUG] %s",
            _safe_error_detail(Exception(raw_err))
        )


        result = classify_failure(
            raw_err
        )


        return (
            f"{_FALLBACK_PREFIX}"
            f"{result.user_message}"
        )