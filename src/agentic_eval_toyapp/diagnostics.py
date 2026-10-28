"""Safe operational diagnostics for agent requests.

Only fixed categories and bounded operational identifiers enter log messages.
Exception messages, model prompts, completions, and report contents do not.
"""

import logging
import re
import ssl
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from time import perf_counter
from typing import Any, cast
from uuid import UUID, uuid4

from openai import APIConnectionError, APIStatusError, APITimeoutError

LOGGER = logging.getLogger("uvicorn.error")
_SAFE_IDENTIFIER = re.compile(r"[^A-Za-z0-9._/-]")


@dataclass(frozen=True, slots=True)
class DiagnosticContext:
    """Identifiers shared by nested operations in one API request."""

    session_id: str = "-"
    turn_id: str = "-"


@dataclass(frozen=True, slots=True)
class CompletionDiagnostics:
    """Safe, optional metadata from one structured completion response."""

    finish_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None

    @classmethod
    def from_response(cls, response: Any) -> "CompletionDiagnostics":
        """Read only allowlisted values from an OpenAI-compatible response."""
        choices: Any = getattr(response, "choices", None)
        first_choice: Any = (
            cast(Any, choices[0])
            if isinstance(choices, (list, tuple)) and choices
            else None
        )
        raw_reason = getattr(first_choice, "finish_reason", None)
        finish_reason = (
            raw_reason
            if isinstance(raw_reason, str)
            and raw_reason
            in {"stop", "length", "content_filter", "tool_calls", "function_call"}
            else "other"
            if raw_reason is not None
            else None
        )
        usage: Any = getattr(response, "usage", None)
        details: Any = getattr(usage, "completion_tokens_details", None)
        return cls(
            finish_reason=finish_reason,
            input_tokens=_safe_token_count(getattr(usage, "prompt_tokens", None)),
            output_tokens=_safe_token_count(getattr(usage, "completion_tokens", None)),
            reasoning_tokens=_safe_token_count(
                getattr(details, "reasoning_tokens", None)
            ),
            total_tokens=_safe_token_count(getattr(usage, "total_tokens", None)),
        )


def _safe_token_count(value: object) -> int | None:
    """Reject malformed or implausible upstream values before logging."""
    return value if type(value) is int and 0 <= value <= 1_000_000_000 else None


_CONTEXT: ContextVar[DiagnosticContext | None] = ContextVar(
    "toyapp_diagnostic_context", default=None
)


@contextmanager
def diagnostic_context(
    session_id: UUID, *, turn: bool = False
) -> Generator[DiagnosticContext, None, None]:
    """Bind safe IDs for an upload or one completed-message attempt."""
    context = DiagnosticContext(
        session_id=str(session_id),
        turn_id=str(uuid4()) if turn else "-",
    )
    token = _CONTEXT.set(context)
    try:
        yield context
    finally:
        _CONTEXT.reset(token)


def _safe_identifier(value: str | None) -> str:
    if not value:
        return "-"
    return _SAFE_IDENTIFIER.sub("_", value[:80])


def elapsed_milliseconds(started_at: float) -> int:
    """Return a non-negative elapsed duration without exposing wall-clock time."""
    return max(0, round((perf_counter() - started_at) * 1000))


def classify_external_error(error: BaseException) -> tuple[str, int | None]:
    """Classify SDK failures using types/status, never exception text."""
    causes: list[BaseException] = []
    current: BaseException | None = error
    while current is not None and len(causes) < 8:
        if any(current is seen for seen in causes):
            break
        causes.append(current)
        current = current.__cause__ or current.__context__

    if any(isinstance(cause, (APITimeoutError, TimeoutError)) for cause in causes):
        return "timeout", None
    if any(isinstance(cause, ssl.SSLError) for cause in causes):
        return "tls_error", None
    if any(isinstance(cause, PermissionError) for cause in causes):
        return "permission_error", None
    status_error = next(
        (cause for cause in causes if isinstance(cause, APIStatusError)), None
    )
    if status_error is not None:
        status = status_error.status_code
        if status in (401, 403):
            return "authentication_error", status
        if status == 429:
            return "rate_limited", status
        if status >= 500:
            return "provider_server_error", status
        return "provider_http_error", status
    if isinstance(error, APIConnectionError) or any(
        isinstance(cause, ConnectionError) for cause in causes
    ):
        return "connection_error", None
    return "unexpected_error", None


def log_failure(
    *,
    event: str,
    agent: str,
    operation: str,
    code: str,
    level: int = logging.ERROR,
    model: str | None = None,
    attempt: int | None = None,
    max_tokens: int | None = None,
    elapsed_ms: int | None = None,
    status_code: int | None = None,
    error: BaseException | None = None,
    finding_codes: tuple[str, ...] = (),
    completion: CompletionDiagnostics | None = None,
) -> None:
    """Emit one grep-friendly event using safe, fixed operational fields."""
    context = _CONTEXT.get() or DiagnosticContext()
    completion = completion or CompletionDiagnostics()
    LOGGER.log(
        level,
        "toyapp_failure event=%s session_id=%s turn_id=%s agent=%s "
        "operation=%s code=%s model=%s attempt=%s max_tokens=%s "
        "elapsed_ms=%s status_code=%s exception_type=%s finding_codes=%s "
        "finish_reason=%s input_tokens=%s output_tokens=%s reasoning_tokens=%s "
        "total_tokens=%s",
        _safe_identifier(event),
        context.session_id,
        context.turn_id,
        _safe_identifier(agent),
        _safe_identifier(operation),
        _safe_identifier(code),
        _safe_identifier(model),
        attempt if attempt is not None else "-",
        max_tokens if max_tokens is not None else "-",
        elapsed_ms if elapsed_ms is not None else "-",
        status_code if status_code is not None else "-",
        type(error).__name__ if error is not None else "-",
        ",".join(_safe_identifier(code) for code in finding_codes)
        if finding_codes
        else "-",
        completion.finish_reason or "-",
        completion.input_tokens if completion.input_tokens is not None else "-",
        completion.output_tokens if completion.output_tokens is not None else "-",
        completion.reasoning_tokens if completion.reasoning_tokens is not None else "-",
        completion.total_tokens if completion.total_tokens is not None else "-",
    )


def log_trace_observation(
    *, event: str, name: str, trace_id: str | None, observation_id: str | None
) -> None:
    """Log only correlation IDs; observation creation is not delivery proof."""
    context = _CONTEXT.get() or DiagnosticContext()
    LOGGER.info(
        "toyapp_trace event=%s session_id=%s turn_id=%s trace_id=%s "
        "observation_id=%s name=%s",
        _safe_identifier(event),
        context.session_id,
        context.turn_id,
        _safe_identifier(trace_id),
        _safe_identifier(observation_id),
        _safe_identifier(name),
    )


def log_provider_failure(
    *,
    agent: str,
    model: str,
    attempt: int,
    max_tokens: int,
    started_at: float,
    error: BaseException,
    code: str | None = None,
    completion: CompletionDiagnostics | None = None,
) -> None:
    """Emit exactly one classified event for a failed live provider call."""
    classified_code, status_code = classify_external_error(error)
    log_failure(
        event="provider_call",
        agent=agent,
        operation="generate_action",
        code=code or classified_code,
        model=model,
        attempt=attempt,
        max_tokens=max_tokens,
        elapsed_ms=elapsed_milliseconds(started_at),
        status_code=status_code,
        error=error,
        completion=completion or getattr(error, "completion", None),
    )
