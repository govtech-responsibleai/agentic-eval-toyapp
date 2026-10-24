"""Safe classification and correlation of operational failures."""

import logging
import ssl
from types import SimpleNamespace
from uuid import uuid4

import pytest
from httpx2 import Request, Response
from openai import APIStatusError

from agentic_eval_toyapp.diagnostics import (
    CompletionDiagnostics,
    classify_external_error,
    diagnostic_context,
    log_failure,
)


def test_missing_or_malformed_provider_usage_remains_unknown() -> None:
    """Gateway extensions must not inject text or invalid counts into logs."""
    response = SimpleNamespace(
        choices=[SimpleNamespace(finish_reason="private\nforged-event")],
        usage=SimpleNamespace(
            prompt_tokens=-1,
            completion_tokens=True,
            total_tokens="99",
            completion_tokens_details=SimpleNamespace(reasoning_tokens=2.5),
        ),
    )

    assert CompletionDiagnostics.from_response(response) == CompletionDiagnostics(
        finish_reason="other"
    )
    assert CompletionDiagnostics.from_response(SimpleNamespace(choices=[])) == (
        CompletionDiagnostics()
    )


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, "authentication_error"),
        (403, "authentication_error"),
        (429, "rate_limited"),
        (503, "provider_server_error"),
        (400, "provider_http_error"),
    ],
)
def test_http_status_is_classified_without_response_body(
    status: int, expected: str
) -> None:
    """The model gateway's status is useful; its body may contain secrets."""
    response = Response(
        status,
        request=Request("POST", "https://provider.example/v1/chat/completions"),
    )
    error = APIStatusError("private response body", response=response, body=None)

    assert classify_external_error(error) == (expected, status)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (TimeoutError("private timeout detail"), "timeout"),
        (ssl.SSLError("private certificate detail"), "tls_error"),
        (PermissionError("private CA path"), "permission_error"),
        (ConnectionError("private endpoint"), "connection_error"),
        (RuntimeError("private upstream detail"), "unexpected_error"),
    ],
)
def test_non_http_failures_are_classified_by_type(
    error: Exception, expected: str
) -> None:
    """No exception text is needed to identify common infrastructure failures."""
    assert classify_external_error(error) == (expected, None)


def test_wrapped_tls_error_is_classified_without_logging_private_cause() -> None:
    """OpenAI may wrap the underlying certificate error."""
    error = RuntimeError("private gateway message")
    error.__cause__ = ssl.SSLError("private CA detail")

    assert classify_external_error(error) == ("tls_error", None)


def test_wrapped_provider_status_retains_safe_status_category() -> None:
    """A wrapper must not hide a rate-limit status from diagnostics."""
    response = Response(
        429,
        request=Request("POST", "https://provider.example/v1/chat/completions"),
    )
    wrapped = RuntimeError("private gateway response")
    wrapped.__cause__ = APIStatusError(
        "private response body", response=response, body=None
    )

    assert classify_external_error(wrapped) == ("rate_limited", 429)


def test_log_context_correlates_a_turn_then_resets_without_leaking_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Nested failures share IDs; a later request cannot inherit them."""
    session_id = uuid4()
    secret = "api-key-and-report-content-must-not-appear"
    with caplog.at_level(logging.ERROR, logger="uvicorn.error"):
        with diagnostic_context(session_id, turn=True) as context:
            log_failure(
                event="provider_call",
                agent="analysis",
                operation="generate_action",
                code="timeout",
                model="platform-model",
                attempt=2,
                max_tokens=1024,
                elapsed_ms=1200,
                error=RuntimeError(secret),
            )
        log_failure(
            event="startup",
            agent="runtime",
            operation="initialise_agent",
            code="initialisation_error",
        )

    first, second = caplog.records
    assert str(session_id) in first.message
    assert context.turn_id in first.message
    assert "agent=analysis" in first.message
    assert "attempt=2" in first.message
    assert "max_tokens=1024" in first.message
    assert "elapsed_ms=1200" in first.message
    assert "exception_type=RuntimeError" in first.message
    assert secret not in caplog.text
    assert "session_id=- turn_id=-" in second.message


def test_log_model_identifier_is_bounded_and_single_line(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Configuration values must not be able to inject another log event."""
    with caplog.at_level(logging.ERROR, logger="uvicorn.error"):
        log_failure(
            event="provider_call",
            agent="rag",
            operation="generate_action",
            code="invalid_schema",
            model="model\nforged-line" + "x" * 200,
        )

    message = caplog.records[0].message
    assert "\n" not in message
    assert "model=model_forged-line" in message
    assert len(message.split("model=", 1)[1].split(" ", 1)[0]) <= 80
