"""Langfuse-compatible tracing boundary for one synchronous agent turn."""

from __future__ import annotations

import base64
import importlib.metadata
import logging
from collections.abc import Generator, Mapping
from contextlib import AbstractContextManager, contextmanager
from time import perf_counter
from typing import Any, Protocol

from langfuse import Langfuse, propagate_attributes
from opentelemetry.exporter.http.transport._base import (
    BaseHTTPResult,
    BaseHTTPTransport,
)
from opentelemetry.exporter.http.transport._urllib3 import Urllib3HTTPTransport
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SpanExporter

from agentic_eval_toyapp.diagnostics import log_failure
from agentic_eval_toyapp.models import (
    AgentAction,
    AgentTurnResult,
    ModelRequest,
    RetrievedChunk,
    ValidationFinding,
)

# Loggers that explain a failed export: the Langfuse SDK, the OpenTelemetry
# exporter that reports "Failed to export spans batch", and urllib3, which logs
# each request line and response status/length.
_EXPORT_DEBUG_LOGGERS = (
    "agentic_eval_toyapp.langfuse_export",
    "langfuse",
    "opentelemetry.exporter",
    "opentelemetry.sdk.trace.export",
    "urllib3.connectionpool",
)
_EXPORT_DEBUG_HANDLER_NAME = "toyapp-langfuse-export-debug"


def enable_langfuse_export_debug_logging() -> None:
    """Route trace-export diagnostics to stderr at DEBUG, once per process."""
    handler = logging.StreamHandler()
    handler.set_name(_EXPORT_DEBUG_HANDLER_NAME)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
    )
    for name in _EXPORT_DEBUG_LOGGERS:
        logger = logging.getLogger(name)
        if any(h.get_name() == _EXPORT_DEBUG_HANDLER_NAME for h in logger.handlers):
            continue
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)


_RESPONSE_LOGGER = logging.getLogger("agentic_eval_toyapp.langfuse_export")
_MAX_LOGGED_BODY_CHARS = 2_000
_REDACTED_RESPONSE_HEADERS = frozenset({"set-cookie"})


class _ResponseLoggingTransport(BaseHTTPTransport):
    """Delegate to the stock transport, logging every rejected response.

    The OTLP exporter reduces a rejected export to "code: 500, reason: ...".
    This keeps the response headers and body, which usually carry the
    reverse-proxy or Langfuse error detail a server admin needs.
    """

    def __init__(self, inner: BaseHTTPTransport) -> None:
        self._inner = inner

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
        data: bytes | None = None,
    ) -> BaseHTTPResult:
        started = perf_counter()
        result = self._inner.request(
            method, url, headers=headers, timeout=timeout, data=data
        )
        elapsed_ms = round((perf_counter() - started) * 1_000)
        if result.error is not None:
            _RESPONSE_LOGGER.error(
                "langfuse_export_failed method=%s url=%s request_bytes=%s "
                "elapsed_ms=%s exception_type=%s exception=%s",
                method,
                url,
                len(data) if data is not None else 0,
                elapsed_ms,
                type(result.error).__name__,
                result.error,
            )
        elif result.status_code is not None and result.status_code >= 400:
            _RESPONSE_LOGGER.error(
                "langfuse_export_rejected status=%s reason=%s method=%s url=%s "
                "request_bytes=%s elapsed_ms=%s response_headers=%r "
                "response_body=%r",
                result.status_code,
                result.reason,
                method,
                url,
                len(data) if data is not None else 0,
                elapsed_ms,
                {
                    name: value
                    for name, value in result.headers().items()
                    if name.lower() not in _REDACTED_RESPONSE_HEADERS
                },
                result.content()[:_MAX_LOGGED_BODY_CHARS].decode(
                    "utf-8", errors="replace"
                ),
            )
        return result

    def close(self) -> None:
        self._inner.close()

    def is_connection_error(self, exception: Exception | None) -> bool:
        return self._inner.is_connection_error(exception)


def _build_debug_span_exporter(
    *, public_key: str, secret_key: str, base_url: str
) -> SpanExporter:
    """Mirror the SDK's default OTLP exporter, adding failed-response logging."""
    token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode("ascii")
    return OTLPSpanExporter(
        endpoint=f"{base_url}/api/public/otel/v1/traces",
        headers={
            "Authorization": f"Basic {token}",
            "x-langfuse-sdk-name": "python",
            "x-langfuse-sdk-version": importlib.metadata.version("langfuse"),
            "x-langfuse-public-key": public_key,
        },
        _transport=_ResponseLoggingTransport(Urllib3HTTPTransport()),
    )


class ModelActionObservation(Protocol):
    """Mutable result boundary for one provider action-selection call."""

    def succeed(self, action: AgentAction) -> None:
        """Record the provider's validated structured action."""
        ...

    def fail(self, error_code: str) -> None:
        """Record a stable, non-sensitive failure category."""
        ...


class TurnTrace(Protocol):
    """Observations emitted while one agent turn is active."""

    def model_action(
        self, request: ModelRequest
    ) -> AbstractContextManager[ModelActionObservation]:
        """Create one observation around an action-selection call."""
        ...

    def record_search(
        self,
        *,
        query: str,
        top_k: int,
        results: tuple[RetrievedChunk, ...] | None,
        error_code: str | None = None,
    ) -> None:
        """Record one executed knowledge-base search."""
        ...

    def record_clarification(self, question: str) -> None:
        """Record a user-input request using METIS's recognised tool name."""
        ...

    def record_validation(
        self,
        *,
        citations: tuple[str, ...],
        candidate_ids: tuple[str, ...],
        findings: tuple[ValidationFinding, ...],
    ) -> None:
        """Record deterministic citation consistency findings."""
        ...

    def complete(self, result: AgentTurnResult) -> None:
        """Set the public result used as the root trace output."""
        ...


class TraceRecorder(Protocol):
    """Process-level tracing dependency used by the agent."""

    @property
    def enabled(self) -> bool:
        """Return whether observations are exported."""
        ...

    def start_turn(
        self,
        *,
        session_id: str | None,
        user_message: str,
        history_message_count: int,
    ) -> AbstractContextManager[TurnTrace]:
        """Create the root observation for one user turn."""
        ...

    def flush(self) -> None:
        """Send pending observations to the configured backend."""
        ...

    def shutdown(self) -> None:
        """Flush and release process-level tracing resources."""
        ...


class _NoOpModelActionObservation:
    def succeed(self, action: AgentAction) -> None:
        del action

    def fail(self, error_code: str) -> None:
        del error_code


class _NoOpTurnTrace:
    @contextmanager
    def model_action(self, request: ModelRequest) -> Generator[ModelActionObservation]:
        del request
        yield _NoOpModelActionObservation()

    def record_search(
        self,
        *,
        query: str,
        top_k: int,
        results: tuple[RetrievedChunk, ...] | None,
        error_code: str | None = None,
    ) -> None:
        del query, top_k, results, error_code

    def record_clarification(self, question: str) -> None:
        del question

    def record_validation(
        self,
        *,
        citations: tuple[str, ...],
        candidate_ids: tuple[str, ...],
        findings: tuple[ValidationFinding, ...],
    ) -> None:
        del citations, candidate_ids, findings

    def complete(self, result: AgentTurnResult) -> None:
        del result


class NoOpTraceRecorder:
    """Zero-export implementation used by offline tests and local opt-out."""

    @property
    def enabled(self) -> bool:
        return False

    @contextmanager
    def start_turn(
        self,
        *,
        session_id: str | None,
        user_message: str,
        history_message_count: int,
    ) -> Generator[TurnTrace]:
        del session_id, user_message, history_message_count
        yield _NoOpTurnTrace()

    def flush(self) -> None:
        return None

    def shutdown(self) -> None:
        return None


class _LangfuseModelActionObservation:
    def __init__(self, observation: Any) -> None:
        self._observation = observation

    def succeed(self, action: AgentAction) -> None:
        self._observation.update(output=action.model_dump(mode="json"))

    def fail(self, error_code: str) -> None:
        self._observation.update(
            output={"error_code": error_code},
            level="ERROR",
            status_message=error_code,
        )


class LangfuseTurnTrace:
    def __init__(self, *, client: Langfuse, root_observation: Any) -> None:
        self._client = client
        self._root_observation = root_observation
        self._result: AgentTurnResult | None = None

    @contextmanager
    def model_action(self, request: ModelRequest) -> Generator[ModelActionObservation]:
        request_input = {
            "user_message": request.user_message,
            "history_message_count": len(request.history),
            "retrieval_attempted": request.retrieval_attempted,
            "candidate_chunk_ids": [
                chunk.chunk_id for chunk in request.retrieved_chunks
            ],
            "available_actions": [action.value for action in request.available_actions],
            "model_call_number": request.model_call_number,
            "remaining_searches": request.remaining_searches,
        }
        with self._client.start_as_current_observation(
            name="select-action",
            as_type="span",
            input=request_input,
        ) as observation:
            yield _LangfuseModelActionObservation(observation)

    def record_search(
        self,
        *,
        query: str,
        top_k: int,
        results: tuple[RetrievedChunk, ...] | None,
        error_code: str | None = None,
    ) -> None:
        output: object
        if results is None:
            output = {"error_code": error_code or "retrieval_error"}
        else:
            output = [result.model_dump(mode="json") for result in results]
        with self._client.start_as_current_observation(
            name="search_knowledge_base",
            as_type="tool",
            input={"query": query, "top_k": top_k},
        ) as observation:
            observation.update(
                output=output,
                level="ERROR" if error_code else "DEFAULT",
                status_message=error_code,
                metadata={
                    "result_count": 0 if results is None else len(results),
                },
            )

    def record_clarification(self, question: str) -> None:
        with self._client.start_as_current_observation(
            name="request_user_input",
            as_type="tool",
            input={"question": question},
        ) as observation:
            observation.update(output={"status": "requested"})

    def record_validation(
        self,
        *,
        citations: tuple[str, ...],
        candidate_ids: tuple[str, ...],
        findings: tuple[ValidationFinding, ...],
    ) -> None:
        with self._client.start_as_current_observation(
            name="validate_answer",
            as_type="evaluator",
            input={
                "citations": list(citations),
                "candidate_chunk_ids": list(candidate_ids),
            },
        ) as observation:
            observation.update(
                output=[finding.model_dump(mode="json") for finding in findings],
                metadata={"finding_count": len(findings)},
            )

    def complete(self, result: AgentTurnResult) -> None:
        self._result = result
        self._root_observation.update(
            output=result.message,
            metadata={
                "status": result.status.value,
                "termination_reason": result.termination_reason.value,
                "model_calls": result.model_calls,
                "searches": result.searches,
                "citations": list(result.citations),
                "candidate_chunk_ids": [
                    chunk.chunk_id for chunk in result.retrieved_chunks
                ],
                "controller_events": [
                    event.model_dump(mode="json") for event in result.controller_events
                ],
                "validation_findings": [
                    finding.model_dump(mode="json")
                    for finding in result.validation_findings
                ],
            },
        )


class LangfuseTraceRecorder:
    """Export output-bearing agent-turn traces grouped by app session UUID."""

    def __init__(
        self,
        *,
        public_key: str,
        secret_key: str,
        base_url: str,
        environment: str,
        release: str,
        base_metadata: Mapping[str, str],
        flush_on_turn: bool = True,
        debug: bool = False,
        client: Langfuse | None = None,
    ) -> None:
        if debug:
            enable_langfuse_export_debug_logging()
        self._client = client or Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            base_url=base_url,
            environment=environment,
            release=release,
            debug=debug,
            span_exporter=_build_debug_span_exporter(
                public_key=public_key, secret_key=secret_key, base_url=base_url
            )
            if debug
            else None,
        )
        self._release = release
        self._base_metadata = dict(base_metadata)
        self._flush_on_turn = flush_on_turn

    @property
    def enabled(self) -> bool:
        return True

    @property
    def client(self) -> Langfuse:
        """Expose the configured client for live compatibility verification."""
        return self._client

    @contextmanager
    def start_turn(
        self,
        *,
        session_id: str | None,
        user_message: str,
        history_message_count: int,
    ) -> Generator[TurnTrace]:
        trace_session_id = session_id or "unscoped-direct-call"
        with (
            propagate_attributes(
                session_id=trace_session_id,
                trace_name="agent-turn",
                metadata=self._base_metadata,
                version=self._release,
                tags=["agentic-eval-toyapp", "rag"],
            ),
            self._client.start_as_current_observation(
                name="agent-turn",
                as_type="agent",
                input=user_message,
                metadata={"history_message_count": history_message_count},
            ) as root,
        ):
            turn = LangfuseTurnTrace(
                client=self._client,
                root_observation=root,
            )
            try:
                yield turn
            except Exception:
                root.update(
                    level="ERROR",
                    status_message="unhandled_agent_error",
                )
                raise
        if self._flush_on_turn:
            self.flush()

    def flush(self) -> None:
        try:
            self._client.flush()
        except Exception as error:
            log_failure(
                event="trace_export",
                agent="runtime",
                operation="flush",
                code="langfuse_flush_error",
                error=error,
            )
            raise RuntimeError("Langfuse trace flush failed") from None

    def shutdown(self) -> None:
        try:
            self._client.shutdown()
        except Exception as error:
            log_failure(
                event="trace_export",
                agent="runtime",
                operation="shutdown",
                code="langfuse_shutdown_error",
                error=error,
            )
            raise RuntimeError("Langfuse trace shutdown failed") from None
