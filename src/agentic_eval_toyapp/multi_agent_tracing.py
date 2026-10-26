"""One coordinator-owned Langfuse trace for each submitted chat turn."""

from __future__ import annotations

from collections.abc import Generator, Mapping
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from typing import Any, Protocol, cast

from langfuse import Langfuse, propagate_attributes
from pydantic import BaseModel

from agentic_eval_toyapp.analysis_models import AnalysisModelRequest
from agentic_eval_toyapp.diagnostics import (
    CompletionDiagnostics,
    log_trace_observation,
)
from agentic_eval_toyapp.models import AgentTurnResult, ModelRequest, ValidationFinding
from agentic_eval_toyapp.orchestration_models import (
    CoordinatorAttachment,
    CoordinatorRequest,
)
from agentic_eval_toyapp.report_sources import SourceBlock
from agentic_eval_toyapp.tracing import (
    LangfuseTraceRecorder,
    LangfuseTurnTrace,
    ModelActionObservation,
    NoOpTraceRecorder,
    TurnTrace,
)


class ActionObservation(Protocol):
    """Record the effective structured action from one model invocation."""

    def succeed(self, action: BaseModel) -> None: ...

    def fail(self, code: str) -> None: ...


class AnalysisTrace(Protocol):
    """Observations owned by one selected report-analysis delegation."""

    def inspect_content(self, blocks: tuple[SourceBlock, ...]) -> None: ...

    def model_action(
        self, request: AnalysisModelRequest, *, model: str
    ) -> AbstractContextManager[ActionObservation]: ...

    def record_validation(
        self,
        *,
        reference_count: int,
        findings: tuple[ValidationFinding, ...],
    ) -> None: ...

    def complete(self, result: AgentTurnResult) -> None: ...


class CoordinatorTrace(Protocol):
    """Public coordinator steps under a single output-bearing turn root."""

    def inspect_input(self, attachments: tuple[CoordinatorAttachment, ...]) -> None: ...

    def model_action(
        self, request: CoordinatorRequest, *, model: str
    ) -> AbstractContextManager[ActionObservation]: ...

    def delegate_rag(self, task: str) -> AbstractContextManager[TurnTrace]: ...

    def delegate_analysis(
        self, task: str, attachments: tuple[CoordinatorAttachment, ...]
    ) -> AbstractContextManager[AnalysisTrace]: ...

    def return_response(self, result: AgentTurnResult) -> None: ...

    def complete(self, result: AgentTurnResult) -> None: ...


class CoordinatorTraceRecorder(Protocol):
    """Coordinator tracing dependency; disabled implementation is network-free."""

    @property
    def enabled(self) -> bool: ...

    def start_turn(
        self, *, session_id: str | None, user_message: str, history_message_count: int
    ) -> AbstractContextManager[CoordinatorTrace]: ...

    def shutdown(self) -> None: ...


class _NoOpActionObservation:
    def succeed(self, action: BaseModel) -> None:
        del action

    def fail(self, code: str) -> None:
        del code


class _NoOpAnalysisTrace:
    def inspect_content(self, blocks: tuple[SourceBlock, ...]) -> None:
        del blocks

    @contextmanager
    def model_action(
        self, request: AnalysisModelRequest, *, model: str
    ) -> Generator[ActionObservation]:
        del request, model
        yield _NoOpActionObservation()

    def record_validation(
        self, *, reference_count: int, findings: tuple[ValidationFinding, ...]
    ) -> None:
        del reference_count, findings

    def complete(self, result: AgentTurnResult) -> None:
        del result


class _NoOpCoordinatorTrace:
    def inspect_input(self, attachments: tuple[CoordinatorAttachment, ...]) -> None:
        del attachments

    @contextmanager
    def model_action(
        self, request: CoordinatorRequest, *, model: str
    ) -> Generator[ActionObservation]:
        del request, model
        yield _NoOpActionObservation()

    @contextmanager
    def delegate_rag(self, task: str) -> Generator[TurnTrace]:
        with NoOpTraceRecorder().start_turn(
            session_id=None, user_message=task, history_message_count=0
        ) as trace:
            yield trace

    @contextmanager
    def delegate_analysis(
        self, task: str, attachments: tuple[CoordinatorAttachment, ...]
    ) -> Generator[AnalysisTrace]:
        del task, attachments
        yield _NoOpAnalysisTrace()

    def return_response(self, result: AgentTurnResult) -> None:
        del result

    def complete(self, result: AgentTurnResult) -> None:
        del result


class NoOpCoordinatorTraceRecorder:
    @property
    def enabled(self) -> bool:
        return False

    @contextmanager
    def start_turn(
        self, *, session_id: str | None, user_message: str, history_message_count: int
    ) -> Generator[CoordinatorTrace]:
        del session_id, user_message, history_message_count
        yield _NoOpCoordinatorTrace()

    def shutdown(self) -> None:
        return None


_ACTIVE_GENERATION: ContextVar[Any | None] = ContextVar(
    "toyapp_active_coordinator_generation", default=None
)
_ACTIVE_RAG_TRACE: ContextVar[TurnTrace | None] = ContextVar(
    "toyapp_active_delegated_rag_trace", default=None
)
_ACTIVE_ANALYSIS_TRACE: ContextVar[AnalysisTrace | None] = ContextVar(
    "toyapp_active_analysis_trace", default=None
)


def delegated_rag_trace() -> TurnTrace | None:
    """Return the active nested trace without creating another root."""
    return _ACTIVE_RAG_TRACE.get()


def analysis_trace() -> AnalysisTrace:
    """Return the active nested report trace for the current turn."""
    return _ACTIVE_ANALYSIS_TRACE.get() or _NoOpAnalysisTrace()


def record_completion_usage(diagnostics: CompletionDiagnostics) -> None:
    """Attach provider-reported counts to an app-owned generation, if active."""
    observation = _ACTIVE_GENERATION.get()
    if observation is None:
        return
    usage = {
        key: value
        for key, value in (
            ("input", diagnostics.input_tokens),
            ("output", diagnostics.output_tokens),
            ("total", diagnostics.total_tokens),
        )
        if value is not None
    }
    metadata: dict[str, object] = {}
    if diagnostics.finish_reason is not None:
        metadata["finish_reason"] = diagnostics.finish_reason
    if diagnostics.reasoning_tokens is not None:
        metadata["reasoning_tokens"] = diagnostics.reasoning_tokens
    if usage or metadata:
        values: dict[str, object] = {"metadata": metadata} if metadata else {}
        if usage:
            values["usage_details"] = usage
        observation.update(**values)


class _LangfuseActionObservation:
    def __init__(self, observation: Any) -> None:
        self._observation = observation

    def succeed(self, action: BaseModel) -> None:
        self._observation.update(output=action.model_dump(mode="json"))

    def fail(self, code: str) -> None:
        self._observation.update(
            output={"error_code": code}, level="ERROR", status_message=code
        )


@contextmanager
def _generation(
    client: Langfuse, *, name: str, model: str, input_data: object
) -> Generator[ActionObservation]:
    with client.start_as_current_observation(
        name=name, as_type="generation", model=model, input=input_data
    ) as observation:
        token = _ACTIVE_GENERATION.set(observation)
        try:
            yield _LangfuseActionObservation(observation)
        finally:
            _ACTIVE_GENERATION.reset(token)


def _attachment_metadata(attachment: CoordinatorAttachment) -> dict[str, object]:
    return {
        "file_id": str(attachment.file_id),
        "filename": attachment.filename,
        "media_type": attachment.media_type.value,
        "byte_size": attachment.byte_size,
        "extracted_characters": len(attachment.extracted_text),
    }


class _LangfuseDelegatedRagTrace(LangfuseTurnTrace):
    def __init__(
        self, *, client: Langfuse, root_observation: Any, model_id: str
    ) -> None:
        super().__init__(client=client, root_observation=root_observation)
        self._model_id = model_id

    @contextmanager
    def model_action(self, request: ModelRequest) -> Generator[ModelActionObservation]:
        with _generation(
            self._client,
            name="model-action",
            model=self._model_id,
            input_data={
                "user_message": request.user_message,
                "history": [item.model_dump(mode="json") for item in request.history],
                "retrieved_chunks": [
                    chunk.model_dump(mode="json") for chunk in request.retrieved_chunks
                ],
                "available_actions": [
                    action.value for action in request.available_actions
                ],
                "model_call_number": request.model_call_number,
            },
        ) as observation:
            yield cast(ModelActionObservation, observation)


class _LangfuseAnalysisTrace:
    def __init__(
        self, *, client: Langfuse, observation: Any, include_report_text: bool
    ) -> None:
        self._client = client
        self._observation = observation
        self._include_report_text = include_report_text

    def inspect_content(self, blocks: tuple[SourceBlock, ...]) -> None:
        with self._client.start_as_current_observation(
            name="inspect-content",
            as_type="span",
            input={"source_count": len(blocks)},
        ) as observation:
            observation.update(
                output={
                    "sources": [
                        {
                            "file_id": str(block.file_id),
                            "locator": block.locator.model_dump(mode="json"),
                            "characters": len(block.text),
                        }
                        for block in blocks
                    ]
                }
            )

    @contextmanager
    def model_action(
        self, request: AnalysisModelRequest, *, model: str
    ) -> Generator[ActionObservation]:
        sources: list[dict[str, object]] = []
        for block in request.blocks:
            source: dict[str, object] = {
                "file_id": str(block.file_id),
                "locator": block.locator.model_dump(mode="json"),
                "characters": len(block.text),
            }
            if self._include_report_text:
                source["text"] = block.text
            sources.append(source)
        with _generation(
            self._client,
            name="analyse-report",
            model=model,
            input_data={
                "task": request.task,
                "history": [item.model_dump(mode="json") for item in request.history],
                "sources": sources,
                "report_text_exported": self._include_report_text,
            },
        ) as observation:
            yield observation

    def record_validation(
        self, *, reference_count: int, findings: tuple[ValidationFinding, ...]
    ) -> None:
        with self._client.start_as_current_observation(
            name="validate-report-references",
            as_type="evaluator",
            input={"declared_reference_count": reference_count},
        ) as observation:
            observation.update(
                output=[finding.model_dump(mode="json") for finding in findings],
                metadata={"finding_count": len(findings)},
            )

    def complete(self, result: AgentTurnResult) -> None:
        self._observation.update(
            output=result.message,
            metadata={
                "status": result.status.value,
                "termination_reason": result.termination_reason.value,
                "report_reference_count": len(result.report_references),
            },
        )


class _LangfuseCoordinatorTrace:
    def __init__(
        self,
        *,
        client: Langfuse,
        root: Any,
        include_report_text: bool,
        rag_model_id: str,
    ) -> None:
        self._client = client
        self._root = root
        self._include_report_text = include_report_text
        self._rag_model_id = rag_model_id

    def inspect_input(self, attachments: tuple[CoordinatorAttachment, ...]) -> None:
        with self._client.start_as_current_observation(
            name="inspect-input",
            as_type="span",
            input={"attachment_count": len(attachments)},
        ) as observation:
            log_trace_observation(
                event="started",
                name="inspect-input",
                trace_id=self._client.get_current_trace_id(),
                observation_id=self._client.get_current_observation_id(),
            )
            observation.update(
                output={"attachments": [_attachment_metadata(a) for a in attachments]}
            )

    @contextmanager
    def model_action(
        self, request: CoordinatorRequest, *, model: str
    ) -> Generator[ActionObservation]:
        attachments: list[dict[str, object]] = []
        for attachment in request.attachments:
            entry = _attachment_metadata(attachment)
            if self._include_report_text:
                entry["extracted_text"] = attachment.extracted_text
            attachments.append(entry)
        with _generation(
            self._client,
            name="coordinator-action",
            model=model,
            input_data={
                "user_message": request.user_message,
                "history": [item.model_dump(mode="json") for item in request.history],
                "attachments": attachments,
                "available_actions": [
                    action.value for action in request.available_actions
                ],
                "model_call_number": request.model_call_number,
                "report_text_exported": self._include_report_text,
            },
        ) as observation:
            yield observation

    @contextmanager
    def delegate_rag(self, task: str) -> Generator[TurnTrace]:
        with self._client.start_as_current_observation(
            name="delegate-rag", as_type="agent", input={"task": task}
        ) as observation:
            trace = _LangfuseDelegatedRagTrace(
                client=self._client,
                root_observation=observation,
                model_id=self._rag_model_id,
            )
            token = _ACTIVE_RAG_TRACE.set(trace)
            try:
                yield trace
            finally:
                _ACTIVE_RAG_TRACE.reset(token)

    @contextmanager
    def delegate_analysis(
        self, task: str, attachments: tuple[CoordinatorAttachment, ...]
    ) -> Generator[AnalysisTrace]:
        with self._client.start_as_current_observation(
            name="delegate-analysis",
            as_type="agent",
            input={
                "task": task,
                "attachments": [_attachment_metadata(a) for a in attachments],
            },
        ) as observation:
            trace = _LangfuseAnalysisTrace(
                client=self._client,
                observation=observation,
                include_report_text=self._include_report_text,
            )
            token = _ACTIVE_ANALYSIS_TRACE.set(trace)
            try:
                yield trace
            finally:
                _ACTIVE_ANALYSIS_TRACE.reset(token)

    def return_response(self, result: AgentTurnResult) -> None:
        with self._client.start_as_current_observation(
            name="return-response",
            as_type="span",
            input={"specialist_status": result.status.value},
        ) as observation:
            observation.update(output=result.message)

    def complete(self, result: AgentTurnResult) -> None:
        self._root.update(
            output=result.message,
            metadata={
                "status": result.status.value,
                "termination_reason": result.termination_reason.value,
                "model_calls": result.model_calls,
                "searches": result.searches,
                "execution_steps": [
                    step.model_dump(mode="json") for step in result.execution_steps
                ],
                "validation_findings": [
                    finding.model_dump(mode="json")
                    for finding in result.validation_findings
                ],
            },
        )


class LangfuseCoordinatorTraceRecorder:
    """Emit one root per submitted coordinator turn and nested specialist work."""

    def __init__(
        self,
        *,
        include_report_text: bool = False,
        client: Langfuse | None = None,
        public_key: str,
        secret_key: str,
        base_url: str,
        environment: str,
        release: str,
        base_metadata: Mapping[str, str],
        rag_model_id: str,
        flush_on_turn: bool = True,
    ) -> None:
        self._recorder = LangfuseTraceRecorder(
            public_key=public_key,
            secret_key=secret_key,
            base_url=base_url,
            environment=environment,
            release=release,
            base_metadata=base_metadata,
            flush_on_turn=flush_on_turn,
            client=client,
        )
        self._include_report_text = include_report_text
        self._rag_model_id = rag_model_id
        self._base_metadata = dict(base_metadata)
        self._release = release
        self._flush_on_turn = flush_on_turn

    @property
    def enabled(self) -> bool:
        return True

    @property
    def client(self) -> Langfuse:
        """Expose the configured SDK for opt-in v4 retrieval checks."""
        return self._recorder.client

    def shutdown(self) -> None:
        self._recorder.shutdown()

    @contextmanager
    def start_turn(
        self, *, session_id: str | None, user_message: str, history_message_count: int
    ) -> Generator[CoordinatorTrace]:
        with (
            propagate_attributes(
                session_id=session_id or "unscoped-direct-call",
                trace_name="agent-turn",
                metadata=self._base_metadata,
                version=self._release,
                tags=["agentic-eval-toyapp", "coordinator"],
            ),
            self._recorder.client.start_as_current_observation(
                name="agent-turn",
                as_type="agent",
                input=user_message,
                metadata={
                    "history_message_count": history_message_count,
                    "report_text_exported": self._include_report_text,
                },
            ) as root,
        ):
            trace_id = self._recorder.client.get_current_trace_id()
            root_id = self._recorder.client.get_current_observation_id()
            log_trace_observation(
                event="started",
                name="agent-turn",
                trace_id=trace_id,
                observation_id=root_id,
            )
            trace = _LangfuseCoordinatorTrace(
                client=self._recorder.client,
                root=root,
                include_report_text=self._include_report_text,
                rag_model_id=self._rag_model_id,
            )
            try:
                yield trace
            except Exception:
                root.update(level="ERROR", status_message="unhandled_agent_error")
                raise
        if self._flush_on_turn:
            self._recorder.flush()
            log_trace_observation(
                event="flush_attempted",
                name="agent-turn",
                trace_id=trace_id,
                observation_id=root_id,
            )
