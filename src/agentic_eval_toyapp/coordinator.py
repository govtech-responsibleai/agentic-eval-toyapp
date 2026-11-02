"""Bounded coordinator routing for direct and specialist-backed turns."""

import json
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol, cast, runtime_checkable
from uuid import UUID

from pydantic import (
    ConfigDict,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from agentic_eval_toyapp.diagnostics import log_failure, log_provider_failure
from agentic_eval_toyapp.models import (
    AgentTurnResult,
    AgentTurnStatus,
    AnswerAction,
    ClarifyAction,
    ControllerEvent,
    ControllerEventKind,
    ConversationMessage,
    ExecutionAction,
    ExecutionAgent,
    ExecutionStatus,
    ExecutionStep,
    NonEmptyString,
    TerminationReason,
)
from agentic_eval_toyapp.multi_agent_tracing import (
    CoordinatorTrace,
    CoordinatorTraceRecorder,
    NoOpCoordinatorTraceRecorder,
)
from agentic_eval_toyapp.orchestration_models import (
    AnalysisHandoffRequest,
    CoordinatorAction,
    CoordinatorActionKind,
    CoordinatorAnswerAction,
    CoordinatorAttachment,
    CoordinatorAttachmentPage,
    CoordinatorClarifyAction,
    CoordinatorRequest,
    CoordinatorRouteAnalysisAction,
    CoordinatorRouteRagAction,
    RagHandoffRequest,
)
from agentic_eval_toyapp.providers import (
    GenerationProviderError,
    InvalidGenerationActionError,
    OpenAICompatibleChatClient,
    StructuredCompletionClient,
    make_object_schemas_strict,
    unpack_completion,
)
from agentic_eval_toyapp.reports import StoredReport
from agentic_eval_toyapp.structured_models import NullableOmittedAsNone

DEFAULT_COORDINATOR_PROMPT_PATH = Path(__file__).parent / "prompts" / "coordinator.md"
_COORDINATOR_ACTION_ADAPTER: TypeAdapter[CoordinatorAction] = TypeAdapter(
    CoordinatorAction
)


class CoordinatorPromptLoadError(ValueError):
    """Raised when the coordinator prompt is unavailable or blank."""


def load_coordinator_prompt(
    path: Path = DEFAULT_COORDINATOR_PROMPT_PATH,
) -> str:
    """Read the version-controlled coordinator instructions."""
    try:
        prompt = path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise CoordinatorPromptLoadError(
            f"could not read coordinator prompt: {path}"
        ) from error
    if not prompt:
        raise CoordinatorPromptLoadError(f"coordinator prompt is blank: {path}")
    return prompt


@runtime_checkable
class CoordinatorGenerationProvider(Protocol):
    """Select one validated coordinator decision from public turn context."""

    @property
    def model_id(self) -> str:
        """Return the model or fixture identifier."""
        ...

    @property
    def revision(self) -> str:
        """Return the model or fixture revision."""
        ...

    async def generate_action(self, request: CoordinatorRequest) -> CoordinatorAction:
        """Return exactly one coordinator action."""
        ...


class RagSpecialist(Protocol):
    """Future RAG delegation boundary implemented in Stage 13."""

    async def run(
        self,
        request: RagHandoffRequest,
        *,
        session_id: str | None = None,
    ) -> AgentTurnResult:
        """Run one focused policy task."""
        ...


class AnalysisSpecialist(Protocol):
    """Report-analysis boundary receiving coordinator-selected content."""

    async def run(
        self,
        request: AnalysisHandoffRequest,
        *,
        session_id: str | None = None,
    ) -> AgentTurnResult:
        """Run one focused analysis task over authorized content."""
        ...


@runtime_checkable
class ShutdownableSpecialist(Protocol):
    """Optional lifecycle hook for specialist-owned resources."""

    def shutdown(self) -> None:
        """Release process-level specialist resources."""
        ...


class _StructuredCoordinatorAction(NullableOmittedAsNone):
    """Gateway-friendly flat form of the coordinator decision union."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: CoordinatorActionKind
    markdown: NonEmptyString | None
    question: NonEmptyString | None
    task: NonEmptyString | None
    file_ids: tuple[UUID, ...]

    @model_validator(mode="after")
    def validate_action_fields(self) -> "_StructuredCoordinatorAction":
        """Require only the fields belonging to the selected decision."""
        if self.kind is CoordinatorActionKind.ANSWER:
            if self.markdown is None or any(
                value is not None for value in (self.question, self.task)
            ):
                raise ValueError("answer requires only markdown")
            if self.file_ids:
                raise ValueError("answer cannot select files")
        elif self.kind is CoordinatorActionKind.CLARIFY:
            if self.question is None or any(
                value is not None for value in (self.markdown, self.task)
            ):
                raise ValueError("clarify requires only question")
            if self.file_ids:
                raise ValueError("clarify cannot select files")
        elif self.kind is CoordinatorActionKind.ROUTE_RAG:
            if self.task is None or any(
                value is not None for value in (self.markdown, self.question)
            ):
                raise ValueError("route_rag requires only task")
            if self.file_ids:
                raise ValueError("route_rag cannot select report evidence")
        else:
            if self.task is None or any(
                value is not None for value in (self.markdown, self.question)
            ):
                raise ValueError("route_analysis requires only task and file_ids")
            if not self.file_ids or len(self.file_ids) != len(set(self.file_ids)):
                raise ValueError("route_analysis requires unique file_ids")
        return self

    def to_domain_action(self) -> CoordinatorAction:
        """Convert the transport form into the discriminated domain union."""
        payload = self.model_dump(mode="json", exclude_none=True)
        if self.kind is not CoordinatorActionKind.ROUTE_ANALYSIS:
            payload.pop("file_ids", None)
        return _COORDINATOR_ACTION_ADAPTER.validate_python(payload)


def _coordinator_action_schema() -> dict[str, Any]:
    schema = _StructuredCoordinatorAction.model_json_schema()
    make_object_schemas_strict(schema)
    return schema


class OpenAICoordinatorGenerationProvider:
    """Select coordinator decisions through an OpenAI-compatible endpoint."""

    def __init__(
        self,
        *,
        model: str,
        revision: str,
        temperature: float,
        max_tokens: int,
        client: StructuredCompletionClient,
    ) -> None:
        if not model.strip() or not revision.strip():
            raise ValueError("model and revision must not be blank")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        self._model_id = model.strip()
        self._revision = revision.strip()
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = client
        self._response_schema = _coordinator_action_schema()

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def revision(self) -> str:
        return self._revision

    async def generate_action(self, request: CoordinatorRequest) -> CoordinatorAction:
        """Request and validate one structured coordinator decision."""
        response_schema = deepcopy(self._response_schema)
        definitions = cast(dict[str, Any], response_schema["$defs"])
        action_kind = cast(dict[str, Any], definitions["CoordinatorActionKind"])
        action_kind["enum"] = [action.value for action in request.available_actions]
        context = {
            "task": "Choose exactly one next action for this user turn.",
            "available_actions": [action.value for action in request.available_actions],
            "response_field_rules": {
                "answer": (
                    "Set markdown only; set question and task to null and "
                    "file_ids to []."
                ),
                "clarify": (
                    "Set question only; set markdown and task to null and "
                    "file_ids to []."
                ),
                "route_rag": (
                    "Set task only; set markdown and question to null and "
                    "file_ids to []."
                ),
                "route_analysis": (
                    "Set task and attached file_ids only; set markdown and "
                    "question to null."
                ),
            },
            "action_feedback": request.action_feedback,
            "user_message": request.user_message,
            "conversation_history": [
                message.model_dump(mode="json") for message in request.history
            ],
            "attachments": [
                attachment.model_dump(mode="json") for attachment in request.attachments
            ],
            "model_call_number": request.model_call_number,
        }
        messages = (
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
        )
        started_at = perf_counter()
        try:
            completion = await self._client.create(
                model=self._model_id,
                messages=messages,
                response_schema=response_schema,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            content, diagnostics = unpack_completion(completion)
        except GenerationProviderError as error:
            log_provider_failure(
                agent="coordinator",
                model=self._model_id,
                attempt=request.model_call_number,
                max_tokens=self._max_tokens,
                started_at=started_at,
                error=error,
                code=error.code,
            )
            raise
        except Exception as error:
            log_provider_failure(
                agent="coordinator",
                model=self._model_id,
                attempt=request.model_call_number,
                max_tokens=self._max_tokens,
                started_at=started_at,
                error=error,
            )
            raise GenerationProviderError("live coordinator request failed") from error
        try:
            structured_action = _StructuredCoordinatorAction.model_validate_json(
                content
            )
            action = structured_action.to_domain_action()
        except ValidationError as error:
            log_provider_failure(
                agent="coordinator",
                model=self._model_id,
                attempt=request.model_call_number,
                max_tokens=self._max_tokens,
                started_at=started_at,
                error=error,
                code="invalid_schema",
                completion=diagnostics,
            )
            raise InvalidGenerationActionError(
                "live model returned an invalid coordinator action"
            ) from error
        if structured_action.kind not in request.available_actions:
            error = InvalidGenerationActionError(
                "live model returned an unavailable coordinator action"
            )
            log_provider_failure(
                agent="coordinator",
                model=self._model_id,
                attempt=request.model_call_number,
                max_tokens=self._max_tokens,
                started_at=started_at,
                error=error,
                code="unavailable_action",
                completion=diagnostics,
            )
            raise error
        return action


class ScriptedCoordinatorGenerationProvider:
    """Return a fixed coordinator sequence for deterministic offline tests."""

    def __init__(
        self,
        actions: Sequence[CoordinatorAction],
        *,
        model_id: str = "scripted-coordinator-fixture",
        revision: str = "v1",
    ) -> None:
        if not actions:
            raise ValueError("at least one scripted action is required")
        self._actions = tuple(actions)
        self._model_id = model_id.strip()
        self._revision = revision.strip()
        if not self._model_id or not self._revision:
            raise ValueError("model_id and revision must not be blank")
        self._position = 0
        self._requests: list[CoordinatorRequest] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def revision(self) -> str:
        return self._revision

    @property
    def requests(self) -> tuple[CoordinatorRequest, ...]:
        """Return immutable request history for assertions."""
        return tuple(self._requests)

    async def generate_action(self, request: CoordinatorRequest) -> CoordinatorAction:
        self._requests.append(request)
        if self._position >= len(self._actions):
            raise GenerationProviderError("scripted action sequence is exhausted")
        action = self._actions[self._position]
        self._position += 1
        return action


class DeterministicCoordinatorGenerationProvider:
    """Provide reproducible routing for offline development and CI."""

    _ANALYSIS_TERMS = frozenset(
        {
            "analyse",
            "analyze",
            "analysis",
            "summary",
            "summarise",
            "summarize",
            "findings",
            "insights",
            "trend",
        }
    )
    _POLICY_TERMS = frozenset(
        {
            "policy",
            "leave",
            "remote",
            "expense",
            "claim",
            "data",
            "security",
            "procurement",
            "purchase",
            "learning",
            "benefit",
            "conduct",
            "travel",
            "overtime",
        }
    )

    @property
    def model_id(self) -> str:
        return "deterministic-coordinator-provider"

    @property
    def revision(self) -> str:
        return "v1"

    async def generate_action(self, request: CoordinatorRequest) -> CoordinatorAction:
        """Route using transparent fixtures rather than stochastic inference."""
        message = request.user_message.casefold()
        history = " ".join(item.content.casefold() for item in request.history)
        report_text = " ".join(
            attachment.extracted_text.casefold() for attachment in request.attachments
        )
        message_terms = set(message.replace("?", " ").replace(".", " ").split())
        combined_terms = set(
            f"{message} {history} {report_text}".replace("?", " ")
            .replace(".", " ")
            .split()
        )

        if message_terms & self._ANALYSIS_TERMS:
            if not request.attachments:
                return CoordinatorClarifyAction(
                    question="Please upload the report you want me to analyse."
                )
            return CoordinatorRouteAnalysisAction(
                task=request.user_message,
                file_ids=tuple(
                    attachment.file_id for attachment in request.attachments
                ),
            )
        if combined_terms & self._POLICY_TERMS:
            focused_task = request.user_message
            if not (message_terms & self._POLICY_TERMS) and request.attachments:
                focused_task = request.attachments[0].extracted_text[:8_000]
            return CoordinatorRouteRagAction(task=focused_task)
        if message.strip() in {"help", "can you help", "tell me more"}:
            return CoordinatorClarifyAction(
                question="What would you like help with: a Northstar policy or an uploaded report?"
            )
        return CoordinatorAnswerAction(
            markdown=(
                "I can help with Northstar policy questions and analyse uploaded reports."
            )
        )


class CoordinatorAgent:
    """Inspect one turn, select one route, and enforce delegation boundaries."""

    def __init__(
        self,
        *,
        provider: CoordinatorGenerationProvider,
        system_prompt: str,
        max_model_calls: int = 2,
        rag_specialist: RagSpecialist | None = None,
        analysis_specialist: AnalysisSpecialist | None = None,
        trace_recorder: CoordinatorTraceRecorder | None = None,
    ) -> None:
        if not system_prompt.strip():
            raise ValueError("system_prompt must not be blank")
        if max_model_calls < 1:
            raise ValueError("max_model_calls must be at least 1")
        self._provider = provider
        self._system_prompt = system_prompt.strip()
        self._max_model_calls = max_model_calls
        self._rag_specialist = rag_specialist
        self._analysis_specialist = analysis_specialist
        self._trace_recorder = trace_recorder or NoOpCoordinatorTraceRecorder()

    @property
    def tracing_enabled(self) -> bool:
        """Return whether coordinator turns are exported to Langfuse."""
        return self._trace_recorder.enabled

    def shutdown(self) -> None:
        """Release resources owned by connected specialists."""
        for specialist in (self._rag_specialist, self._analysis_specialist):
            if isinstance(specialist, ShutdownableSpecialist):
                specialist.shutdown()
        self._trace_recorder.shutdown()

    async def run(
        self,
        user_message: str,
        *,
        history: Sequence[ConversationMessage] = (),
        session_id: str | None = None,
        reports: Sequence[StoredReport] = (),
    ) -> AgentTurnResult:
        """Run one coordinator decision and at most one specialist delegation."""
        stripped_message = user_message.strip()
        if not stripped_message:
            raise ValueError("user_message must not be blank")
        async with self._trace_recorder.start_turn(
            session_id=session_id,
            user_message=stripped_message,
            history_message_count=len(history),
        ) as trace:
            result = await self._execute_turn(
                stripped_message,
                history=history,
                session_id=session_id,
                reports=reports,
                trace=trace,
            )
            trace.complete(result)
            return result

    async def _execute_turn(
        self,
        user_message: str,
        *,
        history: Sequence[ConversationMessage],
        session_id: str | None,
        reports: Sequence[StoredReport],
        trace: CoordinatorTrace,
    ) -> AgentTurnResult:
        """Execute the selected action while retaining one trace root."""
        attachments = tuple(self.attachment_from_report(report) for report in reports)
        trace.inspect_input(attachments)
        steps = [
            ExecutionStep(
                sequence=1,
                agent=ExecutionAgent.COORDINATOR,
                action=ExecutionAction.INSPECT_INPUT,
                status=ExecutionStatus.COMPLETED,
                summary=self._inspection_summary(attachments),
            )
        ]
        action_feedback: str | None = None
        repair_pending = False
        controller_events: list[ControllerEvent] = []

        for model_call_number in range(1, self._max_model_calls + 1):
            request = CoordinatorRequest(
                system_prompt=self._system_prompt,
                user_message=user_message,
                history=tuple(history),
                attachments=attachments,
                action_feedback=action_feedback,
                model_call_number=model_call_number,
            )
            with trace.model_action(
                request, model=self._provider.model_id
            ) as model_trace:
                try:
                    action = await self._provider.generate_action(request)
                except InvalidGenerationActionError as error:
                    model_trace.fail(error.code)
                    if repair_pending or model_call_number >= self._max_model_calls:
                        return self._failure(
                            message="The coordinator could not produce a valid action.",
                            reason=TerminationReason.PROVIDER_ERROR,
                            model_calls=model_call_number,
                            steps=steps,
                            controller_events=controller_events,
                        )
                    repair_pending = True
                    action_feedback = (
                        "The previous response did not match the coordinator action "
                        "schema. Return exactly one available action."
                    )
                    continue
                except GenerationProviderError as error:
                    model_trace.fail(error.code)
                    return self._failure(
                        message="The coordinator provider could not produce an action.",
                        reason=TerminationReason.PROVIDER_ERROR,
                        model_calls=model_call_number,
                        steps=steps,
                        controller_events=controller_events,
                    )
                model_trace.succeed(action)

            if repair_pending:
                controller_events.append(
                    ControllerEvent(
                        kind=ControllerEventKind.STRUCTURED_ACTION_REPAIRED,
                        detail=(
                            "Accepted a valid coordinator action after one schema "
                            "correction call."
                        ),
                    )
                )
            return await self._execute_action(
                action,
                history=tuple(history),
                session_id=session_id,
                attachments=attachments,
                model_calls=model_call_number,
                steps=steps,
                controller_events=controller_events,
                trace=trace,
            )

        raise AssertionError("bounded coordinator loop exited without a result")

    async def _execute_action(
        self,
        action: CoordinatorAction,
        *,
        history: tuple[ConversationMessage, ...],
        session_id: str | None,
        attachments: tuple[CoordinatorAttachment, ...],
        model_calls: int,
        steps: list[ExecutionStep],
        controller_events: list[ControllerEvent],
        trace: CoordinatorTrace,
    ) -> AgentTurnResult:
        if isinstance(action, CoordinatorAnswerAction):
            steps.append(
                self._step(
                    steps,
                    action=ExecutionAction.ANSWER,
                    summary="Returned a direct coordinator answer.",
                )
            )
            answer = AnswerAction(markdown=action.markdown, citations=())
            return AgentTurnResult(
                status=AgentTurnStatus.ANSWERED,
                message=action.markdown,
                controller_events=tuple(controller_events),
                model_actions=(answer,),
                execution_steps=tuple(steps),
                termination_reason=TerminationReason.ANSWERED,
                model_calls=model_calls,
                searches=0,
            )
        if isinstance(action, CoordinatorClarifyAction):
            steps.append(
                self._step(
                    steps,
                    action=ExecutionAction.CLARIFY,
                    summary="Requested additional user information.",
                )
            )
            clarification = ClarifyAction(question=action.question)
            return AgentTurnResult(
                status=AgentTurnStatus.CLARIFICATION_REQUIRED,
                message=action.question,
                controller_events=tuple(controller_events),
                model_actions=(clarification,),
                execution_steps=tuple(steps),
                termination_reason=TerminationReason.CLARIFICATION_REQUIRED,
                model_calls=model_calls,
                searches=0,
            )
        if isinstance(action, CoordinatorRouteRagAction):
            return await self._route_rag(
                action,
                history=history,
                session_id=session_id,
                model_calls=model_calls,
                steps=steps,
                controller_events=controller_events,
                trace=trace,
            )
        return await self._route_analysis(
            action,
            history=history,
            session_id=session_id,
            attachments=attachments,
            model_calls=model_calls,
            steps=steps,
            controller_events=controller_events,
            trace=trace,
        )

    async def _route_rag(
        self,
        action: CoordinatorRouteRagAction,
        *,
        history: tuple[ConversationMessage, ...],
        session_id: str | None,
        model_calls: int,
        steps: list[ExecutionStep],
        controller_events: list[ControllerEvent],
        trace: CoordinatorTrace,
    ) -> AgentTurnResult:
        specialist = self._rag_specialist
        if specialist is None:
            steps.append(
                self._step(
                    steps,
                    action=ExecutionAction.ROUTE_RAG,
                    status=ExecutionStatus.FAILED,
                    summary="Selected the RAG specialist, which is not connected yet.",
                )
            )
            return self._failure(
                message="The RAG specialist is not available in this runtime.",
                reason=TerminationReason.SPECIALIST_UNAVAILABLE,
                model_calls=model_calls,
                steps=steps,
                controller_events=controller_events,
            )
        request = RagHandoffRequest(task=action.task, history=history)
        steps.append(
            self._step(
                steps,
                action=ExecutionAction.ROUTE_RAG,
                summary="Delegated one focused policy task to the RAG specialist.",
            )
        )
        with trace.delegate_rag(action.task) as specialist_trace:
            result = await self._delegate(
                lambda: specialist.run(request, session_id=session_id),
                model_calls=model_calls,
                steps=steps,
                controller_events=controller_events,
            )
            specialist_trace.complete(result)
        if result.execution_steps[-1].action is ExecutionAction.RETURN_RESPONSE:
            trace.return_response(result)
        return result

    async def _route_analysis(
        self,
        action: CoordinatorRouteAnalysisAction,
        *,
        history: tuple[ConversationMessage, ...],
        session_id: str | None,
        attachments: tuple[CoordinatorAttachment, ...],
        model_calls: int,
        steps: list[ExecutionStep],
        controller_events: list[ControllerEvent],
        trace: CoordinatorTrace,
    ) -> AgentTurnResult:
        attachment_by_id = {
            attachment.file_id: attachment for attachment in attachments
        }
        if any(file_id not in attachment_by_id for file_id in action.file_ids):
            steps.append(
                self._step(
                    steps,
                    action=ExecutionAction.ROUTE_ANALYSIS,
                    status=ExecutionStatus.FAILED,
                    summary="Rejected a report handoff with unavailable file IDs.",
                )
            )
            return self._failure(
                message="The coordinator selected an unavailable report.",
                reason=TerminationReason.INVALID_HANDOFF,
                model_calls=model_calls,
                steps=steps,
                controller_events=controller_events,
            )
        selected = tuple(attachment_by_id[file_id] for file_id in action.file_ids)
        specialist = self._analysis_specialist
        if specialist is None:
            steps.append(
                self._step(
                    steps,
                    action=ExecutionAction.ROUTE_ANALYSIS,
                    status=ExecutionStatus.FAILED,
                    summary=(
                        "Selected the analysis specialist, which is not connected yet."
                    ),
                )
            )
            return self._failure(
                message="The analysis specialist is not available in this runtime.",
                reason=TerminationReason.SPECIALIST_UNAVAILABLE,
                model_calls=model_calls,
                steps=steps,
                controller_events=controller_events,
            )
        request = AnalysisHandoffRequest(
            task=action.task,
            history=history,
            attachments=selected,
        )
        steps.append(
            self._step(
                steps,
                action=ExecutionAction.ROUTE_ANALYSIS,
                summary="Delegated one focused report task to the analysis specialist.",
            )
        )
        with trace.delegate_analysis(action.task, selected) as specialist_trace:
            result = await self._delegate(
                lambda: specialist.run(request, session_id=session_id),
                model_calls=model_calls,
                steps=steps,
                controller_events=controller_events,
            )
            specialist_trace.complete(result)
        if result.execution_steps[-1].action is ExecutionAction.RETURN_RESPONSE:
            trace.return_response(result)
        return result

    async def _delegate(
        self,
        operation: "ProtocolCall",
        *,
        model_calls: int,
        steps: list[ExecutionStep],
        controller_events: list[ControllerEvent],
    ) -> AgentTurnResult:
        try:
            specialist_result = await operation()
        except Exception as error:
            log_failure(
                event="delegation",
                agent="coordinator",
                operation="run_specialist",
                code="delegation_error",
                error=error,
            )
            steps[-1] = steps[-1].model_copy(
                update={
                    "status": ExecutionStatus.FAILED,
                    "summary": "The selected specialist failed before returning a result.",
                }
            )
            return self._failure(
                message="The selected specialist could not complete the request.",
                reason=TerminationReason.DELEGATION_ERROR,
                model_calls=model_calls,
                steps=steps,
                controller_events=controller_events,
            )

        combined_steps = list(steps)
        for specialist_step in specialist_result.execution_steps:
            combined_steps.append(
                specialist_step.model_copy(update={"sequence": len(combined_steps) + 1})
            )
        combined_steps.append(
            ExecutionStep(
                sequence=len(combined_steps) + 1,
                agent=ExecutionAgent.COORDINATOR,
                action=ExecutionAction.RETURN_RESPONSE,
                status=ExecutionStatus.COMPLETED,
                summary="Returned the specialist result without changing its claims.",
            )
        )
        return AgentTurnResult(
            status=specialist_result.status,
            message=specialist_result.message,
            citations=specialist_result.citations,
            retrieved_chunks=specialist_result.retrieved_chunks,
            action_summaries=specialist_result.action_summaries,
            controller_events=(
                *controller_events,
                *specialist_result.controller_events,
            ),
            validation_findings=specialist_result.validation_findings,
            model_actions=specialist_result.model_actions,
            execution_steps=tuple(combined_steps),
            report_references=specialist_result.report_references,
            termination_reason=specialist_result.termination_reason,
            model_calls=model_calls + specialist_result.model_calls,
            searches=specialist_result.searches,
        )

    @staticmethod
    def attachment_from_report(report: StoredReport) -> CoordinatorAttachment:
        """Convert one authorized stored report into coordinator input."""
        return CoordinatorAttachment(
            file_id=report.metadata.file_id,
            filename=report.metadata.filename,
            media_type=report.metadata.media_type,
            byte_size=report.metadata.byte_size,
            extracted_text=report.extracted_text,
            pages=tuple(
                CoordinatorAttachmentPage(
                    page_number=page.page_number,
                    text=page.text,
                )
                for page in report.pages
            ),
        )

    @staticmethod
    def _inspection_summary(
        attachments: tuple[CoordinatorAttachment, ...],
    ) -> str:
        if not attachments:
            return "Inspected text input with no report attachments."
        details = ", ".join(
            f"{attachment.filename} ({attachment.file_id}, "
            f"{len(attachment.extracted_text)} characters)"
            for attachment in attachments
        )
        summary = (
            f"Inspected text input with {len(attachments)} report "
            f"{'attachment' if len(attachments) == 1 else 'attachments'}: {details}."
        )
        return summary if len(summary) <= 500 else f"{summary[:497]}..."

    @staticmethod
    def _step(
        steps: Sequence[ExecutionStep],
        *,
        action: ExecutionAction,
        summary: str,
        status: ExecutionStatus = ExecutionStatus.COMPLETED,
    ) -> ExecutionStep:
        return ExecutionStep(
            sequence=len(steps) + 1,
            agent=ExecutionAgent.COORDINATOR,
            action=action,
            status=status,
            summary=summary,
        )

    @staticmethod
    def _failure(
        *,
        message: str,
        reason: TerminationReason,
        model_calls: int,
        steps: Sequence[ExecutionStep],
        controller_events: Sequence[ControllerEvent],
    ) -> AgentTurnResult:
        if reason not in (
            TerminationReason.PROVIDER_ERROR,
            TerminationReason.DELEGATION_ERROR,
        ):
            log_failure(
                event="agent_stop",
                agent="coordinator",
                operation="dispatch",
                code=reason.value,
                attempt=model_calls,
            )
        return AgentTurnResult(
            status=AgentTurnStatus.FAILED,
            message=message,
            controller_events=tuple(controller_events),
            execution_steps=tuple(steps),
            termination_reason=reason,
            model_calls=model_calls,
            searches=0,
        )


class ProtocolCall(Protocol):
    """Zero-argument operation returning one typed specialist result."""

    async def __call__(self) -> AgentTurnResult: ...


def build_openai_coordinator_provider(
    *,
    api_key: str,
    base_url: str,
    model: str,
    revision: str,
    temperature: float,
    max_tokens: int,
    timeout_seconds: float,
    client: StructuredCompletionClient | None = None,
) -> OpenAICoordinatorGenerationProvider:
    """Construct the live adapter without enabling incomplete Stage 16 tracing."""
    client = client or OpenAICompatibleChatClient(
        api_key=api_key,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        instrument_with_langfuse=False,
    )
    return OpenAICoordinatorGenerationProvider(
        model=model,
        revision=revision,
        temperature=temperature,
        max_tokens=max_tokens,
        client=client,
    )
