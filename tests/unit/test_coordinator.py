"""Tests for bounded coordinator routing and provider adapters."""

import json
import logging
from collections.abc import Sequence
from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError

from agentic_eval_toyapp.coordinator import (
    CoordinatorAgent,
    DeterministicCoordinatorGenerationProvider,
    OpenAICoordinatorGenerationProvider,
    ScriptedCoordinatorGenerationProvider,
    _StructuredCoordinatorAction,  # pyright: ignore[reportPrivateUsage]
    load_coordinator_prompt,
)
from agentic_eval_toyapp.diagnostics import CompletionDiagnostics
from agentic_eval_toyapp.models import (
    AgentTurnResult,
    AgentTurnStatus,
    AnswerAction,
    ControllerEventKind,
    ConversationMessage,
    ConversationRole,
    ExecutionAction,
    ExecutionAgent,
    ExecutionStatus,
    ExecutionStep,
    ReportMediaType,
    ReportUploadResponse,
    TerminationReason,
)
from agentic_eval_toyapp.orchestration_models import (
    AnalysisHandoffRequest,
    CoordinatorAnswerAction,
    CoordinatorClarifyAction,
    CoordinatorRequest,
    CoordinatorRouteAnalysisAction,
    CoordinatorRouteRagAction,
    RagHandoffRequest,
)
from agentic_eval_toyapp.providers import (
    GenerationProviderError,
    InvalidGenerationActionError,
    StructuredCompletion,
    make_object_schemas_strict,
)
from agentic_eval_toyapp.reports import StoredReport


def _report(text: str = "Revenue increased by 12%.") -> StoredReport:
    metadata = ReportUploadResponse(
        file_id=uuid4(),
        filename="quarterly.md",
        media_type=ReportMediaType.MARKDOWN,
        byte_size=len(text.encode()),
        extracted_text_length=len(text),
        sha256="0" * 64,
    )
    return StoredReport(
        metadata=metadata,
        original_bytes=text.encode(),
        extracted_text=text,
    )


def _answer_result(message: str, *, agent: ExecutionAgent) -> AgentTurnResult:
    action = AnswerAction(markdown=message, citations=())
    return AgentTurnResult(
        status=AgentTurnStatus.ANSWERED,
        message=message,
        model_actions=(action,),
        execution_steps=(
            ExecutionStep(
                sequence=1,
                agent=agent,
                action=ExecutionAction.ANSWER,
                status=ExecutionStatus.COMPLETED,
                summary="Returned a specialist answer.",
            ),
        ),
        termination_reason=TerminationReason.ANSWERED,
        model_calls=1,
        searches=0,
    )


class _RecordingRagSpecialist:
    def __init__(self) -> None:
        self.requests: list[RagHandoffRequest] = []
        self.session_ids: list[str | None] = []
        self.shutdown_calls = 0

    async def run(
        self,
        request: RagHandoffRequest,
        *,
        session_id: str | None = None,
    ) -> AgentTurnResult:
        self.requests.append(request)
        self.session_ids.append(session_id)
        return _answer_result("Policy specialist answer", agent=ExecutionAgent.RAG)

    def shutdown(self) -> None:
        self.shutdown_calls += 1


class _RecordingAnalysisSpecialist:
    def __init__(self) -> None:
        self.requests: list[AnalysisHandoffRequest] = []

    async def run(
        self,
        request: AnalysisHandoffRequest,
        *,
        session_id: str | None = None,
    ) -> AgentTurnResult:
        del session_id
        self.requests.append(request)
        return _answer_result(
            "Analysis specialist answer", agent=ExecutionAgent.ANALYSIS
        )


class _RepairingProvider:
    model_id = "repair-fixture"
    revision = "v1"

    def __init__(self) -> None:
        self.requests: list[CoordinatorRequest] = []

    async def generate_action(
        self, request: CoordinatorRequest
    ) -> CoordinatorAnswerAction:
        self.requests.append(request)
        if len(self.requests) == 1:
            raise InvalidGenerationActionError("malformed fixture")
        return CoordinatorAnswerAction(markdown="Repaired answer")


class _FailingProvider:
    model_id = "failure-fixture"
    revision = "v1"

    async def generate_action(
        self, request: CoordinatorRequest
    ) -> CoordinatorAnswerAction:
        del request
        raise GenerationProviderError("private upstream error")


class _FailingRagSpecialist:
    async def run(
        self,
        request: RagHandoffRequest,
        *,
        session_id: str | None = None,
    ) -> AgentTurnResult:
        del request, session_id
        raise RuntimeError("private specialist error")


class _RecordingCompletionClient:
    def __init__(
        self,
        response: dict[str, object],
        *,
        error: Exception | None = None,
        diagnostics: CompletionDiagnostics | None = None,
    ) -> None:
        self._response = response
        self._error = error
        self._diagnostics = diagnostics
        self.calls: list[dict[str, object]] = []

    async def create(
        self,
        *,
        model: str,
        messages: tuple[dict[str, str], ...],
        response_schema: dict[str, Any],
        temperature: float,
        max_tokens: int,
    ) -> str | StructuredCompletion:
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "response_schema": response_schema,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        if self._error is not None:
            raise self._error
        content = json.dumps(self._response)
        if self._diagnostics is not None:
            return StructuredCompletion(content, self._diagnostics)
        return content


@pytest.mark.anyio
async def test_coordinator_returns_direct_answers_and_clarification() -> None:
    """Non-specialist actions should produce valid results and public steps."""
    answer_provider = ScriptedCoordinatorGenerationProvider(
        (CoordinatorAnswerAction(markdown="Hello from Northstar."),)
    )
    clarification_provider = ScriptedCoordinatorGenerationProvider(
        (CoordinatorClarifyAction(question="Which policy do you mean?"),)
    )

    answer = await CoordinatorAgent(
        provider=answer_provider,
        system_prompt="Coordinate requests.",
    ).run("Hello")
    clarification = await CoordinatorAgent(
        provider=clarification_provider,
        system_prompt="Coordinate requests.",
    ).run("Tell me more")

    assert answer.status is AgentTurnStatus.ANSWERED
    assert answer.message == "Hello from Northstar."
    assert [step.action for step in answer.execution_steps] == [
        ExecutionAction.INSPECT_INPUT,
        ExecutionAction.ANSWER,
    ]
    assert clarification.status is AgentTurnStatus.CLARIFICATION_REQUIRED
    assert clarification.termination_reason is TerminationReason.CLARIFICATION_REQUIRED
    assert clarification.execution_steps[-1].action is ExecutionAction.CLARIFY


def test_coordinator_shutdown_releases_connected_specialist_resources() -> None:
    """The process lifecycle should reach resources owned by the RAG specialist."""
    rag = _RecordingRagSpecialist()
    coordinator = CoordinatorAgent(
        provider=ScriptedCoordinatorGenerationProvider(
            (CoordinatorAnswerAction(markdown="Hello"),)
        ),
        system_prompt="Coordinate requests.",
        rag_specialist=rag,
    )

    coordinator.shutdown()

    assert rag.shutdown_calls == 1


@pytest.mark.anyio
async def test_coordinator_passes_history_and_safe_attachment_context() -> None:
    """The model sees authorized content while public steps omit that content."""
    secret_text = "Confidential finding: supplier concentration is 78%."
    report = _report(secret_text)
    provider = ScriptedCoordinatorGenerationProvider(
        (CoordinatorAnswerAction(markdown="Acknowledged."),)
    )
    history = (
        ConversationMessage(role=ConversationRole.USER, content="Earlier request"),
        ConversationMessage(role=ConversationRole.ASSISTANT, content="Earlier reply"),
    )

    result = await CoordinatorAgent(
        provider=provider,
        system_prompt="Coordinate requests.",
    ).run("Use my context", history=history, reports=(report,))

    request = provider.requests[0]
    assert request.history == history
    assert request.attachments[0].extracted_text == secret_text
    assert all(secret_text not in step.summary for step in result.execution_steps)
    assert report.metadata.filename in result.execution_steps[0].summary
    assert str(report.metadata.file_id) in result.execution_steps[0].summary


@pytest.mark.anyio
async def test_coordinator_delegates_each_valid_route_once_and_preserves_result() -> (
    None
):
    """One model route should produce one typed handoff and unchanged claims."""
    rag = _RecordingRagSpecialist()
    rag_result = await CoordinatorAgent(
        provider=ScriptedCoordinatorGenerationProvider(
            (CoordinatorRouteRagAction(task="Explain sick leave evidence"),)
        ),
        system_prompt="Coordinate requests.",
        rag_specialist=rag,
    ).run("Sick leave", session_id="session-1")

    report = _report()
    analysis = _RecordingAnalysisSpecialist()
    analysis_result = await CoordinatorAgent(
        provider=ScriptedCoordinatorGenerationProvider(
            (
                CoordinatorRouteAnalysisAction(
                    task="Summarise the main finding",
                    file_ids=(report.metadata.file_id,),
                ),
            )
        ),
        system_prompt="Coordinate requests.",
        analysis_specialist=analysis,
    ).run("Summarise it", reports=(report,))

    assert rag.requests[0].task == "Explain sick leave evidence"
    assert rag.session_ids == ["session-1"]
    assert rag_result.message == "Policy specialist answer"
    assert rag_result.model_calls == 2
    assert [step.action for step in rag_result.execution_steps] == [
        ExecutionAction.INSPECT_INPUT,
        ExecutionAction.ROUTE_RAG,
        ExecutionAction.ANSWER,
        ExecutionAction.RETURN_RESPONSE,
    ]
    assert len(analysis.requests) == 1
    assert analysis.requests[0].attachments[0].extracted_text == report.extracted_text
    assert analysis_result.message == "Analysis specialist answer"


@pytest.mark.anyio
async def test_coordinator_preserves_wrong_route_and_rejects_unknown_file_handoff(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The controller must expose model-quality errors instead of rerouting them."""
    report = _report("The annual leave policy allows 18 days.")
    analysis = _RecordingAnalysisSpecialist()
    wrong_route = await CoordinatorAgent(
        provider=ScriptedCoordinatorGenerationProvider(
            (
                CoordinatorRouteAnalysisAction(
                    task="Answer the policy question",
                    file_ids=(report.metadata.file_id,),
                ),
            )
        ),
        system_prompt="Coordinate requests.",
        analysis_specialist=analysis,
    ).run("What is the annual leave policy?", reports=(report,))

    with caplog.at_level(logging.ERROR, logger="uvicorn.error"):
        unknown_file = await CoordinatorAgent(
            provider=ScriptedCoordinatorGenerationProvider(
                (
                    CoordinatorRouteAnalysisAction(
                        task="Analyse",
                        file_ids=(uuid4(),),
                    ),
                )
            ),
            system_prompt="Coordinate requests.",
            analysis_specialist=analysis,
        ).run("Analyse this", reports=(report,))

    assert wrong_route.message == "Analysis specialist answer"
    assert len(analysis.requests) == 1
    assert unknown_file.status is AgentTurnStatus.FAILED
    assert unknown_file.termination_reason is TerminationReason.INVALID_HANDOFF
    assert unknown_file.execution_steps[-1].action is ExecutionAction.ROUTE_ANALYSIS
    assert unknown_file.execution_steps[-1].status is ExecutionStatus.FAILED
    assert "code=invalid_handoff" in caplog.text


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("action", "expected_action"),
    [
        (CoordinatorRouteRagAction(task="Policy task"), ExecutionAction.ROUTE_RAG),
        (
            CoordinatorRouteAnalysisAction(
                task="Analysis task",
                file_ids=(uuid4(),),
            ),
            ExecutionAction.ROUTE_ANALYSIS,
        ),
    ],
)
async def test_unconnected_specialist_routes_fail_explicitly(
    action: CoordinatorRouteRagAction | CoordinatorRouteAnalysisAction,
    expected_action: ExecutionAction,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Stage 12 routes must not pretend deferred specialists are operational."""
    reports: Sequence[StoredReport] = ()
    if isinstance(action, CoordinatorRouteAnalysisAction):
        reports = (_report(),)
        action = CoordinatorRouteAnalysisAction(
            task=action.task,
            file_ids=(reports[0].metadata.file_id,),
        )
    with caplog.at_level(logging.ERROR, logger="uvicorn.error"):
        result = await CoordinatorAgent(
            provider=ScriptedCoordinatorGenerationProvider((action,)),
            system_prompt="Coordinate requests.",
        ).run("Request", reports=reports)

    assert result.status is AgentTurnStatus.FAILED
    assert result.termination_reason is TerminationReason.SPECIALIST_UNAVAILABLE
    assert result.execution_steps[-1].action is expected_action
    assert result.execution_steps[-1].status is ExecutionStatus.FAILED
    assert "code=specialist_unavailable" in caplog.text


@pytest.mark.anyio
async def test_coordinator_repairs_once_and_bounds_provider_failures() -> None:
    """One schema repair is allowed; provider failures expose no private detail."""
    provider = _RepairingProvider()
    repaired = await CoordinatorAgent(
        provider=provider,
        system_prompt="Coordinate requests.",
        max_model_calls=2,
    ).run("Hello")
    failed = await CoordinatorAgent(
        provider=_FailingProvider(),
        system_prompt="Coordinate requests.",
    ).run("Hello")
    exhausted = await CoordinatorAgent(
        provider=_RepairingProvider(),
        system_prompt="Coordinate requests.",
        max_model_calls=1,
    ).run("Hello")

    assert repaired.message == "Repaired answer"
    assert repaired.model_calls == 2
    assert provider.requests[1].action_feedback is not None
    assert repaired.controller_events[0].kind is (
        ControllerEventKind.STRUCTURED_ACTION_REPAIRED
    )
    assert failed.termination_reason is TerminationReason.PROVIDER_ERROR
    assert "private" not in failed.message
    assert exhausted.termination_reason is TerminationReason.PROVIDER_ERROR
    assert exhausted.model_calls == 1


@pytest.mark.anyio
async def test_specialist_exception_is_bounded_and_route_is_not_repeated(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A specialist exception should end the single allowed delegation."""
    with caplog.at_level(logging.ERROR, logger="uvicorn.error"):
        result = await CoordinatorAgent(
            provider=ScriptedCoordinatorGenerationProvider(
                (CoordinatorRouteRagAction(task="Policy task"),)
            ),
            system_prompt="Coordinate requests.",
            rag_specialist=_FailingRagSpecialist(),
        ).run("Policy question")

    assert result.termination_reason is TerminationReason.DELEGATION_ERROR
    assert result.model_calls == 1
    assert [step.action for step in result.execution_steps].count(
        ExecutionAction.ROUTE_RAG
    ) == 1
    assert "private" not in result.message
    assert len(caplog.records) == 1
    assert "code=delegation_error" in caplog.text
    assert "private specialist error" not in caplog.text


@pytest.mark.anyio
async def test_live_coordinator_logs_timeout_without_private_provider_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A gateway timeout is distinguishable from an invalid route decision."""
    provider = OpenAICoordinatorGenerationProvider(
        model="platform-model",
        revision="provider-managed",
        temperature=0,
        max_tokens=512,
        client=_RecordingCompletionClient({}, error=TimeoutError("private token")),
    )
    request = CoordinatorRequest(
        system_prompt="Coordinate",
        user_message="Question",
        model_call_number=1,
    )

    with (
        caplog.at_level(logging.ERROR, logger="uvicorn.error"),
        pytest.raises(GenerationProviderError, match="live coordinator request failed"),
    ):
        await provider.generate_action(request)

    assert len(caplog.records) == 1
    assert "agent=coordinator" in caplog.text
    assert "code=timeout" in caplog.text


@pytest.mark.anyio
async def test_live_coordinator_schema_error_logs_provider_usage(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A bad coordinator action retains safe completion metadata."""
    provider = OpenAICoordinatorGenerationProvider(
        model="platform-model",
        revision="provider-managed",
        temperature=0,
        max_tokens=512,
        client=_RecordingCompletionClient(
            {"kind": "not-a-route", "private": "report text"},
            diagnostics=CompletionDiagnostics(
                finish_reason="stop", input_tokens=410, output_tokens=21
            ),
        ),
    )

    with (
        caplog.at_level(logging.ERROR, logger="uvicorn.error"),
        pytest.raises(InvalidGenerationActionError),
    ):
        await provider.generate_action(
            CoordinatorRequest(
                system_prompt="Coordinate",
                user_message="Analyse the report",
                model_call_number=1,
            )
        )

    assert len(caplog.records) == 1
    assert "agent=coordinator" in caplog.text
    assert "finish_reason=stop input_tokens=410 output_tokens=21" in caplog.text
    assert "report text" not in caplog.text
    assert "private token" not in caplog.text


@pytest.mark.anyio
async def test_deterministic_coordinator_routes_by_task_not_file_presence() -> None:
    """Offline routing should distinguish policy content from report analysis."""
    provider = DeterministicCoordinatorGenerationProvider()
    policy_report = _report("What evidence is needed for sick leave?")
    policy_request = CoordinatorRequest(
        system_prompt="Coordinate",
        user_message="Please answer the question in this file",
        attachments=(CoordinatorAgent.attachment_from_report(policy_report),),
        model_call_number=1,
    )
    analysis_request = CoordinatorRequest(
        system_prompt="Coordinate",
        user_message="Summarise the main findings",
        attachments=(CoordinatorAgent.attachment_from_report(policy_report),),
        model_call_number=1,
    )

    assert isinstance(
        await provider.generate_action(policy_request), CoordinatorRouteRagAction
    )
    assert isinstance(
        await provider.generate_action(analysis_request), CoordinatorRouteAnalysisAction
    )


@pytest.mark.anyio
async def test_openai_coordinator_provider_sends_context_and_validates_response() -> (
    None
):
    """The live adapter should use a strict schema and preserve the model route."""
    client = _RecordingCompletionClient(
        {
            "kind": "route_rag",
            "markdown": None,
            "question": None,
            "task": "Explain the sick leave evidence requirement",
            "file_ids": [],
        }
    )
    provider = OpenAICoordinatorGenerationProvider(
        model="platform-model",
        revision="provider-managed",
        temperature=0.0,
        max_tokens=512,
        client=client,
    )
    request = CoordinatorRequest(
        system_prompt="Coordinate",
        user_message="What proof do I need for sick leave?",
        model_call_number=1,
    )

    action = await provider.generate_action(request)

    assert isinstance(action, CoordinatorRouteRagAction)
    call = client.calls[0]
    schema = call["response_schema"]
    assert isinstance(schema, dict)
    assert schema["additionalProperties"] is False
    messages = call["messages"]
    assert isinstance(messages, tuple)
    assert "sick leave" in messages[1]["content"]


def test_coordinator_prompt_is_non_empty_and_defends_report_boundary() -> None:
    """The versioned prompt should define routing and untrusted-data handling."""
    prompt = load_coordinator_prompt()

    assert "route_rag" in prompt
    assert "route_analysis" in prompt
    assert "untrusted data" in prompt


_COORDINATOR_FULL: dict[str, object] = {
    "kind": "answer",
    "markdown": "Hello",
    "question": None,
    "task": None,
    "file_ids": [],
}


@pytest.mark.anyio
async def test_coordinator_accepts_reply_that_omits_null_fields() -> None:
    """The exact reply the bedrock route sent must parse as a direct answer."""
    client = _RecordingCompletionClient(
        {"kind": "answer", "markdown": "Hi! How can I help you today?", "file_ids": []}
    )
    provider = OpenAICoordinatorGenerationProvider(
        model="platform-model",
        revision="provider-managed",
        temperature=0.0,
        max_tokens=512,
        client=client,
    )

    action = await provider.generate_action(
        CoordinatorRequest(
            system_prompt="Coordinate", user_message="Hello", model_call_number=1
        )
    )

    assert action == CoordinatorAnswerAction(markdown="Hi! How can I help you today?")


def test_structured_coordinator_action_omitted_nullables_equal_explicit_nulls() -> None:
    omitted = _StructuredCoordinatorAction.model_validate(
        {"kind": "answer", "markdown": "Hello", "file_ids": []}
    )

    assert omitted == _StructuredCoordinatorAction.model_validate(_COORDINATOR_FULL)
    assert omitted.question is None
    assert omitted.task is None


@pytest.mark.parametrize(
    "reply",
    [
        {"kind": "answer", "question": "Which?", "file_ids": []},
        {"kind": "answer", "markdown": "Hi", "task": "Search", "file_ids": []},
        {"kind": "route_rag", "markdown": "Hi", "file_ids": []},
        {**_COORDINATOR_FULL, "unexpected": 1},
        {"kind": "answer", "markdown": "Hi"},
        {"kind": "route_analysis", "task": "Summarise"},
    ],
    ids=[
        "wrong-field-clarify-data",
        "wrong-field-extra-task",
        "route-without-task",
        "extra-key",
        "omitted-list",
        "analysis-without-files",
    ],
)
def test_structured_coordinator_action_still_rejects_invalid_replies(
    reply: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        _StructuredCoordinatorAction.model_validate(reply)


def test_structured_coordinator_action_schema_required_list_is_unchanged() -> None:
    schema = _StructuredCoordinatorAction.model_json_schema()
    assert "default" not in schema["properties"]["markdown"]
    make_object_schemas_strict(schema)

    assert schema["required"] == ["kind", "markdown", "question", "task", "file_ids"]
    assert schema["additionalProperties"] is False
