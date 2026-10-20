"""Tests for bounded coordinator routing and provider adapters."""

import json
from collections.abc import Sequence
from typing import Any
from uuid import uuid4

import pytest

from agentic_eval_toyapp.coordinator import (
    CoordinatorAgent,
    DeterministicCoordinatorGenerationProvider,
    OpenAICoordinatorGenerationProvider,
    ScriptedCoordinatorGenerationProvider,
    load_coordinator_prompt,
)
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
    return StoredReport(metadata=metadata, extracted_text=text)


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

    def run(
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

    def run(
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

    def generate_action(self, request: CoordinatorRequest) -> CoordinatorAnswerAction:
        self.requests.append(request)
        if len(self.requests) == 1:
            raise InvalidGenerationActionError("malformed fixture")
        return CoordinatorAnswerAction(markdown="Repaired answer")


class _FailingProvider:
    model_id = "failure-fixture"
    revision = "v1"

    def generate_action(self, request: CoordinatorRequest) -> CoordinatorAnswerAction:
        del request
        raise GenerationProviderError("private upstream error")


class _FailingRagSpecialist:
    def run(
        self,
        request: RagHandoffRequest,
        *,
        session_id: str | None = None,
    ) -> AgentTurnResult:
        del request, session_id
        raise RuntimeError("private specialist error")


class _RecordingCompletionClient:
    def __init__(self, response: dict[str, object]) -> None:
        self._response = response
        self.calls: list[dict[str, object]] = []

    def create(
        self,
        *,
        model: str,
        messages: tuple[dict[str, str], ...],
        response_schema: dict[str, Any],
        temperature: float,
        max_tokens: int,
    ) -> str:
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "response_schema": response_schema,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        return json.dumps(self._response)


def test_coordinator_returns_direct_answers_and_clarification() -> None:
    """Non-specialist actions should produce valid results and public steps."""
    answer_provider = ScriptedCoordinatorGenerationProvider(
        (CoordinatorAnswerAction(markdown="Hello from Northstar."),)
    )
    clarification_provider = ScriptedCoordinatorGenerationProvider(
        (CoordinatorClarifyAction(question="Which policy do you mean?"),)
    )

    answer = CoordinatorAgent(
        provider=answer_provider,
        system_prompt="Coordinate requests.",
    ).run("Hello")
    clarification = CoordinatorAgent(
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


def test_coordinator_passes_history_and_safe_attachment_context() -> None:
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

    result = CoordinatorAgent(
        provider=provider,
        system_prompt="Coordinate requests.",
    ).run("Use my context", history=history, reports=(report,))

    request = provider.requests[0]
    assert request.history == history
    assert request.attachments[0].extracted_text == secret_text
    assert all(secret_text not in step.summary for step in result.execution_steps)
    assert report.metadata.filename in result.execution_steps[0].summary
    assert str(report.metadata.file_id) in result.execution_steps[0].summary


def test_coordinator_delegates_each_valid_route_once_and_preserves_result() -> None:
    """One model route should produce one typed handoff and unchanged claims."""
    rag = _RecordingRagSpecialist()
    rag_result = CoordinatorAgent(
        provider=ScriptedCoordinatorGenerationProvider(
            (CoordinatorRouteRagAction(task="Explain sick leave evidence"),)
        ),
        system_prompt="Coordinate requests.",
        rag_specialist=rag,
    ).run("Sick leave", session_id="session-1")

    report = _report()
    analysis = _RecordingAnalysisSpecialist()
    analysis_result = CoordinatorAgent(
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


def test_coordinator_preserves_wrong_route_and_rejects_unknown_file_handoff() -> None:
    """The controller must expose model-quality errors instead of rerouting them."""
    report = _report("The annual leave policy allows 18 days.")
    analysis = _RecordingAnalysisSpecialist()
    wrong_route = CoordinatorAgent(
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

    unknown_file = CoordinatorAgent(
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
def test_unconnected_specialist_routes_fail_explicitly(
    action: CoordinatorRouteRagAction | CoordinatorRouteAnalysisAction,
    expected_action: ExecutionAction,
) -> None:
    """Stage 12 routes must not pretend deferred specialists are operational."""
    reports: Sequence[StoredReport] = ()
    if isinstance(action, CoordinatorRouteAnalysisAction):
        reports = (_report(),)
        action = CoordinatorRouteAnalysisAction(
            task=action.task,
            file_ids=(reports[0].metadata.file_id,),
        )
    result = CoordinatorAgent(
        provider=ScriptedCoordinatorGenerationProvider((action,)),
        system_prompt="Coordinate requests.",
    ).run("Request", reports=reports)

    assert result.status is AgentTurnStatus.FAILED
    assert result.termination_reason is TerminationReason.SPECIALIST_UNAVAILABLE
    assert result.execution_steps[-1].action is expected_action
    assert result.execution_steps[-1].status is ExecutionStatus.FAILED


def test_coordinator_repairs_once_and_bounds_provider_failures() -> None:
    """One schema repair is allowed; provider failures expose no private detail."""
    provider = _RepairingProvider()
    repaired = CoordinatorAgent(
        provider=provider,
        system_prompt="Coordinate requests.",
        max_model_calls=2,
    ).run("Hello")
    failed = CoordinatorAgent(
        provider=_FailingProvider(),
        system_prompt="Coordinate requests.",
    ).run("Hello")
    exhausted = CoordinatorAgent(
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


def test_specialist_exception_is_bounded_and_route_is_not_repeated() -> None:
    """A specialist exception should end the single allowed delegation."""
    result = CoordinatorAgent(
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


def test_deterministic_coordinator_routes_by_task_not_file_presence() -> None:
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
        provider.generate_action(policy_request), CoordinatorRouteRagAction
    )
    assert isinstance(
        provider.generate_action(analysis_request), CoordinatorRouteAnalysisAction
    )


def test_openai_coordinator_provider_sends_context_and_validates_response() -> None:
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

    action = provider.generate_action(request)

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
