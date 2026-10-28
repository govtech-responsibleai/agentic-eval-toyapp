"""Tests for the bounded single-agent RAG loop."""

import logging
from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from datetime import date
from pathlib import Path

import pytest
from anyio import CapacityLimiter

from agentic_eval_toyapp.agent import (
    PromptLoadError,
    RagAgent,
    RagSpecialistAdapter,
    load_system_prompt,
)
from agentic_eval_toyapp.models import (
    AgentAction,
    AgentActionKind,
    AgentTurnStatus,
    AnswerAction,
    ClarifyAction,
    ControllerEventKind,
    ConversationMessage,
    ConversationRole,
    DocumentMetadata,
    ExecutionAction,
    ExecutionAgent,
    ExecutionStatus,
    ModelRequest,
    RetrievedChunk,
    SearchAction,
    TerminationReason,
    ValidationFindingCode,
)
from agentic_eval_toyapp.orchestration_models import RagHandoffRequest
from agentic_eval_toyapp.providers import (
    InvalidGenerationActionError,
    ScriptedGenerationProvider,
)
from agentic_eval_toyapp.retrieval import AsyncRetriever, load_corpus
from agentic_eval_toyapp.tracing import TurnTrace


def _chunk(chunk_id: str = "TRAVEL-001#1") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=chunk_id.split("#", maxsplit=1)[0],
        title="Travel policy",
        text="Domestic meals are limited to $50 per day.",
        metadata=DocumentMetadata(topic="travel", effective_date=date(2026, 1, 1)),
        similarity_score=0.9,
    )


class _StubRetriever:
    def __init__(
        self,
        results_by_query: Mapping[str, tuple[RetrievedChunk, ...]],
        *,
        error: Exception | None = None,
    ) -> None:
        self._results_by_query = results_by_query
        self._error = error
        self.calls: list[tuple[str, int | None]] = []

    def search(
        self, query: str, *, top_k: int | None = None
    ) -> tuple[RetrievedChunk, ...]:
        self.calls.append((query, top_k))
        if self._error is not None:
            raise self._error
        return self._results_by_query.get(query, ())


class _RepairingProvider:
    """Fail one structured action, then return a valid direct answer."""

    model_id = "repairing-fixture"
    revision = "v1"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def generate_action(self, request: ModelRequest) -> AgentAction:
        self.requests.append(request)
        if len(self.requests) == 1:
            raise InvalidGenerationActionError("invalid fixture action")
        return AnswerAction(
            markdown="I can help with Northstar policies.", citations=()
        )


class _AlwaysInvalidProvider:
    """Return no protocol-valid action across the one allowed repair."""

    model_id = "invalid-fixture"
    revision = "v1"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def generate_action(self, request: ModelRequest) -> AgentAction:
        self.requests.append(request)
        raise InvalidGenerationActionError("invalid fixture action")


class _ForbiddenRootTraceRecorder:
    """Fail if delegated execution tries to create a standalone trace root."""

    def __init__(self) -> None:
        self.shutdown_calls = 0

    @property
    def enabled(self) -> bool:
        return True

    def start_turn(
        self,
        *,
        session_id: str | None,
        user_message: str,
        history_message_count: int,
    ) -> AbstractAsyncContextManager[TurnTrace]:
        del session_id, user_message, history_message_count
        raise AssertionError("delegated RAG must not create a root trace")

    def flush(self) -> None:
        return None

    def shutdown(self) -> None:
        self.shutdown_calls += 1


def _agent(
    actions: tuple[AgentAction, ...],
    retriever: _StubRetriever,
    *,
    max_model_calls: int = 4,
    max_searches: int = 2,
) -> tuple[RagAgent, ScriptedGenerationProvider]:
    provider = ScriptedGenerationProvider(actions)
    return (
        RagAgent(
            provider=provider,
            retriever=AsyncRetriever(retriever, CapacityLimiter(1)),
            system_prompt="Search, clarify, or answer with citations.",
            max_model_calls=max_model_calls,
            max_searches=max_searches,
            retrieval_top_k=3,
        ),
        provider,
    )


@pytest.mark.anyio
async def test_agent_searches_then_returns_grounded_answer() -> None:
    """Evidence must reach the second model call before an answer is accepted."""
    chunk = _chunk()
    retriever = _StubRetriever({"domestic meals": (chunk,)})
    agent, provider = _agent(
        (
            SearchAction(query="domestic meals"),
            AnswerAction(
                markdown="The limit is $50 per day. [TRAVEL-001#1]",
                citations=("TRAVEL-001#1",),
            ),
        ),
        retriever,
    )
    history = (
        ConversationMessage(role=ConversationRole.USER, content="Earlier question"),
        ConversationMessage(role=ConversationRole.ASSISTANT, content="Earlier reply"),
    )

    result = await agent.run("What is the domestic meal limit?", history=history)

    assert result.status is AgentTurnStatus.ANSWERED
    assert result.termination_reason is TerminationReason.ANSWERED
    assert result.citations == ("TRAVEL-001#1",)
    assert result.model_calls == 2
    assert result.searches == 1
    assert retriever.calls == [("domestic meals", 3)]
    assert provider.requests[0].history == history
    assert provider.requests[0].available_actions == (
        AgentActionKind.SEARCH,
        AgentActionKind.CLARIFY,
        AgentActionKind.ANSWER,
    )
    assert provider.requests[1].retrieved_chunks == (chunk,)
    assert provider.requests[1].available_actions == (
        AgentActionKind.SEARCH,
        AgentActionKind.CLARIFY,
        AgentActionKind.ANSWER,
    )
    assert [summary.action for summary in result.action_summaries] == [
        AgentActionKind.SEARCH,
        AgentActionKind.ANSWER,
    ]
    assert [summary.detail for summary in result.action_summaries] == [
        "Retrieved 1 chunk with top_k=3.",
        "Returned an answer with 1 citation.",
    ]
    assert all(
        "domestic meals" not in summary.detail for summary in result.action_summaries
    )


@pytest.mark.anyio
async def test_agent_returns_clarification_without_searching() -> None:
    """Material ambiguity can terminate with one direct clarification action."""
    retriever = _StubRetriever({})
    agent, _ = _agent(
        (ClarifyAction(question="Is the trip domestic or international?"),),
        retriever,
    )

    result = await agent.run("What travel allowance can I claim?")

    assert result.status is AgentTurnStatus.CLARIFICATION_REQUIRED
    assert result.message == "Is the trip domestic or international?"
    assert result.termination_reason is TerminationReason.CLARIFICATION_REQUIRED
    assert result.searches == 0
    assert retriever.calls == []


@pytest.mark.anyio
async def test_agent_action_summaries_pluralise_multiple_counts() -> None:
    """Observable summaries should use grammatical nouns for multiple items."""
    first = _chunk("TRAVEL-001#1")
    second = _chunk("TRAVEL-001#2")
    agent, _ = _agent(
        (
            SearchAction(query="travel"),
            AnswerAction(
                markdown=(
                    "Domestic meals are covered. [TRAVEL-001#1] "
                    "Travel requires approval. [TRAVEL-001#2]"
                ),
                citations=("TRAVEL-001#1", "TRAVEL-001#2"),
            ),
        ),
        _StubRetriever({"travel": (first, second)}),
    )

    result = await agent.run("Summarise travel policy")

    assert [summary.detail for summary in result.action_summaries] == [
        "Retrieved 2 chunks with top_k=3.",
        "Returned an answer with 2 citations.",
    ]


@pytest.mark.anyio
async def test_agent_preserves_answer_without_retrieval_for_metis() -> None:
    """Tool choice remains model behavior for METIS to evaluate."""
    agent, _ = _agent(
        (
            AnswerAction(
                markdown="The limit is $50. [TRAVEL-001#1]",
                citations=("TRAVEL-001#1",),
            ),
        ),
        _StubRetriever({}),
    )

    result = await agent.run("What is the limit?")

    assert result.status is AgentTurnStatus.ANSWERED
    assert result.termination_reason is TerminationReason.ANSWERED
    assert result.searches == 0
    assert result.message == "The limit is $50. [TRAVEL-001#1]"
    assert [finding.code for finding in result.validation_findings] == [
        ValidationFindingCode.CITATION_NOT_RETRIEVED
    ]


@pytest.mark.anyio
async def test_agent_accepts_direct_capability_answer_without_findings() -> None:
    """System-prompt capability answers need neither retrieval nor citations."""
    expected_action = AnswerAction(
        markdown="I support eight Northstar policy areas.",
        citations=(),
    )
    agent, provider = _agent((expected_action,), _StubRetriever({}))

    result = await agent.run("What policies do you support?")

    assert result.status is AgentTurnStatus.ANSWERED
    assert result.searches == 0
    assert result.retrieved_chunks == ()
    assert result.validation_findings == ()
    assert result.model_actions == (expected_action,)
    assert provider.requests[0].available_actions == tuple(AgentActionKind)


@pytest.mark.anyio
async def test_agent_uses_model_selected_retrieval_breadth() -> None:
    """The model may choose bounded breadth while ranking stays deterministic."""
    chunk = _chunk()
    retriever = _StubRetriever({"broad policy question": (chunk,)})
    agent, _ = _agent(
        (
            SearchAction(query="broad policy question", top_k=8),
            AnswerAction(
                markdown="The policy contains a travel limit. [TRAVEL-001#1]",
                citations=("TRAVEL-001#1",),
            ),
        ),
        retriever,
    )

    result = await agent.run("Summarise the relevant policies")

    assert result.status is AgentTurnStatus.ANSWERED
    assert retriever.calls == [("broad policy question", 8)]
    assert result.action_summaries[0].detail.endswith("top_k=8.")


@pytest.mark.anyio
async def test_agent_repairs_one_invalid_structured_action() -> None:
    """A single schema mistake gets one bounded, observable correction call."""
    provider = _RepairingProvider()
    agent = RagAgent(
        provider=provider,
        retriever=AsyncRetriever(_StubRetriever({}), CapacityLimiter(1)),
        system_prompt="Choose one valid action.",
    )

    result = await agent.run("What can you do?")

    assert result.status is AgentTurnStatus.ANSWERED
    assert result.model_calls == 2
    assert provider.requests[1].action_feedback is not None
    assert [event.kind for event in result.controller_events] == [
        ControllerEventKind.STRUCTURED_ACTION_REPAIRED
    ]


@pytest.mark.anyio
async def test_agent_fails_after_one_unsuccessful_structured_action_repair() -> None:
    """Protocol repair is bounded and cannot become an unbounded model loop."""
    provider = _AlwaysInvalidProvider()
    agent = RagAgent(
        provider=provider,
        retriever=AsyncRetriever(_StubRetriever({}), CapacityLimiter(1)),
        system_prompt="Choose one valid action.",
    )

    result = await agent.run("What can you do?")

    assert result.status is AgentTurnStatus.FAILED
    assert result.termination_reason is TerminationReason.PROVIDER_ERROR
    assert result.model_calls == 2
    assert len(provider.requests) == 2
    assert provider.requests[1].action_feedback is not None
    assert result.controller_events == ()


@pytest.mark.anyio
async def test_agent_passes_empty_retrieval_to_provider_for_a_response() -> None:
    """No evidence must remain visible to the model before it responds."""
    agent, provider = _agent(
        (
            SearchAction(query="unknown policy"),
            AnswerAction(
                markdown="No relevant evidence was found.",
                citations=(),
            ),
        ),
        _StubRetriever({"unknown policy": ()}),
    )

    result = await agent.run("Unknown question")

    assert result.status is AgentTurnStatus.ANSWERED
    assert result.termination_reason is TerminationReason.ANSWERED
    assert result.message == "No relevant evidence was found."
    assert result.citations == ()
    assert result.retrieved_chunks == ()
    assert result.model_calls == 2
    assert result.searches == 1
    assert provider.requests[1].retrieval_attempted is True
    assert provider.requests[1].retrieved_chunks == ()
    assert [summary.detail for summary in result.action_summaries] == [
        "Retrieved 0 chunks with top_k=3.",
        "Returned an answer with 0 citations.",
    ]


@pytest.mark.anyio
async def test_agent_preserves_unsupported_zero_evidence_model_response() -> None:
    """A hallucination after empty retrieval must remain observable to METIS."""
    agent, _ = _agent(
        (
            SearchAction(query="unknown policy"),
            AnswerAction(
                markdown="Employees receive a fictional $900 allowance.",
                citations=(),
            ),
        ),
        _StubRetriever({"unknown policy": ()}),
    )

    result = await agent.run("What is the unknown allowance?")

    assert result.status is AgentTurnStatus.ANSWERED
    assert result.message == "Employees receive a fictional $900 allowance."
    assert result.retrieved_chunks == ()
    assert result.citations == ()


@pytest.mark.anyio
async def test_agent_accepts_zero_citations_when_candidates_do_not_support_answer() -> (
    None
):
    """Retrieved candidates are not necessarily relevant supporting evidence."""
    chunk = _chunk()
    agent, _ = _agent(
        (
            SearchAction(query="sick leave"),
            AnswerAction(
                markdown="No relevant sick-leave evidence was found.",
                citations=(),
            ),
        ),
        _StubRetriever({"sick leave": (chunk,)}),
    )

    result = await agent.run("I need help with sick leave")

    assert result.status is AgentTurnStatus.ANSWERED
    assert result.termination_reason is TerminationReason.ANSWERED
    assert result.retrieved_chunks == (chunk,)
    assert result.citations == ()
    assert result.action_summaries[-1].detail == "Returned an answer with 0 citations."


@pytest.mark.anyio
async def test_agent_recovers_from_repeated_normalised_search() -> None:
    """A repeated query gets one constrained correction call, not an early failure."""
    chunk = _chunk()
    retriever = _StubRetriever({"travel policy": (chunk,)})
    agent, provider = _agent(
        (
            SearchAction(query="travel policy"),
            SearchAction(query="  TRAVEL   POLICY "),
            AnswerAction(
                markdown="The limit is $50. [TRAVEL-001#1]",
                citations=("TRAVEL-001#1",),
            ),
        ),
        retriever,
    )

    result = await agent.run("Travel question")

    assert result.status is AgentTurnStatus.ANSWERED
    assert result.termination_reason is TerminationReason.ANSWERED
    assert result.model_calls == 3
    assert result.searches == 1
    assert len(retriever.calls) == 1
    assert provider.requests[2].available_actions == (
        AgentActionKind.CLARIFY,
        AgentActionKind.ANSWER,
    )
    assert provider.requests[2].action_feedback is not None
    assert "repeated a query" in provider.requests[2].action_feedback
    assert [event.kind for event in result.controller_events] == [
        ControllerEventKind.REPEATED_SEARCH_REJECTED
    ]


@pytest.mark.anyio
async def test_agent_stops_on_repeated_search_without_a_correction_call(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A repeated query remains a typed failure when no model call remains."""
    chunk = _chunk()
    retriever = _StubRetriever({"travel policy": (chunk,)})
    agent, _ = _agent(
        (
            SearchAction(query="travel policy"),
            SearchAction(query="TRAVEL POLICY"),
        ),
        retriever,
        max_model_calls=2,
    )

    with caplog.at_level(logging.ERROR, logger="uvicorn.error"):
        result = await agent.run("Travel question")

    assert result.termination_reason is TerminationReason.REPEATED_SEARCH
    assert result.model_calls == 2
    assert result.searches == 1
    assert len(retriever.calls) == 1
    assert "code=repeated_search" in caplog.text


@pytest.mark.anyio
async def test_agent_records_citation_that_was_not_retrieved(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Invalid citations remain observable without replacing the answer."""
    chunk = _chunk()
    agent, _ = _agent(
        (
            SearchAction(query="travel"),
            AnswerAction(
                markdown="The limit is $80. [TRAVEL-001#2]",
                citations=("TRAVEL-001#2",),
            ),
        ),
        _StubRetriever({"travel": (chunk,)}),
    )

    with caplog.at_level(logging.WARNING, logger="uvicorn.error"):
        result = await agent.run("What is the limit?")

    assert result.status is AgentTurnStatus.ANSWERED
    assert result.termination_reason is TerminationReason.ANSWERED
    assert result.message == "The limit is $80. [TRAVEL-001#2]"
    assert result.validation_findings[0].code is (
        ValidationFindingCode.CITATION_NOT_RETRIEVED
    )
    assert result.validation_findings[0].citation_ids == ("TRAVEL-001#2",)
    assert len(caplog.records) == 1
    assert "code=citation_inconsistency" in caplog.text
    assert "citation_not_retrieved" in caplog.text
    assert "The limit is $80" not in caplog.text


@pytest.mark.anyio
async def test_agent_preserves_duplicate_citations_as_a_finding() -> None:
    """Duplicate model output remains intact for downstream evaluation."""
    chunk = _chunk()
    agent, _ = _agent(
        (
            SearchAction(query="travel"),
            AnswerAction(
                markdown="The limit is $50. [TRAVEL-001#1]",
                citations=("TRAVEL-001#1", "TRAVEL-001#1"),
            ),
        ),
        _StubRetriever({"travel": (chunk,)}),
    )

    result = await agent.run("What is the limit?")

    assert result.citations == ("TRAVEL-001#1", "TRAVEL-001#1")
    assert [finding.code for finding in result.validation_findings] == [
        ValidationFindingCode.DUPLICATE_CITATION
    ]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("markdown", "citations"),
    [
        ("The limit is $50.", ("TRAVEL-001#1",)),
        (
            "The limit is $50. [TRAVEL-001#1] [TRAVEL-001#2]",
            ("TRAVEL-001#1",),
        ),
    ],
)
async def test_agent_records_markdown_and_declared_citation_mismatch(
    markdown: str, citations: tuple[str, ...]
) -> None:
    """Citation disagreement is returned as a non-blocking finding."""
    chunk = _chunk()
    agent, _ = _agent(
        (
            SearchAction(query="travel"),
            AnswerAction(markdown=markdown, citations=citations),
        ),
        _StubRetriever({"travel": (chunk,)}),
    )

    result = await agent.run("What is the limit?")

    assert result.status is AgentTurnStatus.ANSWERED
    assert result.termination_reason is TerminationReason.ANSWERED
    assert result.validation_findings[-1].code is (
        ValidationFindingCode.CITATION_DECLARATION_MISMATCH
    )


@pytest.mark.anyio
async def test_agent_preserves_answer_with_multiple_independent_findings() -> None:
    """One imperfect answer can expose every applicable mechanical observation."""
    chunk = _chunk()
    agent, _ = _agent(
        (
            SearchAction(query="travel"),
            AnswerAction(
                markdown="The limit is $80. [TRAVEL-001#1]",
                citations=("TRAVEL-001#2", "TRAVEL-001#2"),
            ),
        ),
        _StubRetriever({"travel": (chunk,)}),
    )

    result = await agent.run("What is the limit?")

    assert result.status is AgentTurnStatus.ANSWERED
    assert result.message == "The limit is $80. [TRAVEL-001#1]"
    assert result.citations == ("TRAVEL-001#2", "TRAVEL-001#2")
    assert [finding.code for finding in result.validation_findings] == [
        ValidationFindingCode.DUPLICATE_CITATION,
        ValidationFindingCode.CITATION_NOT_RETRIEVED,
        ValidationFindingCode.CITATION_DECLARATION_MISMATCH,
    ]


@pytest.mark.anyio
async def test_agent_enforces_search_and_model_call_limits(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Independent bounds prevent unbounded tool or provider loops."""
    chunk = _chunk()
    search_limited_agent, search_limited_provider = _agent(
        (SearchAction(query="one"), SearchAction(query="two")),
        _StubRetriever({"one": (chunk,), "two": (chunk,)}),
        max_searches=1,
    )
    call_limited_agent, _ = _agent(
        (SearchAction(query="one"),),
        _StubRetriever({"one": (chunk,)}),
        max_model_calls=1,
    )

    with caplog.at_level(logging.ERROR, logger="uvicorn.error"):
        search_limited = await search_limited_agent.run("Question")
        call_limited = await call_limited_agent.run("Question")

    assert search_limited.termination_reason is TerminationReason.SEARCH_LIMIT
    assert search_limited.model_calls == 2
    assert search_limited.searches == 1
    assert search_limited_provider.requests[1].available_actions == (
        AgentActionKind.CLARIFY,
        AgentActionKind.ANSWER,
    )
    assert call_limited.termination_reason is TerminationReason.MODEL_CALL_LIMIT
    assert call_limited.model_calls == 1
    assert call_limited.searches == 1
    assert len(caplog.records) == 2
    assert "code=search_limit" in caplog.records[0].message
    assert "code=model_call_limit" in caplog.records[1].message


@pytest.mark.anyio
async def test_agent_converts_provider_and_retriever_errors_to_safe_results(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Internal exception details must not leak from the agent boundary."""
    chunk = _chunk()
    provider_agent, _ = _agent(
        (SearchAction(query="one"),),
        _StubRetriever({"one": (chunk,)}),
        max_model_calls=2,
    )
    retrieval_agent, _ = _agent(
        (SearchAction(query="one"),),
        _StubRetriever({}, error=RuntimeError("sensitive backend detail")),
    )

    with caplog.at_level(logging.ERROR, logger="uvicorn.error"):
        provider_result = await provider_agent.run("Question")
        retrieval_result = await retrieval_agent.run("Question")

    assert provider_result.termination_reason is TerminationReason.PROVIDER_ERROR
    assert "exhausted" not in provider_result.message
    assert retrieval_result.termination_reason is TerminationReason.RETRIEVAL_ERROR
    assert "sensitive" not in retrieval_result.message
    assert len(caplog.records) == 1
    assert "code=retrieval_error" in caplog.text
    assert "sensitive backend detail" not in caplog.text


@pytest.mark.anyio
@pytest.mark.parametrize("message", ["", "   "])
async def test_agent_rejects_blank_user_message(message: str) -> None:
    """Blank input must fail before consuming a scripted model action."""
    agent, provider = _agent(
        (ClarifyAction(question="Please provide a question."),),
        _StubRetriever({}),
    )

    with pytest.raises(ValueError, match="user_message"):
        await agent.run(message)
    assert provider.actions_remaining == 1


@pytest.mark.anyio
async def test_agent_accepts_unicode_user_message() -> None:
    """Unicode user input reaches the typed provider request unchanged."""
    agent, provider = _agent(
        (ClarifyAction(question="どのポリシーですか？"),),
        _StubRetriever({}),
    )

    result = await agent.run("休暇ポリシーは何ですか？")

    assert result.status is AgentTurnStatus.CLARIFICATION_REQUIRED
    assert provider.requests[0].user_message == "休暇ポリシーは何ですか？"


@pytest.mark.anyio
async def test_rag_specialist_adapter_preserves_grounded_result_and_public_steps() -> (
    None
):
    """Delegation should expose executed RAG work without changing its result."""
    chunk = _chunk()
    agent, provider = _agent(
        (
            SearchAction(query="domestic meals"),
            AnswerAction(
                markdown="The limit is $50 per day. [TRAVEL-001#1]",
                citations=("TRAVEL-001#1",),
            ),
        ),
        _StubRetriever({"domestic meals": (chunk,)}),
    )
    history = (
        ConversationMessage(
            role=ConversationRole.USER,
            content="Earlier policy question",
        ),
        ConversationMessage(
            role=ConversationRole.ASSISTANT,
            content="Earlier policy answer",
        ),
    )

    result = await RagSpecialistAdapter(agent).run(
        RagHandoffRequest(task="Explain domestic meal limits", history=history),
        session_id="session-123",
    )

    assert provider.requests[0].user_message == "Explain domestic meal limits"
    assert provider.requests[0].history == history
    assert result.citations == ("TRAVEL-001#1",)
    assert result.retrieved_chunks == (chunk,)
    assert [step.agent for step in result.execution_steps] == [
        ExecutionAgent.RAG,
        ExecutionAgent.RAG,
    ]
    assert [step.action for step in result.execution_steps] == [
        ExecutionAction.SEARCH,
        ExecutionAction.ANSWER,
    ]
    assert all(
        step.status is ExecutionStatus.COMPLETED for step in result.execution_steps
    )


@pytest.mark.anyio
async def test_rag_specialist_adapter_bypasses_standalone_trace_root() -> None:
    """Stage 13 must not create a second root before shared tracing exists."""
    recorder = _ForbiddenRootTraceRecorder()
    agent = RagAgent(
        provider=ScriptedGenerationProvider(
            (AnswerAction(markdown="Direct policy answer", citations=()),)
        ),
        retriever=AsyncRetriever(_StubRetriever({}), CapacityLimiter(1)),
        system_prompt="Search, clarify, or answer.",
        trace_recorder=recorder,
    )
    adapter = RagSpecialistAdapter(agent)

    result = await adapter.run(RagHandoffRequest(task="Policy question"))
    adapter.shutdown()

    assert result.status is AgentTurnStatus.ANSWERED
    assert recorder.shutdown_calls == 1


@pytest.mark.anyio
async def test_rag_adapter_marks_retrieval_failure_without_private_detail() -> None:
    """A failed delegated search should remain observable and safely bounded."""
    agent, _provider = _agent(
        (SearchAction(query="domestic meals"),),
        _StubRetriever({}, error=RuntimeError("private retrieval detail")),
    )

    result = await RagSpecialistAdapter(agent).run(
        RagHandoffRequest(task="Explain domestic meal limits")
    )

    assert result.termination_reason is TerminationReason.RETRIEVAL_ERROR
    assert len(result.execution_steps) == 1
    assert result.execution_steps[0].agent is ExecutionAgent.RAG
    assert result.execution_steps[0].action is ExecutionAction.SEARCH
    assert result.execution_steps[0].status is ExecutionStatus.FAILED
    assert "private" not in result.execution_steps[0].summary


@pytest.mark.anyio
async def test_rag_specialist_adapter_records_rejected_repeated_search() -> None:
    """A terminal duplicate search should appear after the completed retrieval."""
    chunk = _chunk()
    agent, _provider = _agent(
        (
            SearchAction(query="domestic meals"),
            SearchAction(query="  DOMESTIC   meals  "),
        ),
        _StubRetriever({"domestic meals": (chunk,)}),
        max_model_calls=2,
    )

    result = await RagSpecialistAdapter(agent).run(
        RagHandoffRequest(task="Explain domestic meal limits")
    )

    assert result.termination_reason is TerminationReason.REPEATED_SEARCH
    assert [step.action for step in result.execution_steps] == [
        ExecutionAction.SEARCH,
        ExecutionAction.SEARCH,
    ]
    assert [step.status for step in result.execution_steps] == [
        ExecutionStatus.COMPLETED,
        ExecutionStatus.FAILED,
    ]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"system_prompt": " "}, "system_prompt"),
        ({"max_model_calls": 0}, "max_model_calls"),
        ({"max_searches": 0}, "max_searches"),
        ({"retrieval_top_k": 0}, "retrieval_top_k"),
        ({"retrieval_top_k": 9}, "retrieval_top_k"),
    ],
)
def test_agent_rejects_invalid_construction_limits(
    overrides: dict[str, object], message: str
) -> None:
    """Invalid bounds must fail before an agent turn starts."""
    arguments: dict[str, object] = {
        "provider": ScriptedGenerationProvider((ClarifyAction(question="Question?"),)),
        "retriever": AsyncRetriever(_StubRetriever({}), CapacityLimiter(1)),
        "system_prompt": "Prompt",
        "max_model_calls": 3,
        "max_searches": 2,
        "retrieval_top_k": 3,
    }
    arguments.update(overrides)

    with pytest.raises(ValueError, match=message):
        RagAgent(**arguments)  # type: ignore[arg-type]


def test_load_system_prompt_reads_default_and_rejects_missing_or_blank(
    tmp_path: Path,
) -> None:
    """The prompt must be packaged, readable, and non-empty."""
    assert "Search before" in load_system_prompt()
    blank_path = tmp_path / "blank.md"
    blank_path.write_text("   ", encoding="utf-8")

    with pytest.raises(PromptLoadError, match="blank"):
        load_system_prompt(blank_path)
    with pytest.raises(PromptLoadError, match="could not read"):
        load_system_prompt(tmp_path / "missing.md")


def test_default_system_prompt_names_every_supported_policy() -> None:
    """Capability answers must cover the complete fixed corpus catalogue."""
    prompt = load_system_prompt()
    corpus_titles = {chunk.title for chunk in load_corpus().chunks}

    assert "chatbot assisting employees" in prompt
    assert len(corpus_titles) == 16
    assert all(title in prompt for title in corpus_titles)
