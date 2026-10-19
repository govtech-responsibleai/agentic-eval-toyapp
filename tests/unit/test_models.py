"""Tests for public API models."""

from datetime import date
from typing import Any
from uuid import uuid4

import pytest
from pydantic import TypeAdapter, ValidationError

from agentic_eval_toyapp.models import (
    AgentAction,
    AgentActionKind,
    AgentTurnResult,
    AgentTurnStatus,
    AnswerAction,
    CapabilityResponse,
    ClarifyAction,
    ConversationMessage,
    ConversationRole,
    DocumentChunk,
    DocumentMetadata,
    ErrorResponse,
    ExecutionAction,
    ExecutionAgent,
    ExecutionStatus,
    ExecutionStep,
    FeatureAvailability,
    HealthResponse,
    KnowledgeCorpus,
    LifecycleMode,
    ModelRequest,
    PageReportLocator,
    ReportMediaType,
    ReportReference,
    ReportUploadResponse,
    RetrievedChunk,
    SearchAction,
    SectionReportLocator,
    SessionMessageRequest,
    SessionResponse,
    TerminationReason,
)

_AGENT_ACTION_ADAPTER: TypeAdapter[AgentAction] = TypeAdapter(AgentAction)


def test_capability_defaults_do_not_advertise_future_features() -> None:
    """Unimplemented target features must remain disabled."""
    response = CapabilityResponse(service_version="0.1.0")

    assert response.lifecycle is LifecycleMode.SYNCHRONOUS
    assert response.features == FeatureAvailability()
    assert response.actions == ()
    assert response.output_formats == ("application/json",)


def test_api_models_reject_undeclared_fields() -> None:
    """Unexpected response fields indicate contract drift."""
    with pytest.raises(ValidationError):
        HealthResponse.model_validate(
            {
                "status": "ok",
                "service": "toyapp",
                "version": "0.1.0",
                "unexpected": True,
            }
        )


@pytest.mark.parametrize(
    ("payload", "field"),
    [
        ({"code": "", "message": "failure"}, "code"),
        ({"code": "invalid_request", "message": ""}, "message"),
    ],
)
def test_error_response_rejects_empty_fields(
    payload: dict[str, str], field: str
) -> None:
    """Public errors require stable non-empty codes and messages."""
    with pytest.raises(ValidationError) as exc_info:
        ErrorResponse.model_validate(payload)

    assert field in str(exc_info.value)


def _document_chunk(*, chunk_id: str = "POLICY-001#1") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="POLICY-001",
        title="Synthetic policy",
        text="A synthetic policy statement.",
        metadata=DocumentMetadata(
            topic="testing",
            effective_date=date(2026, 1, 1),
        ),
    )


def test_document_chunk_is_immutable_and_strips_text_fields() -> None:
    """Stable corpus records must not change after index construction."""
    chunk = DocumentChunk.model_validate(
        {
            "chunk_id": " POLICY-001#1 ",
            "document_id": " POLICY-001 ",
            "title": " Synthetic policy ",
            "text": " A synthetic policy statement. ",
            "metadata": {"topic": " testing ", "effective_date": "2026-01-01"},
        }
    )

    assert chunk.chunk_id == "POLICY-001#1"
    assert chunk.metadata.topic == "testing"
    with pytest.raises(ValidationError):
        chunk.title = "Changed"  # type: ignore[misc]


def test_document_chunk_requires_matching_document_prefix() -> None:
    """Citation IDs must identify the document that owns the chunk."""
    with pytest.raises(ValidationError, match="document_id"):
        _document_chunk(chunk_id="OTHER-001#1")


def test_corpus_rejects_duplicate_chunk_ids() -> None:
    """Duplicate IDs would make citations and FAISS row mapping ambiguous."""
    chunk = _document_chunk()

    with pytest.raises(ValidationError, match="unique"):
        KnowledgeCorpus(version="v1", chunks=(chunk, chunk))


@pytest.mark.parametrize("score", [-1.01, 1.01, float("nan")])
def test_retrieved_chunk_rejects_invalid_similarity(score: float) -> None:
    """Cosine similarity must be finite and remain in [-1, 1]."""
    chunk = _document_chunk()

    with pytest.raises(ValidationError):
        RetrievedChunk(**chunk.model_dump(), similarity_score=score)


@pytest.mark.parametrize(
    ("payload", "action_type"),
    [
        ({"kind": "search", "query": "travel allowance"}, SearchAction),
        (
            {"kind": "clarify", "question": "Is the trip domestic?"},
            ClarifyAction,
        ),
        (
            {
                "kind": "answer",
                "markdown": "The limit is $50. [TRAVEL-001#1]",
                "citations": ["TRAVEL-001#1"],
            },
            AnswerAction,
        ),
    ],
)
def test_agent_action_union_uses_kind_discriminator(
    payload: dict[str, Any], action_type: type[object]
) -> None:
    """Provider output must resolve to exactly one known action schema."""
    action = _AGENT_ACTION_ADAPTER.validate_python(payload)

    assert isinstance(action, action_type)


@pytest.mark.parametrize(
    "payload",
    [
        {"kind": "search", "query": "   "},
        {"kind": "search", "query": "policy", "top_k": 0},
        {"kind": "search", "query": "policy", "top_k": 9},
        {"kind": "unknown", "value": "anything"},
    ],
)
def test_agent_action_union_rejects_invalid_payloads(payload: dict[str, Any]) -> None:
    """Blank and unknown actions cannot enter the agent loop."""
    with pytest.raises(ValidationError):
        _AGENT_ACTION_ADAPTER.validate_python(payload)


def test_answer_action_preserves_duplicate_citations() -> None:
    """Model citation mistakes must reach the agent's finding collector."""
    action = AnswerAction(
        markdown="Policy text. [POLICY-001#1]",
        citations=("POLICY-001#1", "POLICY-001#1"),
    )

    assert action.citations == ("POLICY-001#1", "POLICY-001#1")


def test_answer_action_allows_no_citations_for_post_retrieval_model_response() -> None:
    """The agent, rather than the action schema, validates retrieval context."""
    action = AnswerAction(markdown="No relevant evidence was found.", citations=())

    assert action.citations == ()


def test_model_request_rejects_evidence_without_retrieval_attempt() -> None:
    """Provider context must distinguish no search from an empty search."""
    chunk = _document_chunk()
    retrieved = RetrievedChunk(**chunk.model_dump(), similarity_score=0.9)

    with pytest.raises(ValidationError, match="attempted retrieval"):
        ModelRequest(
            system_prompt="Use evidence.",
            user_message="Question",
            retrieved_chunks=(retrieved,),
            model_call_number=2,
            remaining_searches=1,
        )


@pytest.mark.parametrize(
    ("available_actions", "remaining_searches", "message"),
    [
        ((), 1, "at least one available action"),
        (
            (AgentActionKind.ANSWER, AgentActionKind.ANSWER),
            1,
            "available actions must be unique",
        ),
        (
            (AgentActionKind.SEARCH, AgentActionKind.ANSWER),
            0,
            "search cannot be available",
        ),
    ],
)
def test_model_request_rejects_inconsistent_available_actions(
    available_actions: tuple[AgentActionKind, ...],
    remaining_searches: int,
    message: str,
) -> None:
    """Fixed action schemas must agree with the controller's current budget."""
    with pytest.raises(ValidationError, match=message):
        ModelRequest(
            system_prompt="Choose one action.",
            user_message="Question",
            available_actions=available_actions,
            model_call_number=1,
            remaining_searches=remaining_searches,
        )


def test_termination_reasons_exclude_superseded_semantic_failures() -> None:
    """Observation-first answers must not expose old strict-controller outcomes."""
    values = {reason.value for reason in TerminationReason}

    assert values.isdisjoint(
        {
            "empty_retrieval",
            "answer_without_retrieval",
            "invalid_citations",
            "citation_mismatch",
        }
    )


@pytest.mark.parametrize(
    "payload",
    [
        {
            "status": "answered",
            "message": "Answer",
            "termination_reason": "answered",
            "model_calls": 1,
            "searches": 0,
        },
        {
            "status": "clarification_required",
            "message": "Question?",
            "termination_reason": "provider_error",
            "model_calls": 1,
            "searches": 0,
        },
        {
            "status": "failed",
            "message": "Failure",
            "termination_reason": "answered",
            "model_calls": 1,
            "searches": 0,
        },
        {
            "status": "failed",
            "message": "Failure",
            "termination_reason": "search_limit",
            "model_calls": 1,
            "searches": 2,
        },
    ],
)
def test_agent_turn_result_rejects_inconsistent_outcomes(
    payload: dict[str, Any],
) -> None:
    """Public turn results cannot advertise contradictory states."""
    with pytest.raises(ValidationError):
        AgentTurnResult.model_validate(payload)


def test_agent_turn_result_accepts_grounded_answer() -> None:
    """An answered outcome identifies evidence supporting every citation."""
    chunk = _document_chunk()
    retrieved = RetrievedChunk(**chunk.model_dump(), similarity_score=0.9)

    action = AnswerAction(
        markdown="Policy statement. [POLICY-001#1]",
        citations=("POLICY-001#1",),
    )
    result = AgentTurnResult(
        status=AgentTurnStatus.ANSWERED,
        message="Policy statement. [POLICY-001#1]",
        citations=("POLICY-001#1",),
        retrieved_chunks=(retrieved,),
        termination_reason=TerminationReason.ANSWERED,
        model_calls=2,
        searches=1,
        model_actions=(action,),
    )

    assert result.citations == ("POLICY-001#1",)


def test_agent_turn_result_accepts_model_response_after_empty_retrieval() -> None:
    """Zero-evidence text remains observable after a recorded search."""
    action = AnswerAction(markdown="No relevant evidence was found.", citations=())
    result = AgentTurnResult(
        status=AgentTurnStatus.ANSWERED,
        message="No relevant evidence was found.",
        citations=(),
        retrieved_chunks=(),
        termination_reason=TerminationReason.ANSWERED,
        model_calls=2,
        searches=1,
        model_actions=(action,),
    )

    assert result.citations == ()
    assert result.searches == 1


def test_agent_turn_result_accepts_uncited_irrelevant_candidates() -> None:
    """Candidates can remain observable even when none supports the response."""
    chunk = _document_chunk()
    retrieved = RetrievedChunk(**chunk.model_dump(), similarity_score=0.9)

    action = AnswerAction(
        markdown="No supporting policy evidence was found.",
        citations=(),
    )
    result = AgentTurnResult(
        status=AgentTurnStatus.ANSWERED,
        message="No supporting policy evidence was found.",
        citations=(),
        retrieved_chunks=(retrieved,),
        termination_reason=TerminationReason.ANSWERED,
        model_calls=2,
        searches=1,
        model_actions=(action,),
    )

    assert result.retrieved_chunks == (retrieved,)
    assert result.citations == ()


@pytest.mark.parametrize("message", ["", "   ", "x" * 8_001])
def test_session_message_request_rejects_invalid_lengths(message: str) -> None:
    """Chat requests must be non-blank and bounded before entering memory."""
    with pytest.raises(ValidationError):
        SessionMessageRequest(message=message)


def test_session_message_request_strips_and_preserves_unicode() -> None:
    """User text should retain Unicode while removing boundary whitespace."""
    request = SessionMessageRequest(message="  休暇ポリシーは何ですか？  ")

    assert request.message == "休暇ポリシーは何ですか？"
    assert request.file_ids == ()


def test_session_message_request_accepts_one_unique_file_id() -> None:
    """One optional report reference should preserve its UUID exactly."""
    file_id = uuid4()

    request = SessionMessageRequest(message="Analyse this", file_ids=(file_id,))

    assert request.file_ids == (file_id,)


def test_session_message_request_rejects_invalid_file_ids() -> None:
    """Duplicate and unbounded attachment lists must fail schema validation."""
    file_id = uuid4()
    with pytest.raises(ValidationError):
        SessionMessageRequest(
            message="Analyse this",
            file_ids=(file_id, file_id),
        )
    with pytest.raises(ValidationError):
        SessionMessageRequest(
            message="Analyse this",
            file_ids=tuple(uuid4() for _ in range(11)),
        )


def test_report_upload_metadata_requires_media_specific_page_count() -> None:
    """PDF page counts and non-PDF metadata must remain internally consistent."""
    file_id = uuid4()

    pdf = ReportUploadResponse(
        file_id=file_id,
        filename="report.pdf",
        media_type=ReportMediaType.PDF,
        byte_size=10,
        extracted_text_length=5,
        sha256="0" * 64,
        page_count=2,
    )

    assert pdf.page_count == 2
    with pytest.raises(ValidationError, match="page_count"):
        ReportUploadResponse(
            file_id=file_id,
            filename="report.pdf",
            media_type=ReportMediaType.PDF,
            byte_size=10,
            extracted_text_length=5,
            sha256="0" * 64,
        )
    with pytest.raises(ValidationError, match="page_count"):
        ReportUploadResponse(
            file_id=file_id,
            filename="report.txt",
            media_type=ReportMediaType.TEXT,
            byte_size=10,
            extracted_text_length=5,
            sha256="0" * 64,
            page_count=1,
        )


def test_report_references_keep_page_and_section_locators_typed() -> None:
    """Report evidence must remain distinct from policy chunk citations."""
    file_id = uuid4()
    page_reference = ReportReference(
        file_id=file_id,
        locator=PageReportLocator(page_number=3),
        excerpt="Revenue increased by 12%.",
    )
    section_reference = ReportReference(
        file_id=file_id,
        locator=SectionReportLocator(section="Risk summary"),
        excerpt="Supplier concentration remains high.",
    )

    assert page_reference.locator.kind == "page"
    assert section_reference.locator.kind == "section"
    with pytest.raises(ValidationError):
        PageReportLocator(page_number=0)
    with pytest.raises(ValidationError):
        ReportReference(
            file_id=file_id,
            locator=SectionReportLocator(section="Risk"),
            excerpt=" ",
        )


def test_agent_result_requires_contiguous_execution_step_sequence() -> None:
    """Execution timelines should be stable, ordered, and one-indexed."""
    action = AnswerAction(markdown="Answer", citations=())
    step = ExecutionStep(
        sequence=1,
        agent=ExecutionAgent.RAG,
        action=ExecutionAction.ANSWER,
        status=ExecutionStatus.COMPLETED,
        summary="Returned an answer.",
        duration_ms=1.5,
    )
    result = AgentTurnResult(
        status=AgentTurnStatus.ANSWERED,
        message="Answer",
        termination_reason=TerminationReason.ANSWERED,
        model_calls=1,
        searches=0,
        model_actions=(action,),
        execution_steps=(step,),
    )

    assert result.execution_steps == (step,)
    assert result.report_references == ()
    with pytest.raises(ValidationError, match="contiguous"):
        AgentTurnResult(
            status=AgentTurnStatus.ANSWERED,
            message="Answer",
            termination_reason=TerminationReason.ANSWERED,
            model_calls=1,
            searches=0,
            model_actions=(action,),
            execution_steps=(step.model_copy(update={"sequence": 2}),),
        )


@pytest.mark.parametrize(
    "messages",
    [
        (ConversationMessage(role=ConversationRole.USER, content="Question"),),
        (
            ConversationMessage(role=ConversationRole.ASSISTANT, content="Answer"),
            ConversationMessage(role=ConversationRole.USER, content="Question"),
        ),
        (
            ConversationMessage(role=ConversationRole.USER, content="Question"),
            ConversationMessage(
                role=ConversationRole.USER,
                content="Another question",
            ),
        ),
    ],
)
def test_session_response_rejects_incomplete_or_misordered_turns(
    messages: tuple[ConversationMessage, ...],
) -> None:
    """Public history must contain complete chronological conversation turns."""
    with pytest.raises(ValidationError):
        SessionResponse(
            session_id=uuid4(),
            messages=messages,
            retained_turns=1,
        )


def test_session_response_accepts_empty_and_complete_histories() -> None:
    """A new session is empty and later snapshots contain complete pairs."""
    session_id = uuid4()
    empty = SessionResponse(session_id=session_id, retained_turns=0)
    complete = SessionResponse(
        session_id=session_id,
        messages=(
            ConversationMessage(role=ConversationRole.USER, content="Question"),
            ConversationMessage(role=ConversationRole.ASSISTANT, content="Answer"),
        ),
        retained_turns=1,
    )

    assert empty.messages == ()
    assert complete.retained_turns == 1
