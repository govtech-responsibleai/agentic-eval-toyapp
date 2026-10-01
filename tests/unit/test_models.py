"""Tests for public API models."""

from datetime import date
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from agentic_eval_toyapp.models import (
    AgentAction,
    AgentTurnResult,
    AgentTurnStatus,
    AnswerAction,
    CapabilityResponse,
    ClarifyAction,
    DocumentChunk,
    DocumentMetadata,
    ErrorResponse,
    FeatureAvailability,
    HealthResponse,
    KnowledgeCorpus,
    LifecycleMode,
    RetrievedChunk,
    SearchAction,
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
        {"kind": "answer", "markdown": "No citation", "citations": []},
        {"kind": "unknown", "value": "anything"},
    ],
)
def test_agent_action_union_rejects_invalid_payloads(payload: dict[str, Any]) -> None:
    """Blank, uncited, and unknown actions cannot enter the agent loop."""
    with pytest.raises(ValidationError):
        _AGENT_ACTION_ADAPTER.validate_python(payload)


def test_answer_action_rejects_duplicate_citations() -> None:
    """One evidence chunk should appear once in structured citations."""
    with pytest.raises(ValidationError, match="unique"):
        AnswerAction(
            markdown="Policy text. [POLICY-001#1]",
            citations=("POLICY-001#1", "POLICY-001#1"),
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

    result = AgentTurnResult(
        status=AgentTurnStatus.ANSWERED,
        message="Policy statement. [POLICY-001#1]",
        citations=("POLICY-001#1",),
        retrieved_chunks=(retrieved,),
        termination_reason=TerminationReason.ANSWERED,
        model_calls=2,
        searches=1,
    )

    assert result.citations == ("POLICY-001#1",)
