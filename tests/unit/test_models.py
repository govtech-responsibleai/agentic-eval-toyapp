"""Tests for public API models."""

from datetime import date

import pytest
from pydantic import ValidationError

from agentic_eval_toyapp.models import (
    CapabilityResponse,
    DocumentChunk,
    DocumentMetadata,
    ErrorResponse,
    FeatureAvailability,
    HealthResponse,
    KnowledgeCorpus,
    LifecycleMode,
    RetrievedChunk,
)


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
