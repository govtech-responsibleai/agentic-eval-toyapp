"""Pydantic contracts for the initial service API."""

from datetime import date
from enum import StrEnum
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ApiModel(BaseModel):
    """Base contract that rejects undeclared fields and is immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class HealthStatus(StrEnum):
    """Liveness status values."""

    OK = "ok"


class ReadinessStatus(StrEnum):
    """Aggregate readiness status values."""

    READY = "ready"
    NOT_READY = "not_ready"


class CheckStatus(StrEnum):
    """Individual readiness-check status values."""

    PASS = "pass"
    FAIL = "fail"


class LifecycleMode(StrEnum):
    """Supported request lifecycle modes."""

    SYNCHRONOUS = "synchronous"


class HealthResponse(ApiModel):
    """Response returned by the liveness endpoint."""

    status: HealthStatus = HealthStatus.OK
    service: str = Field(min_length=1)
    version: str = Field(min_length=1)


class ReadinessCheck(ApiModel):
    """One named readiness check."""

    name: str = Field(min_length=1)
    status: CheckStatus
    detail: str | None = None


class ReadinessResponse(ApiModel):
    """Aggregate service readiness response."""

    status: ReadinessStatus
    checks: tuple[ReadinessCheck, ...]


class FeatureAvailability(ApiModel):
    """Feature flags advertised to target clients."""

    session_chat: bool = False
    semantic_retrieval: bool = False
    langfuse_tracing: bool = False
    controlled_scenarios: bool = False


class CapabilityResponse(ApiModel):
    """Machine-readable description of the currently available target API."""

    api_version: str = "v1"
    service_version: str = Field(min_length=1)
    lifecycle: LifecycleMode = LifecycleMode.SYNCHRONOUS
    features: FeatureAvailability = Field(default_factory=FeatureAvailability)
    actions: tuple[str, ...] = ()
    output_formats: tuple[str, ...] = ("application/json",)


class ErrorResponse(ApiModel):
    """Stable public error shape for future API routes."""

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)


class DocumentMetadata(ApiModel):
    """Stable metadata attached to one synthetic policy chunk."""

    topic: NonEmptyString
    effective_date: date


class DocumentChunk(ApiModel):
    """One immutable, independently citable corpus chunk."""

    chunk_id: NonEmptyString
    document_id: NonEmptyString
    title: NonEmptyString
    text: NonEmptyString
    metadata: DocumentMetadata

    @model_validator(mode="after")
    def validate_chunk_id_prefix(self) -> Self:
        """Keep chunk citations tied to their parent document ID."""
        if not self.chunk_id.startswith(f"{self.document_id}#"):
            raise ValueError("chunk_id must start with '<document_id>#'")
        return self


class KnowledgeCorpus(ApiModel):
    """Versioned, non-empty collection of unique document chunks."""

    version: NonEmptyString
    chunks: tuple[DocumentChunk, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_chunk_ids(self) -> Self:
        """Reject ambiguous citation identifiers."""
        chunk_ids = [chunk.chunk_id for chunk in self.chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("chunk IDs must be unique")
        return self


class RetrievedChunk(ApiModel):
    """Ranked corpus chunk returned by semantic retrieval."""

    chunk_id: NonEmptyString
    document_id: NonEmptyString
    title: NonEmptyString
    text: NonEmptyString
    metadata: DocumentMetadata
    similarity_score: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)


class RetrieverMetadata(ApiModel):
    """Reproducibility metadata for one immutable retrieval index."""

    corpus_version: NonEmptyString
    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    embedding_model_id: NonEmptyString
    embedding_model_revision: NonEmptyString
    embedding_dimension: int = Field(gt=0)
    index_type: str = "IndexFlatIP"
    chunk_count: int = Field(gt=0)
