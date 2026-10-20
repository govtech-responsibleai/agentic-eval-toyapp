"""Pydantic contracts for the initial service API."""

from datetime import date
from enum import StrEnum
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
FilenameString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=255),
]
BoundedSummaryString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=500),
]
ReportExcerptString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=2_000),
]
UserMessageString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=8_000),
]


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
    file_uploads: bool = False


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


class ConversationRole(StrEnum):
    """Roles retained in bounded public conversation history."""

    USER = "user"
    ASSISTANT = "assistant"


class ConversationMessage(ApiModel):
    """One public conversation message without hidden reasoning."""

    role: ConversationRole
    content: NonEmptyString


class AgentActionKind(StrEnum):
    """Actions available to the bounded RAG agent."""

    SEARCH = "search"
    CLARIFY = "clarify"
    ANSWER = "answer"


class SearchAction(ApiModel):
    """Request semantic retrieval for one focused query."""

    kind: Literal["search"] = "search"
    query: NonEmptyString
    top_k: int = Field(default=3, ge=1, le=8)


class ClarifyAction(ApiModel):
    """Ask the user for information required to proceed."""

    kind: Literal["clarify"] = "clarify"
    question: NonEmptyString


class AnswerAction(ApiModel):
    """Return Markdown and the model's declared chunk citations unchanged."""

    kind: Literal["answer"] = "answer"
    markdown: NonEmptyString
    citations: tuple[NonEmptyString, ...]


AgentAction = Annotated[
    SearchAction | ClarifyAction | AnswerAction,
    Field(discriminator="kind"),
]


class ModelRequest(ApiModel):
    """Information exposed to a generation provider for one action."""

    system_prompt: NonEmptyString
    user_message: NonEmptyString
    history: tuple[ConversationMessage, ...] = ()
    retrieved_chunks: tuple[RetrievedChunk, ...] = ()
    retrieval_attempted: bool = False
    action_feedback: NonEmptyString | None = None
    available_actions: tuple[AgentActionKind, ...] = (
        AgentActionKind.SEARCH,
        AgentActionKind.CLARIFY,
        AgentActionKind.ANSWER,
    )
    model_call_number: int = Field(ge=1)
    remaining_searches: int = Field(ge=0)
    default_top_k: int = Field(default=3, ge=1, le=8)

    @model_validator(mode="after")
    def validate_retrieval_state(self) -> Self:
        """Keep retrieval context and the current action budget consistent."""
        if self.retrieved_chunks and not self.retrieval_attempted:
            raise ValueError("retrieved chunks require an attempted retrieval")
        if not self.available_actions:
            raise ValueError("at least one available action is required")
        if len(self.available_actions) != len(set(self.available_actions)):
            raise ValueError("available actions must be unique")
        if (
            AgentActionKind.SEARCH in self.available_actions
            and self.remaining_searches == 0
        ):
            raise ValueError("search cannot be available when no searches remain")
        return self


class AgentActionSummary(ApiModel):
    """Observable action record that excludes private chain-of-thought."""

    action: AgentActionKind
    detail: NonEmptyString


class ValidationFindingCode(StrEnum):
    """Deterministic citation inconsistencies preserved for evaluation."""

    CITATION_NOT_RETRIEVED = "citation_not_retrieved"
    CITATION_DECLARATION_MISMATCH = "citation_declaration_mismatch"
    DUPLICATE_CITATION = "duplicate_citation"
    REPORT_REFERENCE_INVALID = "report_reference_invalid"
    REPORT_EXCERPT_NOT_FOUND = "report_excerpt_not_found"


class ValidationFinding(ApiModel):
    """One non-blocking, mechanically detected citation inconsistency."""

    code: ValidationFindingCode
    detail: NonEmptyString
    citation_ids: tuple[NonEmptyString, ...] = ()


class ControllerEventKind(StrEnum):
    """Recoverable controller interventions visible to evaluators."""

    REPEATED_SEARCH_REJECTED = "repeated_search_rejected"
    STRUCTURED_ACTION_REPAIRED = "structured_action_repaired"


class ControllerEvent(ApiModel):
    """One bounded controller intervention without private reasoning."""

    kind: ControllerEventKind
    detail: NonEmptyString


class ReportMediaType(StrEnum):
    """Supported, explicitly validated report media types."""

    TEXT = "text/plain"
    MARKDOWN = "text/markdown"
    PDF = "application/pdf"


class ReportUploadResponse(ApiModel):
    """Safe metadata returned after one report is processed and retained."""

    file_id: UUID
    filename: FilenameString
    media_type: ReportMediaType
    byte_size: int = Field(gt=0)
    extracted_text_length: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    page_count: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_page_count(self) -> Self:
        """Only PDF reports expose a page count."""
        if self.media_type is ReportMediaType.PDF and self.page_count is None:
            raise ValueError("PDF reports require page_count")
        if self.media_type is not ReportMediaType.PDF and self.page_count is not None:
            raise ValueError("non-PDF reports cannot declare page_count")
        return self


class PageReportLocator(ApiModel):
    """One one-indexed PDF page supporting an analysis claim."""

    kind: Literal["page"] = "page"
    page_number: int = Field(ge=1)


class SectionReportLocator(ApiModel):
    """One named text or Markdown section supporting an analysis claim."""

    kind: Literal["section"] = "section"
    section: NonEmptyString


ReportLocator = Annotated[
    PageReportLocator | SectionReportLocator,
    Field(discriminator="kind"),
]


class ReportReference(ApiModel):
    """A source reference into an uploaded report, separate from RAG citations."""

    file_id: UUID
    locator: ReportLocator
    excerpt: ReportExcerptString


class ExecutionAgent(StrEnum):
    """Public agent roles that may appear in a Phase 2 execution timeline."""

    COORDINATOR = "coordinator"
    RAG = "rag"
    ANALYSIS = "analysis"


class ExecutionAction(StrEnum):
    """Fixed public action vocabulary for the Phase 2 execution timeline."""

    INSPECT_INPUT = "inspect_input"
    ANSWER = "answer"
    CLARIFY = "clarify"
    ROUTE_RAG = "route_rag"
    ROUTE_ANALYSIS = "route_analysis"
    SEARCH = "search"
    INSPECT_CONTENT = "inspect_content"
    RETURN_RESPONSE = "return_response"


class ExecutionStatus(StrEnum):
    """Public completion states for one execution step."""

    COMPLETED = "completed"
    FAILED = "failed"


class ExecutionStep(ApiModel):
    """One ordered public action without hidden reasoning or prompt content."""

    sequence: int = Field(ge=1)
    agent: ExecutionAgent
    action: ExecutionAction
    status: ExecutionStatus
    summary: BoundedSummaryString
    duration_ms: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)


class AgentTurnStatus(StrEnum):
    """Public outcome categories for one agent turn."""

    ANSWERED = "answered"
    CLARIFICATION_REQUIRED = "clarification_required"
    FAILED = "failed"


class TerminationReason(StrEnum):
    """Stable reason why a bounded agent turn stopped."""

    ANSWERED = "answered"
    CLARIFICATION_REQUIRED = "clarification_required"
    REPEATED_SEARCH = "repeated_search"
    MODEL_CALL_LIMIT = "model_call_limit"
    SEARCH_LIMIT = "search_limit"
    PROVIDER_ERROR = "provider_error"
    RETRIEVAL_ERROR = "retrieval_error"
    SPECIALIST_UNAVAILABLE = "specialist_unavailable"
    INVALID_HANDOFF = "invalid_handoff"
    DELEGATION_ERROR = "delegation_error"


class AgentTurnResult(ApiModel):
    """Typed result returned by the bounded agent loop."""

    status: AgentTurnStatus
    message: NonEmptyString
    citations: tuple[NonEmptyString, ...] = ()
    retrieved_chunks: tuple[RetrievedChunk, ...] = ()
    action_summaries: tuple[AgentActionSummary, ...] = ()
    controller_events: tuple[ControllerEvent, ...] = ()
    validation_findings: tuple[ValidationFinding, ...] = ()
    model_actions: tuple[AgentAction, ...] = ()
    execution_steps: tuple[ExecutionStep, ...] = ()
    report_references: tuple[ReportReference, ...] = ()
    termination_reason: TerminationReason
    model_calls: int = Field(ge=0)
    searches: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_outcome_consistency(self) -> Self:
        """Keep status, evidence, citations, and counters internally consistent."""
        if self.searches > self.model_calls:
            raise ValueError("searches cannot exceed model_calls")
        if tuple(step.sequence for step in self.execution_steps) != tuple(
            range(1, len(self.execution_steps) + 1)
        ):
            raise ValueError(
                "execution step sequence must be contiguous and one-indexed"
            )

        if self.status is AgentTurnStatus.ANSWERED:
            if self.termination_reason is not TerminationReason.ANSWERED:
                raise ValueError("answered status requires answered termination reason")
            if not self.model_actions or not isinstance(
                self.model_actions[-1], AnswerAction
            ):
                raise ValueError("answered results require the final answer action")
            final_answer = self.model_actions[-1]
            if (
                self.message != final_answer.markdown
                or self.citations != final_answer.citations
            ):
                raise ValueError(
                    "answered result must preserve the final answer action"
                )
        elif self.status is AgentTurnStatus.CLARIFICATION_REQUIRED:
            if self.termination_reason is not TerminationReason.CLARIFICATION_REQUIRED:
                raise ValueError(
                    "clarification status requires clarification termination reason"
                )
            if self.citations:
                raise ValueError("clarification results cannot contain citations")
            if not self.model_actions or not isinstance(
                self.model_actions[-1], ClarifyAction
            ):
                raise ValueError(
                    "clarification results require the final clarification action"
                )
            final_clarification = self.model_actions[-1]
            if self.message != final_clarification.question:
                raise ValueError(
                    "clarification result must preserve the final clarification action"
                )
        else:
            if self.termination_reason in {
                TerminationReason.ANSWERED,
                TerminationReason.CLARIFICATION_REQUIRED,
            }:
                raise ValueError("failed status requires a failure termination reason")
            if self.citations:
                raise ValueError("failed results cannot contain citations")
            if self.validation_findings:
                raise ValueError("failed results cannot contain validation findings")
        return self


class SessionMessageRequest(ApiModel):
    """One bounded user message submitted to an existing session."""

    message: UserMessageString
    file_ids: tuple[UUID, ...] = Field(default=(), max_length=10)

    @model_validator(mode="after")
    def validate_unique_file_ids(self) -> Self:
        """Reject ambiguous duplicate attachment references."""
        if len(self.file_ids) != len(set(self.file_ids)):
            raise ValueError("file_ids must be unique")
        return self


class SessionResponse(ApiModel):
    """Safe public snapshot of one in-memory conversation session."""

    session_id: UUID
    messages: tuple[ConversationMessage, ...] = ()
    retained_turns: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_complete_turn_history(self) -> Self:
        """Require complete user/assistant pairs in chronological order."""
        if len(self.messages) != self.retained_turns * 2:
            raise ValueError("retained_turns must match complete message pairs")
        for index, message in enumerate(self.messages):
            expected_role = (
                ConversationRole.USER if index % 2 == 0 else ConversationRole.ASSISTANT
            )
            if message.role is not expected_role:
                raise ValueError("messages must alternate user and assistant roles")
        return self


class SessionMessageResponse(ApiModel):
    """Result of one agent turn within a named application session."""

    session_id: UUID
    result: AgentTurnResult
