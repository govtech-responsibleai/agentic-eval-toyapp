"""Typed coordinator decisions and specialist handoff contracts."""

from enum import StrEnum
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import Field, StringConstraints, model_validator

from agentic_eval_toyapp.models import (
    ApiModel,
    ConversationMessage,
    FilenameString,
    NonEmptyString,
    ReportMediaType,
)

CoordinatorText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=8_000),
]


class CoordinatorActionKind(StrEnum):
    """Model-owned decisions available to the coordinator."""

    ANSWER = "answer"
    CLARIFY = "clarify"
    ROUTE_RAG = "route_rag"
    ROUTE_ANALYSIS = "route_analysis"


class CoordinatorAttachmentPage(ApiModel):
    """Extracted text retained for one source PDF page."""

    page_number: int = Field(ge=1)
    text: NonEmptyString


class CoordinatorAttachment(ApiModel):
    """Authorized report content supplied to the coordinator for one turn."""

    file_id: UUID
    filename: FilenameString
    media_type: ReportMediaType
    byte_size: int = Field(gt=0)
    extracted_text: NonEmptyString
    pages: tuple[CoordinatorAttachmentPage, ...] = ()


class CoordinatorAnswerAction(ApiModel):
    """Return a direct answer without invoking a specialist."""

    kind: Literal["answer"] = "answer"
    markdown: CoordinatorText


class CoordinatorClarifyAction(ApiModel):
    """Ask for information required before selecting a route."""

    kind: Literal["clarify"] = "clarify"
    question: CoordinatorText


class CoordinatorRouteRagAction(ApiModel):
    """Delegate one focused policy task to the RAG specialist."""

    kind: Literal["route_rag"] = "route_rag"
    task: CoordinatorText


class CoordinatorRouteAnalysisAction(ApiModel):
    """Delegate one focused task over explicitly selected reports."""

    kind: Literal["route_analysis"] = "route_analysis"
    task: CoordinatorText
    file_ids: tuple[UUID, ...] = Field(min_length=1, max_length=10)

    @model_validator(mode="after")
    def validate_unique_file_ids(self) -> Self:
        """Prevent the same report from entering one handoff more than once."""
        if len(self.file_ids) != len(set(self.file_ids)):
            raise ValueError("file_ids must be unique")
        return self


CoordinatorAction = Annotated[
    CoordinatorAnswerAction
    | CoordinatorClarifyAction
    | CoordinatorRouteRagAction
    | CoordinatorRouteAnalysisAction,
    Field(discriminator="kind"),
]


class CoordinatorRequest(ApiModel):
    """Bounded public context exposed for one coordinator decision."""

    system_prompt: NonEmptyString
    user_message: NonEmptyString
    history: tuple[ConversationMessage, ...] = ()
    attachments: tuple[CoordinatorAttachment, ...] = ()
    available_actions: tuple[CoordinatorActionKind, ...] = tuple(CoordinatorActionKind)
    action_feedback: NonEmptyString | None = None
    model_call_number: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_available_actions(self) -> Self:
        """Require a non-empty, unambiguous action vocabulary."""
        if not self.available_actions:
            raise ValueError("at least one available action is required")
        if len(self.available_actions) != len(set(self.available_actions)):
            raise ValueError("available actions must be unique")
        return self


class RagHandoffRequest(ApiModel):
    """Focused policy task passed by the coordinator to the RAG specialist."""

    task: CoordinatorText
    history: tuple[ConversationMessage, ...] = ()


class AnalysisHandoffRequest(ApiModel):
    """Focused analysis task plus content selected by the coordinator."""

    task: CoordinatorText
    history: tuple[ConversationMessage, ...] = ()
    attachments: tuple[CoordinatorAttachment, ...] = Field(min_length=1)
