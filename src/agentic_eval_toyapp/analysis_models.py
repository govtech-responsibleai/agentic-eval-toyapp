"""Typed model requests and decisions for bounded report analysis."""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, StringConstraints

from agentic_eval_toyapp.models import (
    ApiModel,
    ConversationMessage,
    NonEmptyString,
    ReportReference,
)
from agentic_eval_toyapp.report_sources import SourceBlock

AnalysisAnswerText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=32_000)
]
AnalysisQuestionText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=6_000)
]


class AnalysisActionKind(StrEnum):
    """Available public analysis decisions."""

    ANSWER = "answer"
    CLARIFY = "clarify"


class AnalysisAnswerAction(ApiModel):
    """Grounded Markdown plus model-selected report references."""

    kind: Literal["answer"] = "answer"
    markdown: AnalysisAnswerText
    references: tuple[ReportReference, ...] = Field(default=(), max_length=64)


class AnalysisClarifyAction(ApiModel):
    """Question required to complete the requested report analysis."""

    kind: Literal["clarify"] = "clarify"
    question: AnalysisQuestionText


AnalysisAction = Annotated[
    AnalysisAnswerAction | AnalysisClarifyAction,
    Field(discriminator="kind"),
]


class AnalysisModelRequest(ApiModel):
    """One report-analysis request containing all coordinator-selected sources."""

    system_prompt: NonEmptyString
    task: NonEmptyString
    history: tuple[ConversationMessage, ...] = ()
    blocks: tuple[SourceBlock, ...] = Field(min_length=1)
    model_call_number: int = Field(ge=1)
