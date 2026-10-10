"""Typed local-only contracts for controlled trace generation."""

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agentic_eval_toyapp.models import AgentTurnResult, NonEmptyString


class TraceCaseModel(BaseModel):
    """Strict immutable base for version-controlled trace-case data."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class CaseVariant(StrEnum):
    """Supported sides of a controlled metric pair."""

    CONTROL = "control"
    DEFECT = "defect"


class ControlledCase(TraceCaseModel):
    """One local input and effective answer used to generate a trace."""

    case_id: NonEmptyString
    variant: CaseVariant
    user_message: NonEmptyString
    replacement_markdown: NonEmptyString


class MetricCaseSet(TraceCaseModel):
    """Runtime cases for one supported METIS metric."""

    metric: NonEmptyString
    cases: tuple[ControlledCase, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_pair(self) -> "MetricCaseSet":
        """Require one matched control/defect pair with opaque unique IDs."""
        identifiers = [case.case_id for case in self.cases]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("trace case IDs must be unique")
        variants = [case.variant for case in self.cases]
        if variants.count(CaseVariant.CONTROL) != 1:
            raise ValueError("exactly one control case is required")
        if variants.count(CaseVariant.DEFECT) != 1:
            raise ValueError("exactly one defect case is required")
        messages = {case.user_message for case in self.cases}
        if len(messages) != 1:
            raise ValueError("control and defect cases must use the same input")
        return self


class GroundTruthLabel(TraceCaseModel):
    """Expected binary metric outcome retained outside Langfuse."""

    case_id: NonEmptyString
    expected_pass: bool
    rationale: NonEmptyString


class MetricGroundTruth(TraceCaseModel):
    """Local expected outcomes for one metric's controlled pair."""

    metric: NonEmptyString
    labels: tuple[GroundTruthLabel, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_labels(self) -> "MetricGroundTruth":
        """Reject ambiguous or duplicated expected labels."""
        identifiers = [label.case_id for label in self.labels]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("ground-truth case IDs must be unique")
        outcomes = {label.expected_pass for label in self.labels}
        if outcomes != {False, True}:
            raise ValueError("ground truth must contain one pass and one fail")
        return self


class MetricBundle(TraceCaseModel):
    """Validated runtime cases joined to their local ground truth."""

    cases: MetricCaseSet
    ground_truth: MetricGroundTruth

    @model_validator(mode="after")
    def validate_alignment(self) -> "MetricBundle":
        """Require exact metric and case-ID agreement across both files."""
        if self.cases.metric != self.ground_truth.metric:
            raise ValueError("case and ground-truth metrics do not match")
        case_ids = {case.case_id for case in self.cases.cases}
        label_ids = {label.case_id for label in self.ground_truth.labels}
        if case_ids != label_ids:
            raise ValueError("case and ground-truth IDs do not match")
        return self


class CompletionCapture(TraceCaseModel):
    """Private record of one live response and its effective replacement."""

    original_response: NonEmptyString
    effective_response: NonEmptyString
    altered: bool


class TraceCaseRun(TraceCaseModel):
    """Local run manifest that maps a realistic trace to ground truth."""

    generated_at: datetime
    metric: NonEmptyString
    case_id: NonEmptyString
    variant: CaseVariant
    expected_pass: bool
    expected_rationale: NonEmptyString
    session_id: UUID
    user_message: NonEmptyString
    controlled_response_applied: bool
    result: AgentTurnResult
    model_completions: tuple[CompletionCapture, ...] = ()
