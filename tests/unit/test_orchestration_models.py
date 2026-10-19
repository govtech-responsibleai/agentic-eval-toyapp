"""Tests for coordinator actions, attachments, and specialist handoffs."""

from uuid import uuid4

import pytest
from pydantic import TypeAdapter, ValidationError

from agentic_eval_toyapp.models import ReportMediaType
from agentic_eval_toyapp.orchestration_models import (
    AnalysisHandoffRequest,
    CoordinatorAction,
    CoordinatorActionKind,
    CoordinatorAttachment,
    CoordinatorRequest,
    CoordinatorRouteAnalysisAction,
)

_ACTION_ADAPTER: TypeAdapter[CoordinatorAction] = TypeAdapter(CoordinatorAction)


def _attachment() -> CoordinatorAttachment:
    return CoordinatorAttachment(
        file_id=uuid4(),
        filename="report.md",
        media_type=ReportMediaType.MARKDOWN,
        byte_size=20,
        extracted_text="Revenue increased.",
    )


def test_coordinator_action_union_rejects_unknown_and_malformed_actions() -> None:
    """Only the fixed action vocabulary and its exact fields are accepted."""
    with pytest.raises(ValidationError):
        _ACTION_ADAPTER.validate_python({"kind": "unknown", "task": "Work"})
    with pytest.raises(ValidationError):
        _ACTION_ADAPTER.validate_python({"kind": "answer", "markdown": " "})
    with pytest.raises(ValidationError):
        _ACTION_ADAPTER.validate_python(
            {"kind": "route_analysis", "task": "Analyse", "file_ids": []}
        )


def test_analysis_route_rejects_duplicate_files() -> None:
    """A report may enter one analysis handoff at most once."""
    file_id = uuid4()

    with pytest.raises(ValidationError, match="unique"):
        CoordinatorRouteAnalysisAction(
            task="Analyse the report",
            file_ids=(file_id, file_id),
        )


def test_coordinator_request_requires_unique_available_actions() -> None:
    """The provider must receive an unambiguous non-empty action set."""
    with pytest.raises(ValidationError, match="at least one"):
        CoordinatorRequest(
            system_prompt="Coordinate",
            user_message="Help",
            available_actions=(),
            model_call_number=1,
        )
    with pytest.raises(ValidationError, match="unique"):
        CoordinatorRequest(
            system_prompt="Coordinate",
            user_message="Help",
            available_actions=(
                CoordinatorActionKind.ANSWER,
                CoordinatorActionKind.ANSWER,
            ),
            model_call_number=1,
        )


def test_analysis_handoff_requires_validated_content() -> None:
    """Analysis cannot run without an authorized non-empty report."""
    attachment = _attachment()
    handoff = AnalysisHandoffRequest(
        task="Summarise the report",
        attachments=(attachment,),
    )

    assert handoff.attachments == (attachment,)
    with pytest.raises(ValidationError):
        AnalysisHandoffRequest(task="Summarise the report", attachments=())
