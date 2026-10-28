"""Report analysis boundary, provider actions, and grounded references."""

import json
import logging
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import pytest
from pydantic import ValidationError

from agentic_eval_toyapp.analysis import (
    AnalysisGenerationProvider,
    DeterministicAnalysisGenerationProvider,
    OpenAIAnalysisGenerationProvider,
    ReportAnalysisAgent,
    load_analysis_prompt,
)
from agentic_eval_toyapp.analysis_models import (
    AnalysisAction,
    AnalysisAnswerAction,
    AnalysisClarifyAction,
    AnalysisModelRequest,
)
from agentic_eval_toyapp.diagnostics import CompletionDiagnostics
from agentic_eval_toyapp.models import (
    AgentTurnStatus,
    PageReportLocator,
    ReportMediaType,
    ReportReference,
    SectionReportLocator,
    TerminationReason,
    ValidationFindingCode,
)
from agentic_eval_toyapp.orchestration_models import (
    AnalysisHandoffRequest,
    CoordinatorAttachment,
    CoordinatorAttachmentPage,
)
from agentic_eval_toyapp.providers import (
    GenerationProviderError,
    InvalidGenerationActionError,
    StructuredCompletion,
)
from agentic_eval_toyapp.report_sources import SourceBlock


def _attachment(
    text: str, *, media_type: ReportMediaType = ReportMediaType.TEXT
) -> CoordinatorAttachment:
    return CoordinatorAttachment(
        file_id=uuid4(),
        filename="report.md"
        if media_type is ReportMediaType.MARKDOWN
        else "report.txt",
        media_type=media_type,
        byte_size=len(text.encode()),
        extracted_text=text,
    )


def _agent(
    provider: AnalysisGenerationProvider,
) -> ReportAnalysisAgent:
    return ReportAnalysisAgent(
        provider=provider,
        system_prompt=load_analysis_prompt(),
    )


class _ScriptedProvider:
    model_id = "scripted-analysis"
    revision = "v1"

    def __init__(self, actions: tuple[AnalysisAction, ...]) -> None:
        self.actions = list(actions)
        self.requests: list[AnalysisModelRequest] = []

    async def generate_action(self, request: AnalysisModelRequest) -> AnalysisAction:
        self.requests.append(request)
        if not self.actions:
            raise GenerationProviderError("private provider detail")
        return self.actions.pop(0)


class _RecordingClient:
    def __init__(
        self,
        response: dict[str, object],
        *,
        error: Exception | None = None,
        diagnostics: CompletionDiagnostics | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.diagnostics = diagnostics
        self.calls: list[dict[str, object]] = []

    async def create(
        self,
        *,
        model: str,
        messages: tuple[dict[str, str], ...],
        response_schema: dict[str, Any],
        temperature: float,
        max_tokens: int,
    ) -> str | StructuredCompletion:
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "response_schema": response_schema,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        if self.error is not None:
            raise self.error
        content = json.dumps(self.response)
        if self.diagnostics is not None:
            return StructuredCompletion(content, self.diagnostics)
        return content


@pytest.mark.anyio
async def test_short_report_returns_one_grounded_answer_without_policy_search() -> None:
    """One short report should need exactly one specialist model call."""
    report = _attachment("Revenue rose 18%. Supplier share is 62%.")
    result = await _agent(DeterministicAnalysisGenerationProvider()).run(
        AnalysisHandoffRequest(task="Summarise findings", attachments=(report,))
    )

    assert result.status is AgentTurnStatus.ANSWERED
    assert result.model_calls == 1
    assert result.searches == 0
    assert result.citations == ()
    assert result.retrieved_chunks == ()
    assert len(result.report_references) == 1
    assert result.report_references[0].excerpt in report.extracted_text
    assert [step.action.value for step in result.execution_steps] == [
        "inspect_content",
        "answer",
    ]


@pytest.mark.anyio
async def test_long_report_is_supplied_in_full_to_one_model_call() -> None:
    """The specialist neither truncates nor splits the checked-in report."""
    report_text = (
        (Path(__file__).parents[1] / "fixtures" / "test_report.md")
        .read_text(encoding="utf-8")
        .strip()
    )
    report = _attachment(report_text, media_type=ReportMediaType.MARKDOWN)
    provider = _ScriptedProvider((AnalysisAnswerAction(markdown="Report reviewed."),))

    result = await _agent(provider).run(
        AnalysisHandoffRequest(task="Find main results", attachments=(report,))
    )

    assert result.status is AgentTurnStatus.ANSWERED
    assert result.model_calls == len(provider.requests) == 1
    blocks = provider.requests[0].blocks
    assert [
        block.locator.section for block in blocks if block.locator.kind == "section"
    ][-2:] == [
        "Data quality and stated limitations",
        "Main findings and follow-up questions",
    ]
    assert "No statistical test" in blocks[-2].text
    assert "broader decision" in blocks[-1].text
    assert result.searches == 0


@pytest.mark.anyio
async def test_direct_answer_above_old_chunk_summary_limit_is_accepted() -> None:
    """The removed map limit must not reject a valid direct answer."""
    report = _attachment("A longer report finding.")
    answer = "Finding. " * 160
    provider = _ScriptedProvider((AnalysisAnswerAction(markdown=answer),))

    result = await _agent(provider).run(
        AnalysisHandoffRequest(task="Summarise", attachments=(report,))
    )

    assert len(answer) > 1_200
    assert result.status is AgentTurnStatus.ANSWERED
    assert result.message == answer.strip()
    assert result.model_calls == 1


@pytest.mark.anyio
async def test_long_answer_and_thirteen_references_fit_analysis_contract() -> None:
    """A useful report answer must not fail the old 6,000/12 bounds."""
    report_id = uuid4()
    answer = "A" * 7_351
    references = [
        {
            "file_id": str(report_id),
            "locator_kind": "section",
            "page_number": None,
            "section": "Document",
            "excerpt": f"Finding {index}.",
        }
        for index in range(13)
    ]
    client = _RecordingClient(
        {
            "kind": "answer",
            "markdown": answer,
            "question": None,
            "references": references,
        }
    )
    provider = OpenAIAnalysisGenerationProvider(
        model="platform-model",
        revision="provider-managed",
        temperature=0,
        max_tokens=16_384,
        client=client,
    )

    action = await provider.generate_action(
        AnalysisModelRequest(
            system_prompt="Analyse faithfully.",
            task="Summarise the main findings",
            blocks=(
                SourceBlock(
                    file_id=report_id,
                    locator=SectionReportLocator(section="Document"),
                    text=" ".join(f"Finding {index}." for index in range(13)),
                ),
            ),
            model_call_number=1,
        )
    )

    assert isinstance(action, AnalysisAnswerAction)
    assert action.markdown == answer
    assert len(action.references) == 13


def test_analysis_answer_and_reference_bounds_remain_finite() -> None:
    """A generous demo allowance must still reject unbounded provider output."""
    reference = ReportReference(
        file_id=uuid4(),
        locator=SectionReportLocator(section="Document"),
        excerpt="Source finding.",
    )

    assert AnalysisAnswerAction(markdown="A" * 32_000).markdown == "A" * 32_000
    accepted = AnalysisAnswerAction(markdown="Answer", references=(reference,) * 64)
    assert len(accepted.references) == 64
    with pytest.raises(ValidationError, match="markdown"):
        AnalysisAnswerAction(markdown="A" * 32_001)
    with pytest.raises(ValidationError, match="references"):
        AnalysisAnswerAction(markdown="Answer", references=(reference,) * 65)
    with pytest.raises(ValidationError, match="question"):
        AnalysisClarifyAction(question="Q" * 6_001)


@pytest.mark.anyio
async def test_invalid_reference_is_not_presented_as_valid_evidence(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Wrong file IDs and fabricated excerpts produce findings, not citations."""
    report = _attachment("Revenue rose 18%.")
    wrong = ReportReference(
        file_id=uuid4(),
        locator=SectionReportLocator(section="Document"),
        excerpt="Revenue rose 18%.",
    )
    missing = ReportReference(
        file_id=report.file_id,
        locator=SectionReportLocator(section="Document"),
        excerpt="Profit rose 99%.",
    )
    provider = _ScriptedProvider(
        (
            AnalysisAnswerAction(
                markdown="Profit rose 99%.", references=(wrong, missing)
            ),
        )
    )

    with caplog.at_level(logging.WARNING, logger="uvicorn.error"):
        result = await _agent(provider).run(
            AnalysisHandoffRequest(task="Analyse", attachments=(report,))
        )

    assert result.status is AgentTurnStatus.ANSWERED
    assert result.message == "Profit rose 99%."
    assert result.report_references == ()
    assert {finding.code for finding in result.validation_findings} == {
        ValidationFindingCode.REPORT_REFERENCE_INVALID,
        ValidationFindingCode.REPORT_EXCERPT_NOT_FOUND,
    }
    assert len(caplog.records) == 1
    assert "code=report_reference_inconsistency" in caplog.text
    assert "report_reference_invalid" in caplog.text
    assert "report_excerpt_not_found" in caplog.text
    assert "Revenue rose 18%" not in caplog.text


@pytest.mark.anyio
async def test_live_analysis_logs_invalid_schema_without_report_content(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A malformed model action is reported without exposing source text."""
    provider = OpenAIAnalysisGenerationProvider(
        model="platform-model",
        revision="provider-managed",
        temperature=0,
        max_tokens=512,
        client=_RecordingClient(
            {"kind": "invalid", "secret": "private report"},
            diagnostics=CompletionDiagnostics(
                finish_reason="length",
                input_tokens=4_200,
                output_tokens=512,
                reasoning_tokens=400,
                total_tokens=4_712,
            ),
        ),
    )
    request = AnalysisModelRequest(
        system_prompt="Analyse",
        task="Summarise",
        blocks=(
            SourceBlock(
                file_id=uuid4(),
                locator=SectionReportLocator(section="Document"),
                text="private report source text",
            ),
        ),
        model_call_number=1,
    )

    with (
        caplog.at_level(logging.ERROR, logger="uvicorn.error"),
        pytest.raises(InvalidGenerationActionError),
    ):
        await provider.generate_action(request)

    assert len(caplog.records) == 1
    assert "agent=analysis" in caplog.text
    assert "code=invalid_schema" in caplog.text
    assert "finish_reason=length" in caplog.text
    assert "input_tokens=4200 output_tokens=512 reasoning_tokens=400" in caplog.text
    assert "private report" not in caplog.text


@pytest.mark.anyio
async def test_analysis_clarification_and_provider_failure() -> None:
    """Each terminal path should have one honest call and a safe public reason."""
    report = _attachment("Finding.")
    clarification = await _agent(
        _ScriptedProvider((AnalysisClarifyAction(question="Which period?"),))
    ).run(AnalysisHandoffRequest(task="Compare periods", attachments=(report,)))
    failure = await _agent(_ScriptedProvider(())).run(
        AnalysisHandoffRequest(task="Summarise", attachments=(report,))
    )
    assert clarification.status is AgentTurnStatus.CLARIFICATION_REQUIRED
    assert clarification.message == "Which period?"
    assert clarification.model_calls == 1
    assert failure.termination_reason is TerminationReason.PROVIDER_ERROR
    assert failure.model_calls == 1
    assert "private" not in failure.message


@pytest.mark.anyio
async def test_pdf_reference_requires_supplied_page_and_exact_excerpt() -> None:
    """A page number outside the extracted report must not become evidence."""
    report = CoordinatorAttachment(
        file_id=uuid4(),
        filename="report.pdf",
        media_type=ReportMediaType.PDF,
        byte_size=20,
        extracted_text="Risk is high.",
        pages=(CoordinatorAttachmentPage(page_number=3, text="Risk is high."),),
    )
    wrong_page = ReportReference(
        file_id=report.file_id,
        locator=PageReportLocator(page_number=2),
        excerpt="Risk is high.",
    )
    provider = _ScriptedProvider(
        (AnalysisAnswerAction(markdown="Risk is high.", references=(wrong_page,)),)
    )

    result = await _agent(provider).run(
        AnalysisHandoffRequest(task="Summarise", attachments=(report,))
    )

    assert result.report_references == ()
    assert (
        result.validation_findings[0].code
        is ValidationFindingCode.REPORT_REFERENCE_INVALID
    )


@pytest.mark.anyio
async def test_conflicting_sections_reach_one_direct_model_request() -> None:
    """The planner must preserve contradictory report claims for model judgment."""
    report = _attachment(
        "# Operations\nRevenue rose 18%.\n# Finance\nRevenue fell 4%.",
        media_type=ReportMediaType.MARKDOWN,
    )
    provider = _ScriptedProvider(
        (AnalysisAnswerAction(markdown="The two sections conflict."),)
    )

    result = await _agent(provider).run(
        AnalysisHandoffRequest(task="Identify contradictions", attachments=(report,))
    )

    assert result.message == "The two sections conflict."
    assert [
        block.locator.section
        for block in provider.requests[0].blocks
        if block.locator.kind == "section"
    ] == ["Operations", "Finance"]
    assert "rose 18%" in provider.requests[0].blocks[0].text
    assert "fell 4%" in provider.requests[0].blocks[1].text


@pytest.mark.anyio
async def test_blank_pdf_page_handoff_requests_usable_report_text() -> None:
    """An anomalous empty extraction must not trigger a model call."""
    report = CoordinatorAttachment(
        file_id=uuid4(),
        filename="report.pdf",
        media_type=ReportMediaType.PDF,
        byte_size=20,
        extracted_text="Metadata only",
        pages=(),
    )
    provider = _ScriptedProvider(())

    result = await _agent(provider).run(
        AnalysisHandoffRequest(task="Summarise", attachments=(report,))
    )

    assert result.status is AgentTurnStatus.CLARIFICATION_REQUIRED
    assert result.model_calls == 0
    assert provider.requests == []


def test_analysis_model_request_requires_source_content() -> None:
    """The provider must never receive a report-less analysis request."""
    with pytest.raises(ValidationError, match="blocks"):
        AnalysisModelRequest(
            system_prompt="Analyse faithfully.",
            task="Summarise",
            blocks=(),
            model_call_number=1,
        )


@pytest.mark.anyio
async def test_openai_analysis_provider_uses_strict_schema_and_validates_action() -> (
    None
):
    """The live adapter should send exact source spans and parse a typed reply."""
    report_id = uuid4()
    client = _RecordingClient(
        {
            "kind": "answer",
            "markdown": "Revenue rose 18%.",
            "question": None,
            "references": [
                {
                    "file_id": str(report_id),
                    "locator_kind": "section",
                    "page_number": None,
                    "section": "Document",
                    "excerpt": "Revenue rose 18%.",
                }
            ],
        }
    )
    provider = OpenAIAnalysisGenerationProvider(
        model="platform-model",
        revision="provider-managed",
        temperature=0,
        max_tokens=512,
        client=client,
    )

    action = await provider.generate_action(
        AnalysisModelRequest(
            system_prompt="Analyse faithfully.",
            task="Find main result",
            blocks=(
                SourceBlock(
                    file_id=report_id,
                    locator=SectionReportLocator(section="Document"),
                    text="Revenue rose 18%.",
                ),
            ),
            model_call_number=1,
        )
    )

    assert isinstance(action, AnalysisAnswerAction)
    assert action.references[0].file_id == report_id
    schema = cast(dict[str, Any], client.calls[0]["response_schema"])
    messages = cast(tuple[dict[str, str], ...], client.calls[0]["messages"])
    assert schema["additionalProperties"] is False
    assert "Revenue rose 18%." in messages[1]["content"]


def test_analysis_prompt_explains_untrusted_input_and_grounding() -> None:
    prompt = load_analysis_prompt()
    assert "untrusted data" in prompt
    assert "exact" in prompt
