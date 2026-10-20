"""Coordinator-only report specialist with one model-backed analysis call."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, cast, runtime_checkable
from uuid import UUID

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from agentic_eval_toyapp.analysis_models import (
    AnalysisAction,
    AnalysisActionKind,
    AnalysisAnswerAction,
    AnalysisClarifyAction,
    AnalysisModelRequest,
)
from agentic_eval_toyapp.models import (
    AgentTurnResult,
    AgentTurnStatus,
    AnswerAction,
    ClarifyAction,
    ExecutionAction,
    ExecutionAgent,
    ExecutionStatus,
    ExecutionStep,
    PageReportLocator,
    ReportReference,
    SectionReportLocator,
    TerminationReason,
    ValidationFinding,
    ValidationFindingCode,
)
from agentic_eval_toyapp.orchestration_models import AnalysisHandoffRequest
from agentic_eval_toyapp.providers import (
    GenerationProviderError,
    InvalidGenerationActionError,
    StructuredCompletionClient,
    make_object_schemas_strict,
)
from agentic_eval_toyapp.report_sources import SourceBlock, source_blocks

DEFAULT_ANALYSIS_PROMPT_PATH = Path(__file__).parent / "prompts" / "analysis.md"


def load_analysis_prompt(path: Path = DEFAULT_ANALYSIS_PROMPT_PATH) -> str:
    """Load versioned report instructions or fail during application startup."""
    try:
        prompt = path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise ValueError("could not read analysis prompt") from error
    if not prompt:
        raise ValueError("analysis prompt must not be blank")
    return prompt


@runtime_checkable
class AnalysisGenerationProvider(Protocol):
    """Select one validated report action from coordinator-supplied content."""

    @property
    def model_id(self) -> str: ...

    @property
    def revision(self) -> str: ...

    def generate_action(self, request: AnalysisModelRequest) -> AnalysisAction: ...


class _FlatReportReference(BaseModel):
    """Gateway-friendly form of a typed report locator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    file_id: UUID
    locator_kind: str
    page_number: int | None
    section: str | None
    excerpt: str

    @model_validator(mode="after")
    def validate_locator(self) -> "_FlatReportReference":
        if self.locator_kind == "page":
            if (
                self.page_number is None
                or self.page_number < 1
                or self.section is not None
            ):
                raise ValueError("page locator requires only a positive page number")
        elif self.locator_kind == "section":
            if not self.section or self.page_number is not None:
                raise ValueError("section locator requires only a section name")
        else:
            raise ValueError("unknown report locator kind")
        if not self.excerpt.strip() or len(self.excerpt) > 2_000:
            raise ValueError("report excerpt must be non-empty and bounded")
        return self

    def to_domain_reference(self) -> ReportReference:
        locator = (
            PageReportLocator(page_number=cast(int, self.page_number))
            if self.locator_kind == "page"
            else SectionReportLocator(section=cast(str, self.section))
        )
        return ReportReference(
            file_id=self.file_id,
            locator=locator,
            excerpt=self.excerpt,
        )


class _FlatAnalysisAction(BaseModel):
    """Flat strict JSON schema compatible with the model gateway."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: AnalysisActionKind
    markdown: str | None
    question: str | None
    references: tuple[_FlatReportReference, ...]

    @model_validator(mode="after")
    def validate_fields(self) -> "_FlatAnalysisAction":
        if self.kind is AnalysisActionKind.ANSWER:
            if not self.markdown or self.question is not None:
                raise ValueError("answer requires only markdown and references")
        elif not self.question or self.markdown is not None or self.references:
            raise ValueError("clarify requires only a question")
        return self

    def to_domain_action(self) -> AnalysisAction:
        if self.kind is AnalysisActionKind.CLARIFY:
            return AnalysisClarifyAction(question=cast(str, self.question))
        return AnalysisAnswerAction(
            markdown=cast(str, self.markdown),
            references=tuple(
                reference.to_domain_reference() for reference in self.references
            ),
        )


class OpenAIAnalysisGenerationProvider:
    """Use the existing OpenAI-compatible transport for report decisions."""

    def __init__(
        self,
        *,
        model: str,
        revision: str,
        temperature: float,
        max_tokens: int,
        client: StructuredCompletionClient,
    ) -> None:
        if not model.strip() or not revision.strip():
            raise ValueError("model and revision must not be blank")
        if not 0 <= temperature <= 2 or max_tokens < 1:
            raise ValueError("invalid model generation settings")
        self._model_id = model.strip()
        self._revision = revision.strip()
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = client
        self._schema = _FlatAnalysisAction.model_json_schema()
        make_object_schemas_strict(self._schema)

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def revision(self) -> str:
        return self._revision

    def generate_action(self, request: AnalysisModelRequest) -> AnalysisAction:
        """Send all selected report sources and validate the model result."""
        context: dict[str, object] = {
            "task": request.task,
            "conversation_history": [
                message.model_dump(mode="json") for message in request.history
            ],
            "source_blocks": [
                {
                    "file_id": str(block.file_id),
                    "locator": block.locator.model_dump(mode="json"),
                    "text": block.text,
                }
                for block in request.blocks
            ],
            "response_field_rules": {
                "answer": "Set markdown and references; set question to null.",
                "clarify": "Set question; set markdown to null and references to [].",
                "reference": "Use file_id, locator_kind, page_number or section, and an exact source excerpt.",
            },
            "model_call_number": request.model_call_number,
        }
        try:
            content = self._client.create(
                model=self._model_id,
                messages=(
                    {"role": "system", "content": request.system_prompt},
                    {
                        "role": "user",
                        "content": json.dumps(context, ensure_ascii=False),
                    },
                ),
                response_schema=self._schema,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        except GenerationProviderError:
            raise
        except Exception as error:
            raise GenerationProviderError("live analysis request failed") from error
        try:
            return _FlatAnalysisAction.model_validate_json(content).to_domain_action()
        except (ValidationError, ValueError) as error:
            raise InvalidGenerationActionError(
                "live model returned an invalid analysis action"
            ) from error


class DeterministicAnalysisGenerationProvider:
    """Extract exact report snippets for network-free mechanics tests."""

    @property
    def model_id(self) -> str:
        return "deterministic-analysis-provider"

    @property
    def revision(self) -> str:
        return "v1"

    def generate_action(self, request: AnalysisModelRequest) -> AnalysisAction:
        statements: list[str] = []
        references: list[ReportReference] = []
        for block in request.blocks:
            excerpt = block.text.strip()[:180].strip()
            if not excerpt:
                continue
            statements.append(f"- {excerpt}")
            references.append(
                ReportReference(
                    file_id=block.file_id,
                    locator=block.locator,
                    excerpt=excerpt,
                )
            )
            if len(statements) == 4:
                break
        if not statements:
            return AnalysisClarifyAction(
                question="Please provide a report with usable text."
            )
        return AnalysisAnswerAction(
            markdown="\n".join(statements)[:6_000],
            references=tuple(references),
        )


class ReportAnalysisAgent:
    """Analyse only content supplied by the coordinator handoff."""

    def __init__(
        self,
        *,
        provider: AnalysisGenerationProvider,
        system_prompt: str,
    ) -> None:
        if not system_prompt.strip():
            raise ValueError("analysis prompt must not be blank")
        self._provider = provider
        self._system_prompt = system_prompt.strip()

    def run(
        self,
        request: AnalysisHandoffRequest,
        *,
        session_id: str | None = None,
    ) -> AgentTurnResult:
        """Analyse all selected report sources in one model call."""
        del session_id  # Stage 16 will own one nested coordinator trace.
        source = source_blocks(request.attachments)
        if not source:
            question = "Please provide a report with usable text."
            return AgentTurnResult(
                status=AgentTurnStatus.CLARIFICATION_REQUIRED,
                message=question,
                model_actions=(ClarifyAction(question=question),),
                execution_steps=(
                    self._step(
                        1, ExecutionAction.CLARIFY, "Requested readable report content."
                    ),
                ),
                termination_reason=TerminationReason.CLARIFICATION_REQUIRED,
                model_calls=0,
                searches=0,
            )

        steps: list[ExecutionStep] = [
            self._step(
                1,
                ExecutionAction.INSPECT_CONTENT,
                (
                    f"Inspected {len(request.attachments)} "
                    f"{'report' if len(request.attachments) == 1 else 'reports'} "
                    f"with {len(source)} source "
                    f"{'span' if len(source) == 1 else 'spans'}."
                ),
            )
        ]
        try:
            action = self._provider.generate_action(
                AnalysisModelRequest(
                    system_prompt=self._system_prompt,
                    task=request.task,
                    history=request.history,
                    blocks=source,
                    model_call_number=1,
                )
            )
        except GenerationProviderError:
            return self._failure(
                "The analysis provider could not produce an action.",
                TerminationReason.PROVIDER_ERROR,
                1,
                tuple(steps),
            )
        if isinstance(action, AnalysisClarifyAction):
            steps.append(
                self._step(
                    len(steps) + 1,
                    ExecutionAction.CLARIFY,
                    "Requested additional report instructions.",
                )
            )
            return AgentTurnResult(
                status=AgentTurnStatus.CLARIFICATION_REQUIRED,
                message=action.question,
                model_actions=(ClarifyAction(question=action.question),),
                execution_steps=tuple(steps),
                termination_reason=TerminationReason.CLARIFICATION_REQUIRED,
                model_calls=1,
                searches=0,
            )

        final, findings = self._validate_references(action, source)

        steps.append(
            self._step(
                len(steps) + 1,
                ExecutionAction.ANSWER,
                (
                    f"Returned report findings with {len(final.references)} "
                    f"validated {'reference' if len(final.references) == 1 else 'references'}."
                ),
            )
        )
        return AgentTurnResult(
            status=AgentTurnStatus.ANSWERED,
            message=final.markdown,
            model_actions=(AnswerAction(markdown=final.markdown, citations=()),),
            report_references=final.references,
            validation_findings=findings,
            execution_steps=tuple(steps),
            termination_reason=TerminationReason.ANSWERED,
            model_calls=1,
            searches=0,
        )

    @staticmethod
    def _validate_references(
        action: AnalysisAnswerAction,
        blocks: Sequence[SourceBlock],
    ) -> tuple[AnalysisAnswerAction, tuple[ValidationFinding, ...]]:
        valid: list[ReportReference] = []
        findings: list[ValidationFinding] = []
        for reference in action.references:
            matching = tuple(
                block
                for block in blocks
                if block.file_id == reference.file_id
                and block.locator == reference.locator
            )
            if not matching:
                findings.append(
                    ValidationFinding(
                        code=ValidationFindingCode.REPORT_REFERENCE_INVALID,
                        detail="A model-declared report reference was not in the supplied evidence.",
                    )
                )
            elif not any(reference.excerpt in block.text for block in matching):
                findings.append(
                    ValidationFinding(
                        code=ValidationFindingCode.REPORT_EXCERPT_NOT_FOUND,
                        detail="A model-declared excerpt did not match its source locator.",
                    )
                )
            elif reference not in valid:
                valid.append(reference)
        return action.model_copy(update={"references": tuple(valid)}), tuple(findings)

    @staticmethod
    def _step(sequence: int, action: ExecutionAction, summary: str) -> ExecutionStep:
        return ExecutionStep(
            sequence=sequence,
            agent=ExecutionAgent.ANALYSIS,
            action=action,
            status=ExecutionStatus.COMPLETED,
            summary=summary,
        )

    @staticmethod
    def _failure(
        message: str,
        reason: TerminationReason,
        calls: int,
        steps: tuple[ExecutionStep, ...],
    ) -> AgentTurnResult:
        return AgentTurnResult(
            status=AgentTurnStatus.FAILED,
            message=message,
            execution_steps=steps,
            termination_reason=reason,
            model_calls=calls,
            searches=0,
        )
