"""Bounded single-agent loop for grounded retrieval-augmented answers."""

import logging
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from agentic_eval_toyapp.diagnostics import log_failure
from agentic_eval_toyapp.models import (
    AgentAction,
    AgentActionKind,
    AgentActionSummary,
    AgentTurnResult,
    AgentTurnStatus,
    ClarifyAction,
    ControllerEvent,
    ControllerEventKind,
    ConversationMessage,
    ExecutionAction,
    ExecutionAgent,
    ExecutionStatus,
    ExecutionStep,
    ModelRequest,
    RetrievedChunk,
    SearchAction,
    TerminationReason,
    ValidationFinding,
    ValidationFindingCode,
)
from agentic_eval_toyapp.multi_agent_tracing import delegated_rag_trace
from agentic_eval_toyapp.orchestration_models import RagHandoffRequest
from agentic_eval_toyapp.providers import (
    GenerationProvider,
    GenerationProviderError,
    InvalidGenerationActionError,
)
from agentic_eval_toyapp.reports import StoredReport
from agentic_eval_toyapp.retrieval import AsyncSearcher
from agentic_eval_toyapp.tracing import NoOpTraceRecorder, TraceRecorder, TurnTrace

DEFAULT_SYSTEM_PROMPT_PATH = Path(__file__).parent / "prompts" / "system.md"
_INLINE_CITATION_PATTERN = re.compile(r"\[([A-Z0-9-]+#\d+)\]")


class PromptLoadError(ValueError):
    """Raised when the system prompt cannot be read or is blank."""


def load_system_prompt(path: Path = DEFAULT_SYSTEM_PROMPT_PATH) -> str:
    """Read the version-controlled system prompt."""
    try:
        prompt = path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise PromptLoadError(f"could not read system prompt: {path}") from error
    if not prompt:
        raise PromptLoadError(f"system prompt is blank: {path}")
    return prompt


class RagAgent:
    """Execute validated actions within strict per-turn limits."""

    def __init__(
        self,
        *,
        provider: GenerationProvider,
        retriever: AsyncSearcher,
        system_prompt: str,
        max_model_calls: int = 4,
        max_searches: int = 2,
        retrieval_top_k: int = 3,
        trace_recorder: TraceRecorder | None = None,
    ) -> None:
        if not system_prompt.strip():
            raise ValueError("system_prompt must not be blank")
        if max_model_calls < 1:
            raise ValueError("max_model_calls must be at least 1")
        if max_searches < 1:
            raise ValueError("max_searches must be at least 1")
        if not 1 <= retrieval_top_k <= 8:
            raise ValueError("retrieval_top_k must be between 1 and 8")
        self._provider = provider
        self._retriever = retriever
        self._system_prompt = system_prompt.strip()
        self._max_model_calls = max_model_calls
        self._max_searches = max_searches
        self._retrieval_top_k = retrieval_top_k
        self._trace_recorder = trace_recorder or NoOpTraceRecorder()

    @property
    def tracing_enabled(self) -> bool:
        """Return whether this agent exports turn observations."""
        return self._trace_recorder.enabled

    @property
    def model_id(self) -> str:
        """Identify the provider used by the RAG loop in nested traces."""
        return self._provider.model_id

    def shutdown(self) -> None:
        """Flush and close the process-level tracing dependency."""
        self._trace_recorder.shutdown()

    async def run(
        self,
        user_message: str,
        *,
        history: Sequence[ConversationMessage] = (),
        session_id: str | None = None,
        reports: Sequence[StoredReport] = (),
    ) -> AgentTurnResult:
        """Run one bounded turn without retaining session state."""
        del reports  # The RAG-only mode never treats uploads as policy evidence.
        stripped_message = user_message.strip()
        if not stripped_message:
            raise ValueError("user_message must not be blank")

        async with self._trace_recorder.start_turn(
            session_id=session_id,
            user_message=stripped_message,
            history_message_count=len(history),
        ) as trace:
            result = await self._execute_turn(
                stripped_message,
                history=history,
                trace=trace,
            )
            trace.complete(result)
            return result

    async def run_delegated(
        self,
        user_message: str,
        *,
        history: Sequence[ConversationMessage] = (),
    ) -> AgentTurnResult:
        """Run beneath a coordinator without creating a standalone trace root."""
        stripped_message = user_message.strip()
        if not stripped_message:
            raise ValueError("user_message must not be blank")

        active_trace = delegated_rag_trace()
        if active_trace is not None:
            return await self._execute_turn(
                stripped_message,
                history=history,
                trace=active_trace,
            )

        no_op_recorder = NoOpTraceRecorder()
        async with no_op_recorder.start_turn(
            session_id=None,
            user_message=stripped_message,
            history_message_count=len(history),
        ) as trace:
            return await self._execute_turn(
                stripped_message,
                history=history,
                trace=trace,
            )

    async def _execute_turn(
        self,
        user_message: str,
        *,
        history: Sequence[ConversationMessage],
        trace: TurnTrace,
    ) -> AgentTurnResult:
        """Execute the bounded action loop inside an active turn trace."""
        stripped_message = user_message

        model_calls = 0
        searches = 0
        seen_queries: set[str] = set()
        retrieved_by_id: dict[str, RetrievedChunk] = {}
        summaries: list[AgentActionSummary] = []
        controller_events: list[ControllerEvent] = []
        model_actions: list[AgentAction] = []
        search_disabled = False
        action_feedback: str | None = None
        repair_used = False
        repair_pending = False

        while model_calls < self._max_model_calls:
            model_calls += 1
            remaining_searches = max(self._max_searches - searches, 0)
            available_actions = tuple(
                action
                for action in AgentActionKind
                if not (
                    action is AgentActionKind.SEARCH
                    and (remaining_searches == 0 or search_disabled)
                )
            )
            request = ModelRequest(
                system_prompt=self._system_prompt,
                user_message=stripped_message,
                history=tuple(history),
                retrieved_chunks=tuple(retrieved_by_id.values()),
                retrieval_attempted=searches > 0,
                action_feedback=action_feedback,
                available_actions=available_actions,
                model_call_number=model_calls,
                remaining_searches=remaining_searches,
                default_top_k=self._retrieval_top_k,
            )
            try:
                with trace.model_action(request) as model_observation:
                    try:
                        action = await self._provider.generate_action(request)
                    except InvalidGenerationActionError:
                        model_observation.fail("invalid_structured_action")
                        raise
                    except GenerationProviderError:
                        model_observation.fail("generation_provider_error")
                        raise
                    else:
                        model_observation.succeed(action)
            except InvalidGenerationActionError:
                if repair_used or model_calls >= self._max_model_calls:
                    return self._failure(
                        message="The generation provider could not produce a valid action.",
                        reason=TerminationReason.PROVIDER_ERROR,
                        model_calls=model_calls,
                        searches=searches,
                        retrieved_chunks=tuple(retrieved_by_id.values()),
                        summaries=tuple(summaries),
                        controller_events=tuple(controller_events),
                        model_actions=tuple(model_actions),
                    )
                repair_used = True
                repair_pending = True
                action_feedback = (
                    "The previous response did not match the required action schema. "
                    "Return exactly one valid action using only the available actions."
                )
                continue
            except GenerationProviderError:
                return self._failure(
                    message="The generation provider could not produce an action.",
                    reason=TerminationReason.PROVIDER_ERROR,
                    model_calls=model_calls,
                    searches=searches,
                    retrieved_chunks=tuple(retrieved_by_id.values()),
                    summaries=tuple(summaries),
                    controller_events=tuple(controller_events),
                    model_actions=tuple(model_actions),
                )

            if repair_pending:
                controller_events.append(
                    ControllerEvent(
                        kind=ControllerEventKind.STRUCTURED_ACTION_REPAIRED,
                        detail="Accepted a valid action after one schema correction call.",
                    )
                )
                repair_pending = False
                action_feedback = None

            model_actions.append(action)

            if isinstance(action, SearchAction):
                if searches >= self._max_searches:
                    return self._failure(
                        message="The search limit was reached before an answer was produced.",
                        reason=TerminationReason.SEARCH_LIMIT,
                        model_calls=model_calls,
                        searches=searches,
                        retrieved_chunks=tuple(retrieved_by_id.values()),
                        summaries=tuple(summaries),
                        controller_events=tuple(controller_events),
                        model_actions=tuple(model_actions),
                    )
                normalised_query = " ".join(action.query.casefold().split())
                if normalised_query in seen_queries:
                    if model_calls >= self._max_model_calls:
                        return self._failure(
                            message="The agent repeated a search and stopped safely.",
                            reason=TerminationReason.REPEATED_SEARCH,
                            model_calls=model_calls,
                            searches=searches,
                            retrieved_chunks=tuple(retrieved_by_id.values()),
                            summaries=tuple(summaries),
                            controller_events=tuple(controller_events),
                            model_actions=tuple(model_actions),
                        )
                    search_disabled = True
                    controller_events.append(
                        ControllerEvent(
                            kind=ControllerEventKind.REPEATED_SEARCH_REJECTED,
                            detail=(
                                "Rejected a repeated normalized search query without "
                                "executing retrieval."
                            ),
                        )
                    )
                    action_feedback = (
                        "The previous search action repeated a query already used "
                        "in this turn. Use the existing retrieval results and choose "
                        "either answer or clarify."
                    )
                    continue
                seen_queries.add(normalised_query)
                searches += 1
                try:
                    results = await self._retriever.search(
                        action.query,
                        top_k=action.top_k,
                    )
                except Exception as error:
                    log_failure(
                        event="retrieval",
                        agent="rag",
                        operation="search_knowledge_base",
                        code="retrieval_error",
                        attempt=searches,
                        error=error,
                    )
                    trace.record_search(
                        query=action.query,
                        top_k=action.top_k,
                        results=None,
                        error_code="retrieval_error",
                    )
                    return self._failure(
                        message="The knowledge-base search failed.",
                        reason=TerminationReason.RETRIEVAL_ERROR,
                        model_calls=model_calls,
                        searches=searches,
                        retrieved_chunks=tuple(retrieved_by_id.values()),
                        summaries=tuple(summaries),
                        controller_events=tuple(controller_events),
                        model_actions=tuple(model_actions),
                    )
                trace.record_search(
                    query=action.query,
                    top_k=action.top_k,
                    results=results,
                )
                summaries.append(
                    AgentActionSummary(
                        action=AgentActionKind.SEARCH,
                        detail=(
                            f"Retrieved {len(results)} "
                            f"{'chunk' if len(results) == 1 else 'chunks'} "
                            f"with top_k={action.top_k}."
                        ),
                    )
                )
                for result in results:
                    retrieved_by_id.setdefault(result.chunk_id, result)
                continue

            if isinstance(action, ClarifyAction):
                trace.record_clarification(action.question)
                summaries.append(
                    AgentActionSummary(
                        action=AgentActionKind.CLARIFY,
                        detail="Returned a clarification question.",
                    )
                )
                return AgentTurnResult(
                    status=AgentTurnStatus.CLARIFICATION_REQUIRED,
                    message=action.question,
                    retrieved_chunks=tuple(retrieved_by_id.values()),
                    action_summaries=tuple(summaries),
                    controller_events=tuple(controller_events),
                    model_actions=tuple(model_actions),
                    termination_reason=TerminationReason.CLARIFICATION_REQUIRED,
                    model_calls=model_calls,
                    searches=searches,
                )

            findings: list[ValidationFinding] = []
            citation_counts = Counter(action.citations)
            duplicate_citations = tuple(
                citation for citation, count in citation_counts.items() if count > 1
            )
            if duplicate_citations:
                findings.append(
                    ValidationFinding(
                        code=ValidationFindingCode.DUPLICATE_CITATION,
                        detail="The model declared one or more citations repeatedly.",
                        citation_ids=duplicate_citations,
                    )
                )
            declared_citations = set(action.citations)
            available_citations = set(retrieved_by_id)
            missing_citations = tuple(
                dict.fromkeys(
                    citation
                    for citation in action.citations
                    if citation not in available_citations
                )
            )
            if missing_citations:
                findings.append(
                    ValidationFinding(
                        code=ValidationFindingCode.CITATION_NOT_RETRIEVED,
                        detail="The model cited IDs absent from retrieved candidates.",
                        citation_ids=missing_citations,
                    )
                )
            inline_citations = set(_INLINE_CITATION_PATTERN.findall(action.markdown))
            if inline_citations != declared_citations:
                findings.append(
                    ValidationFinding(
                        code=ValidationFindingCode.CITATION_DECLARATION_MISMATCH,
                        detail=(
                            "Inline Markdown citations differed from the structured "
                            "citation list."
                        ),
                        citation_ids=tuple(
                            sorted(
                                inline_citations.symmetric_difference(
                                    declared_citations
                                )
                            )
                        ),
                    )
                )
            summaries.append(
                AgentActionSummary(
                    action=AgentActionKind.ANSWER,
                    detail=(
                        f"Returned an answer with {len(action.citations)} "
                        f"{'citation' if len(action.citations) == 1 else 'citations'}."
                    ),
                )
            )
            trace.record_validation(
                citations=action.citations,
                candidate_ids=tuple(retrieved_by_id),
                findings=tuple(findings),
            )
            if findings:
                log_failure(
                    event="validation",
                    agent="rag",
                    operation="validate_answer",
                    code="citation_inconsistency",
                    level=logging.WARNING,
                    attempt=model_calls,
                    finding_codes=tuple(finding.code.value for finding in findings),
                )
            return AgentTurnResult(
                status=AgentTurnStatus.ANSWERED,
                message=action.markdown,
                citations=action.citations,
                retrieved_chunks=tuple(retrieved_by_id.values()),
                action_summaries=tuple(summaries),
                controller_events=tuple(controller_events),
                validation_findings=tuple(findings),
                model_actions=tuple(model_actions),
                termination_reason=TerminationReason.ANSWERED,
                model_calls=model_calls,
                searches=searches,
            )

        return self._failure(
            message="The model-call limit was reached before the turn completed.",
            reason=TerminationReason.MODEL_CALL_LIMIT,
            model_calls=model_calls,
            searches=searches,
            retrieved_chunks=tuple(retrieved_by_id.values()),
            summaries=tuple(summaries),
            controller_events=tuple(controller_events),
            model_actions=tuple(model_actions),
        )

    @staticmethod
    def _failure(
        *,
        message: str,
        reason: TerminationReason,
        model_calls: int,
        searches: int,
        retrieved_chunks: tuple[RetrievedChunk, ...],
        summaries: tuple[AgentActionSummary, ...],
        controller_events: tuple[ControllerEvent, ...],
        model_actions: tuple[AgentAction, ...],
    ) -> AgentTurnResult:
        """Build one safe failure result without internal exception details."""
        if reason not in (
            TerminationReason.PROVIDER_ERROR,
            TerminationReason.RETRIEVAL_ERROR,
        ):
            log_failure(
                event="agent_stop",
                agent="rag",
                operation="execute_turn",
                code=reason.value,
                attempt=model_calls,
            )
        return AgentTurnResult(
            status=AgentTurnStatus.FAILED,
            message=message,
            retrieved_chunks=retrieved_chunks,
            action_summaries=summaries,
            controller_events=controller_events,
            model_actions=model_actions,
            termination_reason=reason,
            model_calls=model_calls,
            searches=searches,
        )


class RagSpecialistAdapter:
    """Adapt the existing RAG loop to the coordinator handoff contract."""

    _ACTION_MAP = {
        AgentActionKind.SEARCH: ExecutionAction.SEARCH,
        AgentActionKind.CLARIFY: ExecutionAction.CLARIFY,
        AgentActionKind.ANSWER: ExecutionAction.ANSWER,
    }
    _FAILED_SEARCH_SUMMARIES = {
        TerminationReason.RETRIEVAL_ERROR: "Knowledge-base retrieval failed.",
        TerminationReason.REPEATED_SEARCH: "Rejected a repeated RAG search.",
        TerminationReason.SEARCH_LIMIT: (
            "Rejected a RAG search after the search budget was exhausted."
        ),
    }

    def __init__(self, agent: RagAgent) -> None:
        self._agent = agent

    async def run(
        self,
        request: RagHandoffRequest,
        *,
        session_id: str | None = None,
    ) -> AgentTurnResult:
        """Execute only the focused task and return public RAG observations."""
        del session_id  # Stage 16 will attach this work to the coordinator trace.
        result = await self._agent.run_delegated(
            request.task,
            history=request.history,
        )
        if result.execution_steps:
            return result
        return result.model_copy(
            update={"execution_steps": self._execution_steps(result)}
        )

    def shutdown(self) -> None:
        """Release resources owned by the adapted RAG agent."""
        self._agent.shutdown()

    @classmethod
    def _execution_steps(
        cls,
        result: AgentTurnResult,
    ) -> tuple[ExecutionStep, ...]:
        steps = [
            ExecutionStep(
                sequence=index,
                agent=ExecutionAgent.RAG,
                action=cls._ACTION_MAP[summary.action],
                status=ExecutionStatus.COMPLETED,
                summary=summary.detail,
            )
            for index, summary in enumerate(result.action_summaries, start=1)
        ]
        failed_search_summary = cls._FAILED_SEARCH_SUMMARIES.get(
            result.termination_reason
        )
        if failed_search_summary is not None:
            steps.append(
                ExecutionStep(
                    sequence=len(steps) + 1,
                    agent=ExecutionAgent.RAG,
                    action=ExecutionAction.SEARCH,
                    status=ExecutionStatus.FAILED,
                    summary=failed_search_summary,
                )
            )
        return tuple(steps)
