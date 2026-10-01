"""Bounded single-agent loop for grounded retrieval-augmented answers."""

import re
from collections.abc import Sequence
from pathlib import Path

from agentic_eval_toyapp.models import (
    AgentActionKind,
    AgentActionSummary,
    AgentTurnResult,
    AgentTurnStatus,
    ClarifyAction,
    ConversationMessage,
    ModelRequest,
    RetrievedChunk,
    SearchAction,
    TerminationReason,
)
from agentic_eval_toyapp.providers import GenerationProvider, GenerationProviderError
from agentic_eval_toyapp.retrieval import Retriever

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
        retriever: Retriever,
        system_prompt: str,
        max_model_calls: int = 3,
        max_searches: int = 2,
        retrieval_top_k: int = 3,
    ) -> None:
        if not system_prompt.strip():
            raise ValueError("system_prompt must not be blank")
        if max_model_calls < 1:
            raise ValueError("max_model_calls must be at least 1")
        if max_searches < 1:
            raise ValueError("max_searches must be at least 1")
        if retrieval_top_k < 1:
            raise ValueError("retrieval_top_k must be at least 1")
        self._provider = provider
        self._retriever = retriever
        self._system_prompt = system_prompt.strip()
        self._max_model_calls = max_model_calls
        self._max_searches = max_searches
        self._retrieval_top_k = retrieval_top_k

    def run(
        self,
        user_message: str,
        *,
        history: Sequence[ConversationMessage] = (),
    ) -> AgentTurnResult:
        """Run one bounded turn without retaining session state."""
        stripped_message = user_message.strip()
        if not stripped_message:
            raise ValueError("user_message must not be blank")

        model_calls = 0
        searches = 0
        seen_queries: set[str] = set()
        retrieved_by_id: dict[str, RetrievedChunk] = {}
        summaries: list[AgentActionSummary] = []

        while model_calls < self._max_model_calls:
            model_calls += 1
            request = ModelRequest(
                system_prompt=self._system_prompt,
                user_message=stripped_message,
                history=tuple(history),
                retrieved_chunks=tuple(retrieved_by_id.values()),
                model_call_number=model_calls,
                remaining_searches=max(self._max_searches - searches, 0),
            )
            try:
                action = self._provider.generate_action(request)
            except GenerationProviderError:
                return self._failure(
                    message="The generation provider could not produce an action.",
                    reason=TerminationReason.PROVIDER_ERROR,
                    model_calls=model_calls,
                    searches=searches,
                    retrieved_chunks=tuple(retrieved_by_id.values()),
                    summaries=tuple(summaries),
                )

            if isinstance(action, SearchAction):
                if searches >= self._max_searches:
                    return self._failure(
                        message="The search limit was reached before an answer was produced.",
                        reason=TerminationReason.SEARCH_LIMIT,
                        model_calls=model_calls,
                        searches=searches,
                        retrieved_chunks=tuple(retrieved_by_id.values()),
                        summaries=tuple(summaries),
                    )
                normalised_query = " ".join(action.query.casefold().split())
                if normalised_query in seen_queries:
                    return self._failure(
                        message="The agent repeated a search and stopped safely.",
                        reason=TerminationReason.REPEATED_SEARCH,
                        model_calls=model_calls,
                        searches=searches,
                        retrieved_chunks=tuple(retrieved_by_id.values()),
                        summaries=tuple(summaries),
                    )
                seen_queries.add(normalised_query)
                searches += 1
                try:
                    results = self._retriever.search(
                        action.query,
                        top_k=self._retrieval_top_k,
                    )
                except Exception:
                    return self._failure(
                        message="The knowledge-base search failed.",
                        reason=TerminationReason.RETRIEVAL_ERROR,
                        model_calls=model_calls,
                        searches=searches,
                        retrieved_chunks=tuple(retrieved_by_id.values()),
                        summaries=tuple(summaries),
                    )
                summaries.append(
                    AgentActionSummary(
                        action=AgentActionKind.SEARCH,
                        detail=f"Retrieved {len(results)} chunks.",
                    )
                )
                if not results:
                    return self._failure(
                        message="No relevant evidence was found in the knowledge base.",
                        reason=TerminationReason.EMPTY_RETRIEVAL,
                        model_calls=model_calls,
                        searches=searches,
                        retrieved_chunks=tuple(retrieved_by_id.values()),
                        summaries=tuple(summaries),
                    )
                for result in results:
                    retrieved_by_id.setdefault(result.chunk_id, result)
                continue

            if isinstance(action, ClarifyAction):
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
                    termination_reason=TerminationReason.CLARIFICATION_REQUIRED,
                    model_calls=model_calls,
                    searches=searches,
                )

            if not retrieved_by_id:
                return self._failure(
                    message="The agent attempted to answer without retrieved evidence.",
                    reason=TerminationReason.ANSWER_WITHOUT_RETRIEVAL,
                    model_calls=model_calls,
                    searches=searches,
                    retrieved_chunks=(),
                    summaries=tuple(summaries),
                )
            declared_citations = set(action.citations)
            available_citations = set(retrieved_by_id)
            if not declared_citations <= available_citations:
                return self._failure(
                    message="The answer cited evidence that was not retrieved.",
                    reason=TerminationReason.INVALID_CITATIONS,
                    model_calls=model_calls,
                    searches=searches,
                    retrieved_chunks=tuple(retrieved_by_id.values()),
                    summaries=tuple(summaries),
                )
            inline_citations = set(_INLINE_CITATION_PATTERN.findall(action.markdown))
            if inline_citations != declared_citations:
                return self._failure(
                    message="The answer citation declarations did not match its Markdown citations.",
                    reason=TerminationReason.CITATION_MISMATCH,
                    model_calls=model_calls,
                    searches=searches,
                    retrieved_chunks=tuple(retrieved_by_id.values()),
                    summaries=tuple(summaries),
                )
            summaries.append(
                AgentActionSummary(
                    action=AgentActionKind.ANSWER,
                    detail=f"Returned an answer with {len(action.citations)} citations.",
                )
            )
            return AgentTurnResult(
                status=AgentTurnStatus.ANSWERED,
                message=action.markdown,
                citations=action.citations,
                retrieved_chunks=tuple(retrieved_by_id.values()),
                action_summaries=tuple(summaries),
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
    ) -> AgentTurnResult:
        """Build one safe failure result without internal exception details."""
        return AgentTurnResult(
            status=AgentTurnStatus.FAILED,
            message=message,
            retrieved_chunks=retrieved_chunks,
            action_summaries=summaries,
            termination_reason=reason,
            model_calls=model_calls,
            searches=searches,
        )
