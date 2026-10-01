"""Tests for the bounded single-agent RAG loop."""

from collections.abc import Mapping
from datetime import date
from pathlib import Path

import pytest

from agentic_eval_toyapp.agent import PromptLoadError, RagAgent, load_system_prompt
from agentic_eval_toyapp.models import (
    AgentAction,
    AgentActionKind,
    AgentTurnStatus,
    AnswerAction,
    ClarifyAction,
    ConversationMessage,
    ConversationRole,
    DocumentMetadata,
    RetrievedChunk,
    SearchAction,
    TerminationReason,
)
from agentic_eval_toyapp.providers import ScriptedGenerationProvider


def _chunk(chunk_id: str = "TRAVEL-001#1") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=chunk_id.split("#", maxsplit=1)[0],
        title="Travel policy",
        text="Domestic meals are limited to $50 per day.",
        metadata=DocumentMetadata(topic="travel", effective_date=date(2026, 1, 1)),
        similarity_score=0.9,
    )


class _StubRetriever:
    def __init__(
        self,
        results_by_query: Mapping[str, tuple[RetrievedChunk, ...]],
        *,
        error: Exception | None = None,
    ) -> None:
        self._results_by_query = results_by_query
        self._error = error
        self.calls: list[tuple[str, int | None]] = []

    def search(
        self, query: str, *, top_k: int | None = None
    ) -> tuple[RetrievedChunk, ...]:
        self.calls.append((query, top_k))
        if self._error is not None:
            raise self._error
        return self._results_by_query.get(query, ())


def _agent(
    actions: tuple[AgentAction, ...],
    retriever: _StubRetriever,
    *,
    max_model_calls: int = 3,
    max_searches: int = 2,
) -> tuple[RagAgent, ScriptedGenerationProvider]:
    provider = ScriptedGenerationProvider(actions)
    return (
        RagAgent(
            provider=provider,
            retriever=retriever,
            system_prompt="Search, clarify, or answer with citations.",
            max_model_calls=max_model_calls,
            max_searches=max_searches,
            retrieval_top_k=3,
        ),
        provider,
    )


def test_agent_searches_then_returns_grounded_answer() -> None:
    """Evidence must reach the second model call before an answer is accepted."""
    chunk = _chunk()
    retriever = _StubRetriever({"domestic meals": (chunk,)})
    agent, provider = _agent(
        (
            SearchAction(query="domestic meals"),
            AnswerAction(
                markdown="The limit is $50 per day. [TRAVEL-001#1]",
                citations=("TRAVEL-001#1",),
            ),
        ),
        retriever,
    )
    history = (
        ConversationMessage(role=ConversationRole.USER, content="Earlier question"),
        ConversationMessage(role=ConversationRole.ASSISTANT, content="Earlier reply"),
    )

    result = agent.run("What is the domestic meal limit?", history=history)

    assert result.status is AgentTurnStatus.ANSWERED
    assert result.termination_reason is TerminationReason.ANSWERED
    assert result.citations == ("TRAVEL-001#1",)
    assert result.model_calls == 2
    assert result.searches == 1
    assert retriever.calls == [("domestic meals", 3)]
    assert provider.requests[0].history == history
    assert provider.requests[1].retrieved_chunks == (chunk,)
    assert [summary.action for summary in result.action_summaries] == [
        AgentActionKind.SEARCH,
        AgentActionKind.ANSWER,
    ]
    assert [summary.detail for summary in result.action_summaries] == [
        "Retrieved 1 chunk.",
        "Returned an answer with 1 citation.",
    ]
    assert all(
        "domestic meals" not in summary.detail for summary in result.action_summaries
    )


def test_agent_returns_clarification_without_searching() -> None:
    """Material ambiguity can terminate with one direct clarification action."""
    retriever = _StubRetriever({})
    agent, _ = _agent(
        (ClarifyAction(question="Is the trip domestic or international?"),),
        retriever,
    )

    result = agent.run("What travel allowance can I claim?")

    assert result.status is AgentTurnStatus.CLARIFICATION_REQUIRED
    assert result.message == "Is the trip domestic or international?"
    assert result.termination_reason is TerminationReason.CLARIFICATION_REQUIRED
    assert result.searches == 0
    assert retriever.calls == []


def test_agent_action_summaries_pluralise_multiple_counts() -> None:
    """Observable summaries should use grammatical nouns for multiple items."""
    first = _chunk("TRAVEL-001#1")
    second = _chunk("TRAVEL-001#2")
    agent, _ = _agent(
        (
            SearchAction(query="travel"),
            AnswerAction(
                markdown=(
                    "Domestic meals are covered. [TRAVEL-001#1] "
                    "Travel requires approval. [TRAVEL-001#2]"
                ),
                citations=("TRAVEL-001#1", "TRAVEL-001#2"),
            ),
        ),
        _StubRetriever({"travel": (first, second)}),
    )

    result = agent.run("Summarise travel policy")

    assert [summary.detail for summary in result.action_summaries] == [
        "Retrieved 2 chunks.",
        "Returned an answer with 2 citations.",
    ]


def test_agent_rejects_answer_without_retrieval() -> None:
    """A provider cannot bypass grounding by answering immediately."""
    agent, _ = _agent(
        (
            AnswerAction(
                markdown="The limit is $50. [TRAVEL-001#1]",
                citations=("TRAVEL-001#1",),
            ),
        ),
        _StubRetriever({}),
    )

    result = agent.run("What is the limit?")

    assert result.status is AgentTurnStatus.FAILED
    assert result.termination_reason is TerminationReason.ANSWER_WITHOUT_RETRIEVAL


def test_agent_stops_on_empty_retrieval() -> None:
    """No evidence produces a safe failure rather than an unsupported answer."""
    agent, _ = _agent(
        (SearchAction(query="unknown policy"),),
        _StubRetriever({"unknown policy": ()}),
    )

    result = agent.run("Unknown question")

    assert result.termination_reason is TerminationReason.EMPTY_RETRIEVAL
    assert result.searches == 1
    assert result.action_summaries[0].detail == "Retrieved 0 chunks."


def test_agent_stops_on_repeated_normalised_search() -> None:
    """Case and whitespace changes cannot evade repeated-search detection."""
    chunk = _chunk()
    retriever = _StubRetriever({"travel policy": (chunk,)})
    agent, _ = _agent(
        (
            SearchAction(query="travel policy"),
            SearchAction(query="  TRAVEL   POLICY "),
        ),
        retriever,
    )

    result = agent.run("Travel question")

    assert result.termination_reason is TerminationReason.REPEATED_SEARCH
    assert result.searches == 1
    assert len(retriever.calls) == 1


def test_agent_rejects_citation_that_was_not_retrieved() -> None:
    """Declared citations must be a subset of retrieved chunk IDs."""
    chunk = _chunk()
    agent, _ = _agent(
        (
            SearchAction(query="travel"),
            AnswerAction(
                markdown="The limit is $80. [TRAVEL-001#2]",
                citations=("TRAVEL-001#2",),
            ),
        ),
        _StubRetriever({"travel": (chunk,)}),
    )

    result = agent.run("What is the limit?")

    assert result.termination_reason is TerminationReason.INVALID_CITATIONS


@pytest.mark.parametrize(
    ("markdown", "citations"),
    [
        ("The limit is $50.", ("TRAVEL-001#1",)),
        (
            "The limit is $50. [TRAVEL-001#1] [TRAVEL-001#2]",
            ("TRAVEL-001#1",),
        ),
    ],
)
def test_agent_rejects_markdown_and_declared_citation_mismatch(
    markdown: str, citations: tuple[str, ...]
) -> None:
    """Inline citations and the structured citation list must agree exactly."""
    chunk = _chunk()
    agent, _ = _agent(
        (
            SearchAction(query="travel"),
            AnswerAction(markdown=markdown, citations=citations),
        ),
        _StubRetriever({"travel": (chunk,)}),
    )

    result = agent.run("What is the limit?")

    assert result.termination_reason is TerminationReason.CITATION_MISMATCH


def test_agent_enforces_search_and_model_call_limits() -> None:
    """Independent bounds prevent unbounded tool or provider loops."""
    chunk = _chunk()
    search_limited_agent, _ = _agent(
        (SearchAction(query="one"), SearchAction(query="two")),
        _StubRetriever({"one": (chunk,), "two": (chunk,)}),
        max_searches=1,
    )
    call_limited_agent, _ = _agent(
        (SearchAction(query="one"),),
        _StubRetriever({"one": (chunk,)}),
        max_model_calls=1,
    )

    search_limited = search_limited_agent.run("Question")
    call_limited = call_limited_agent.run("Question")

    assert search_limited.termination_reason is TerminationReason.SEARCH_LIMIT
    assert search_limited.model_calls == 2
    assert search_limited.searches == 1
    assert call_limited.termination_reason is TerminationReason.MODEL_CALL_LIMIT
    assert call_limited.model_calls == 1
    assert call_limited.searches == 1


def test_agent_converts_provider_and_retriever_errors_to_safe_results() -> None:
    """Internal exception details must not leak from the agent boundary."""
    chunk = _chunk()
    provider_agent, _ = _agent(
        (SearchAction(query="one"),),
        _StubRetriever({"one": (chunk,)}),
        max_model_calls=2,
    )
    retrieval_agent, _ = _agent(
        (SearchAction(query="one"),),
        _StubRetriever({}, error=RuntimeError("sensitive backend detail")),
    )

    provider_result = provider_agent.run("Question")
    retrieval_result = retrieval_agent.run("Question")

    assert provider_result.termination_reason is TerminationReason.PROVIDER_ERROR
    assert "exhausted" not in provider_result.message
    assert retrieval_result.termination_reason is TerminationReason.RETRIEVAL_ERROR
    assert "sensitive" not in retrieval_result.message


@pytest.mark.parametrize("message", ["", "   "])
def test_agent_rejects_blank_user_message(message: str) -> None:
    """Blank input must fail before consuming a scripted model action."""
    agent, provider = _agent(
        (ClarifyAction(question="Please provide a question."),),
        _StubRetriever({}),
    )

    with pytest.raises(ValueError, match="user_message"):
        agent.run(message)
    assert provider.actions_remaining == 1


def test_agent_accepts_unicode_user_message() -> None:
    """Unicode user input reaches the typed provider request unchanged."""
    agent, provider = _agent(
        (ClarifyAction(question="どのポリシーですか？"),),
        _StubRetriever({}),
    )

    result = agent.run("休暇ポリシーは何ですか？")

    assert result.status is AgentTurnStatus.CLARIFICATION_REQUIRED
    assert provider.requests[0].user_message == "休暇ポリシーは何ですか？"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"system_prompt": " "}, "system_prompt"),
        ({"max_model_calls": 0}, "max_model_calls"),
        ({"max_searches": 0}, "max_searches"),
        ({"retrieval_top_k": 0}, "retrieval_top_k"),
    ],
)
def test_agent_rejects_invalid_construction_limits(
    overrides: dict[str, object], message: str
) -> None:
    """Invalid bounds must fail before an agent turn starts."""
    arguments: dict[str, object] = {
        "provider": ScriptedGenerationProvider((ClarifyAction(question="Question?"),)),
        "retriever": _StubRetriever({}),
        "system_prompt": "Prompt",
        "max_model_calls": 3,
        "max_searches": 2,
        "retrieval_top_k": 3,
    }
    arguments.update(overrides)

    with pytest.raises(ValueError, match=message):
        RagAgent(**arguments)  # type: ignore[arg-type]


def test_load_system_prompt_reads_default_and_rejects_missing_or_blank(
    tmp_path: Path,
) -> None:
    """The prompt must be packaged, readable, and non-empty."""
    assert "Search before" in load_system_prompt()
    blank_path = tmp_path / "blank.md"
    blank_path.write_text("   ", encoding="utf-8")

    with pytest.raises(PromptLoadError, match="blank"):
        load_system_prompt(blank_path)
    with pytest.raises(PromptLoadError, match="could not read"):
        load_system_prompt(tmp_path / "missing.md")
