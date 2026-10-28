"""Tests for bounded process-local conversation sessions."""

from collections.abc import Sequence
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from agentic_eval_toyapp.models import (
    AgentTurnResult,
    AgentTurnStatus,
    ClarifyAction,
    ConversationMessage,
    ConversationRole,
    TerminationReason,
)
from agentic_eval_toyapp.reports import StoredReport
from agentic_eval_toyapp.sessions import InMemorySessionStore, SessionNotFoundError


class _RecordingAgent:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[ConversationMessage, ...]]] = []
        self.session_ids: list[str | None] = []

    async def run(
        self,
        user_message: str,
        *,
        history: Sequence[ConversationMessage] = (),
        session_id: str | None = None,
        reports: Sequence[StoredReport] = (),
    ) -> AgentTurnResult:
        del reports
        self.session_ids.append(session_id)
        self.calls.append((user_message, tuple(history)))
        return AgentTurnResult(
            status=AgentTurnStatus.CLARIFICATION_REQUIRED,
            message=f"Please clarify: {user_message}",
            termination_reason=TerminationReason.CLARIFICATION_REQUIRED,
            model_calls=1,
            searches=0,
            model_actions=(ClarifyAction(question=f"Please clarify: {user_message}"),),
        )


class _FailingAgent:
    async def run(
        self,
        user_message: str,
        *,
        history: Sequence[ConversationMessage] = (),
        session_id: str | None = None,
        reports: Sequence[StoredReport] = (),
    ) -> AgentTurnResult:
        del session_id, reports
        raise RuntimeError("unexpected agent failure")


@pytest.mark.anyio
async def test_store_creates_unique_empty_sessions() -> None:
    """Each session should start empty and remain isolated by UUID."""
    store = InMemorySessionStore()

    first = await store.create()
    second = await store.create()

    assert first.session_id != second.session_id
    assert first.messages == second.messages == ()
    assert first.retained_turns == second.retained_turns == 0


@pytest.mark.anyio
async def test_store_passes_ordered_history_and_trims_complete_turns() -> None:
    """Eviction must remove whole pairs while preserving recent chronology."""
    store = InMemorySessionStore(max_turns=2)
    agent = _RecordingAgent()
    session = await store.create()

    for message in ("first", "second", "third"):
        await store.run_turn(session.session_id, message, agent=agent)

    snapshot = await store.get(session.session_id)
    assert snapshot.retained_turns == 2
    assert [(message.role, message.content) for message in snapshot.messages] == [
        (ConversationRole.USER, "second"),
        (ConversationRole.ASSISTANT, "Please clarify: second"),
        (ConversationRole.USER, "third"),
        (ConversationRole.ASSISTANT, "Please clarify: third"),
    ]
    assert [message.content for message in agent.calls[2][1]] == [
        "first",
        "Please clarify: first",
        "second",
        "Please clarify: second",
    ]
    assert agent.session_ids == [str(session.session_id)] * 3


@pytest.mark.anyio
async def test_store_keeps_session_histories_isolated() -> None:
    """Messages from one session must never enter another agent context."""
    store = InMemorySessionStore()
    agent = _RecordingAgent()
    first = await store.create()
    second = await store.create()

    await store.run_turn(first.session_id, "private first-session text", agent=agent)
    await store.run_turn(second.session_id, "second-session text", agent=agent)

    assert agent.calls[1][1] == ()
    assert (await store.get(first.session_id)).messages[0].content == (
        "private first-session text"
    )
    assert (await store.get(second.session_id)).messages[
        0
    ].content == "second-session text"


@pytest.mark.anyio
async def test_store_rejects_unknown_sessions_and_invalid_input() -> None:
    """Unknown IDs and blank direct-call messages must not create state."""
    store = InMemorySessionStore()
    session = await store.create()
    agent = _RecordingAgent()

    with pytest.raises(SessionNotFoundError):
        await store.get(uuid4())
    with pytest.raises(SessionNotFoundError):
        await store.run_turn(uuid4(), "Question", agent=agent)
    with pytest.raises(ValidationError):
        await store.run_turn(session.session_id, "   ", agent=agent)

    assert agent.calls == []
    assert (await store.get(session.session_id)).messages == ()


@pytest.mark.anyio
async def test_store_does_not_append_partial_turn_when_agent_raises() -> None:
    """A failed call outside the typed agent boundary must not corrupt history."""
    store = InMemorySessionStore()
    session = await store.create()

    with pytest.raises(RuntimeError, match="unexpected"):
        await store.run_turn(session.session_id, "Question", agent=_FailingAgent())

    assert (await store.get(session.session_id)).messages == ()


@pytest.mark.anyio
async def test_store_retries_a_colliding_id_and_rejects_invalid_bounds() -> None:
    """An injected UUID collision must not overwrite an existing session."""
    first_id = UUID("00000000-0000-0000-0000-000000000001")
    second_id = UUID("00000000-0000-0000-0000-000000000002")
    generated_ids = iter((first_id, first_id, second_id))
    store = InMemorySessionStore(id_factory=lambda: next(generated_ids))

    first = await store.create()
    second = await store.create()

    assert first.session_id == first_id
    assert second.session_id == second_id
    with pytest.raises(ValueError, match="max_turns"):
        InMemorySessionStore(max_turns=0)
