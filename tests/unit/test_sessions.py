"""Tests for bounded process-local conversation sessions."""

from collections.abc import Sequence
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError
from tests.support.fake_clock import FakeClock

from agentic_eval_toyapp.models import (
    AgentTurnResult,
    AgentTurnStatus,
    ClarifyAction,
    ConversationMessage,
    ConversationRole,
    TerminationReason,
)
from agentic_eval_toyapp.reports import StoredReport
from agentic_eval_toyapp.session_memory import (
    SESSION_ALLOWANCE_BYTES,
    SessionMemoryLedger,
)
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


@pytest.mark.anyio
async def test_seed_read_only_returns_read_only_true() -> None:
    """A session seeded with read_only=True reports read_only in its snapshot."""
    store = InMemorySessionStore()
    session_id = UUID("00000000-0000-0000-0000-000000000001")
    user = ConversationMessage(role=ConversationRole.USER, content="hi")
    assistant = ConversationMessage(role=ConversationRole.ASSISTANT, content="hello")
    store.seed(session_id, ((user, assistant),), read_only=True)

    snapshot = await store.get(session_id)
    assert snapshot.read_only is True


@pytest.mark.anyio
async def test_seed_default_is_mutable() -> None:
    """A session seeded without read_only flag is mutable."""
    store = InMemorySessionStore()
    session_id = UUID("00000000-0000-0000-0000-000000000002")
    user = ConversationMessage(role=ConversationRole.USER, content="hi")
    assistant = ConversationMessage(role=ConversationRole.ASSISTANT, content="hello")
    store.seed(session_id, ((user, assistant),))

    snapshot = await store.get(session_id)
    assert snapshot.read_only is False


@pytest.mark.anyio
async def test_created_session_is_mutable() -> None:
    """A session created through create() is mutable."""
    store = InMemorySessionStore()
    session = await store.create()
    assert session.read_only is False


def _ledger_store(
    *, max_turns: int = 10
) -> tuple[InMemorySessionStore, SessionMemoryLedger, FakeClock]:
    clock = FakeClock()
    ledger = SessionMemoryLedger(clock=clock)
    return InMemorySessionStore(max_turns=max_turns, ledger=ledger), ledger, clock


def _session_bytes(ledger: SessionMemoryLedger, session_id: UUID) -> int:
    footprint = ledger.footprint(session_id)
    assert footprint is not None
    return footprint.session_bytes


@pytest.mark.anyio
async def test_created_session_is_tracked_at_the_allowance() -> None:
    """A new empty session costs exactly the fixed allowance."""
    store, ledger, _clock = _ledger_store()

    session = await store.create()

    assert _session_bytes(ledger, session.session_id) == SESSION_ALLOWANCE_BYTES
    assert ledger.total_bytes == SESSION_ALLOWANCE_BYTES


@pytest.mark.anyio
async def test_get_records_last_access_and_moves_session_to_the_end() -> None:
    """Reading a session counts as using it."""
    store, ledger, clock = _ledger_store()
    first = await store.create()
    second = await store.create()
    clock.advance(30)

    await store.get(first.session_id)

    assert ledger.idle_order() == (second.session_id, first.session_id)
    footprint = ledger.footprint(first.session_id)
    assert footprint is not None
    assert footprint.last_access == 130.0


@pytest.mark.anyio
async def test_get_of_unknown_session_raises_without_tracking() -> None:
    """A missing session is not added to the ledger by a failed read."""
    store, ledger, _clock = _ledger_store()
    unknown = uuid4()

    with pytest.raises(SessionNotFoundError):
        await store.get(unknown)

    assert ledger.footprint(unknown) is None


@pytest.mark.anyio
async def test_turn_adds_both_messages_and_counts_as_an_access() -> None:
    """A stored turn adds its text and refreshes last access after the agent."""
    store, ledger, clock = _ledger_store()
    session = await store.create()
    other = await store.create()
    clock.advance(10)

    await store.run_turn(session.session_id, "hello", agent=_RecordingAgent())

    reply = "Please clarify: hello"
    assert _session_bytes(ledger, session.session_id) == (
        SESSION_ALLOWANCE_BYTES + len("hello") + len(reply)
    )
    assert ledger.idle_order() == (other.session_id, session.session_id)


@pytest.mark.anyio
async def test_turn_dropped_from_a_full_history_is_subtracted() -> None:
    """The deque drops its oldest turn, so the estimate drops it too."""
    store, ledger, _clock = _ledger_store(max_turns=2)
    session = await store.create()
    agent = _RecordingAgent()

    def turn_bytes(message: str) -> int:
        return len(message) + len(f"Please clarify: {message}")

    for message in ("a", "bb", "ccc"):
        await store.run_turn(session.session_id, message, agent=agent)

    assert _session_bytes(ledger, session.session_id) == (
        SESSION_ALLOWANCE_BYTES + turn_bytes("bb") + turn_bytes("ccc")
    )


@pytest.mark.anyio
async def test_failed_turn_leaves_the_estimate_unchanged() -> None:
    """An agent error stores nothing, so nothing is added."""
    store, ledger, _clock = _ledger_store()
    session = await store.create()

    with pytest.raises(RuntimeError):
        await store.run_turn(session.session_id, "hello", agent=_FailingAgent())

    assert ledger.total_bytes == SESSION_ALLOWANCE_BYTES


@pytest.mark.anyio
async def test_seeded_session_is_never_tracked() -> None:
    """Demo sessions stay outside the ledger, even when read."""
    store, ledger, _clock = _ledger_store()
    demo_id = uuid4()
    user = ConversationMessage(role=ConversationRole.USER, content="Q")
    assistant = ConversationMessage(role=ConversationRole.ASSISTANT, content="A")
    store.seed(demo_id, ((user, assistant),), read_only=True)

    await store.get(demo_id)

    assert ledger.footprint(demo_id) is None
    assert ledger.total_bytes == 0


@pytest.mark.anyio
async def test_store_without_a_ledger_still_runs_turns() -> None:
    """The ledger is optional; existing callers are unaffected."""
    store = InMemorySessionStore()
    session = await store.create()

    await store.run_turn(session.session_id, "hello", agent=_RecordingAgent())

    assert (await store.get(session.session_id)).retained_turns == 1
