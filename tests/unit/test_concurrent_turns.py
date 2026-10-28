"""Offline proofs that the async turn path holds many turns in flight."""

import asyncio
from collections.abc import Sequence

import pytest

from agentic_eval_toyapp.models import (
    AgentTurnResult,
    AgentTurnStatus,
    ClarifyAction,
    ConversationMessage,
    TerminationReason,
)
from agentic_eval_toyapp.reports import StoredReport
from agentic_eval_toyapp.sessions import InMemorySessionStore

_TIMEOUT_SECONDS = 5.0


class _GatedAgent:
    """Block every turn on one event while counting turns inside the agent."""

    def __init__(self) -> None:
        self.gate = asyncio.Event()
        self.in_flight = 0
        self.peak = 0
        self.events: list[str] = []
        self.histories: dict[str, int] = {}
        self._changed = asyncio.Condition()

    async def run(
        self,
        user_message: str,
        *,
        history: Sequence[ConversationMessage] = (),
        session_id: str | None = None,
        reports: Sequence[StoredReport] = (),
    ) -> AgentTurnResult:
        del session_id, reports
        self.events.append(f"enter:{user_message}")
        self.histories[user_message] = len(history)
        async with self._changed:
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
            self._changed.notify_all()
        await self.gate.wait()
        async with self._changed:
            self.in_flight -= 1
            self._changed.notify_all()
        self.events.append(f"exit:{user_message}")
        question = f"Clarify: {user_message}"
        return AgentTurnResult(
            status=AgentTurnStatus.CLARIFICATION_REQUIRED,
            message=question,
            termination_reason=TerminationReason.CLARIFICATION_REQUIRED,
            model_calls=1,
            searches=0,
            model_actions=(ClarifyAction(question=question),),
        )

    async def wait_for_in_flight(self, count: int) -> None:
        async with self._changed:
            await asyncio.wait_for(
                self._changed.wait_for(lambda: self.in_flight >= count),
                _TIMEOUT_SECONDS,
            )


@pytest.mark.anyio
async def test_turns_on_distinct_sessions_overlap_in_time() -> None:
    """N sessions must all be inside the agent at once without N threads."""
    turn_count = 8
    store = InMemorySessionStore()
    agent = _GatedAgent()
    sessions = [store.create() for _ in range(turn_count)]
    tasks = [
        asyncio.create_task(
            store.run_turn(session.session_id, f"message-{index}", agent=agent)
        )
        for index, session in enumerate(sessions)
    ]

    await agent.wait_for_in_flight(turn_count)
    assert agent.peak == turn_count
    agent.gate.set()
    results = await asyncio.wait_for(asyncio.gather(*tasks), _TIMEOUT_SECONDS)

    assert [result.message for result in results] == [
        f"Clarify: message-{index}" for index in range(turn_count)
    ]
    assert all(store.get(s.session_id).retained_turns == 1 for s in sessions)


@pytest.mark.anyio
async def test_turns_on_one_session_run_strictly_in_order() -> None:
    """A waiting same-session turn must not enter the agent or see stale history."""
    store = InMemorySessionStore()
    agent = _GatedAgent()
    session = store.create()
    first = asyncio.create_task(
        store.run_turn(session.session_id, "first", agent=agent)
    )
    await agent.wait_for_in_flight(1)
    second = asyncio.create_task(
        store.run_turn(session.session_id, "second", agent=agent)
    )

    # Give the second turn every chance to (wrongly) enter while the first is held.
    for _ in range(20):
        await asyncio.sleep(0)
    assert agent.events == ["enter:first"]
    assert agent.peak == 1

    agent.gate.set()
    await asyncio.wait_for(asyncio.gather(first, second), _TIMEOUT_SECONDS)

    assert agent.events == [
        "enter:first",
        "exit:first",
        "enter:second",
        "exit:second",
    ]
    assert agent.histories == {"first": 0, "second": 2}
    assert [m.content for m in store.get(session.session_id).messages] == [
        "first",
        "Clarify: first",
        "second",
        "Clarify: second",
    ]


@pytest.mark.anyio
async def test_get_does_not_wait_for_an_in_flight_turn() -> None:
    """Reading a session mid-turn returns the committed history immediately."""
    store = InMemorySessionStore()
    agent = _GatedAgent()
    session = store.create()
    turn = asyncio.create_task(store.run_turn(session.session_id, "held", agent=agent))
    await agent.wait_for_in_flight(1)

    snapshot = store.get(session.session_id)

    assert snapshot.messages == ()
    assert snapshot.retained_turns == 0
    agent.gate.set()
    await asyncio.wait_for(turn, _TIMEOUT_SECONDS)
    assert store.get(session.session_id).retained_turns == 1


@pytest.mark.anyio
async def test_cancelled_turn_releases_the_session_lock_without_appending() -> None:
    """Cancelling an in-flight turn must not wedge or corrupt its session."""
    store = InMemorySessionStore()
    agent = _GatedAgent()
    session = store.create()
    turn = asyncio.create_task(
        store.run_turn(session.session_id, "cancelled", agent=agent)
    )
    await agent.wait_for_in_flight(1)

    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn

    assert store.get(session.session_id).messages == ()
    agent.gate.set()
    await asyncio.wait_for(
        store.run_turn(session.session_id, "after", agent=agent), _TIMEOUT_SECONDS
    )
    assert store.get(session.session_id).retained_turns == 1
