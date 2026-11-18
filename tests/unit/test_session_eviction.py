"""Tests for budget eviction shared by the in-memory session and report stores."""

import hashlib
import logging
from collections.abc import Sequence
from functools import partial
from uuid import UUID, uuid4

import anyio
import pytest
from tests.support.fake_clock import FakeClock

from agentic_eval_toyapp.models import (
    AgentTurnResult,
    AgentTurnStatus,
    ClarifyAction,
    ConversationMessage,
    ConversationRole,
    ReportMediaType,
    TerminationReason,
)
from agentic_eval_toyapp.redis_memory import (
    TURN_REPLY_ALLOWANCE_BYTES,
    StorageCapacityError,
)
from agentic_eval_toyapp.reports import (
    InMemoryReportStore,
    ProcessedReport,
    ReportNotFoundError,
    StoredReport,
)
from agentic_eval_toyapp.session_memory import (
    SESSION_ALLOWANCE_BYTES,
    SessionMemoryLedger,
    SessionNotFoundError,
)
from agentic_eval_toyapp.sessions import InMemorySessionStore

KIB = 1024


class _ShortAgent:
    """Reply with a fixed short message and count calls."""

    def __init__(self) -> None:
        self.calls = 0

    async def run(
        self,
        user_message: str,
        *,
        history: Sequence[ConversationMessage] = (),
        session_id: str | None = None,
        reports: Sequence[StoredReport] = (),
    ) -> AgentTurnResult:
        del user_message, history, session_id, reports
        self.calls += 1
        return _result("ok")


class _BlockingAgent:
    """Hold a turn open until the test releases it."""

    def __init__(self) -> None:
        self.started = anyio.Event()
        self.release = anyio.Event()

    async def run(
        self,
        user_message: str,
        *,
        history: Sequence[ConversationMessage] = (),
        session_id: str | None = None,
        reports: Sequence[StoredReport] = (),
    ) -> AgentTurnResult:
        del user_message, history, session_id, reports
        self.started.set()
        await self.release.wait()
        return _result("ok")


class _FailingAgent:
    async def run(
        self,
        user_message: str,
        *,
        history: Sequence[ConversationMessage] = (),
        session_id: str | None = None,
        reports: Sequence[StoredReport] = (),
    ) -> AgentTurnResult:
        del user_message, history, session_id, reports
        raise RuntimeError("agent failed")


def _result(message: str) -> AgentTurnResult:
    return AgentTurnResult(
        status=AgentTurnStatus.CLARIFICATION_REQUIRED,
        message=message,
        termination_reason=TerminationReason.CLARIFICATION_REQUIRED,
        model_calls=1,
        searches=0,
        model_actions=(ClarifyAction(question=message),),
    )


def _report(size: int) -> ProcessedReport:
    """A text report whose footprint is exactly `2 * size` bytes."""
    content = b"x" * size
    return ProcessedReport(
        filename="report.txt",
        media_type=ReportMediaType.TEXT,
        byte_size=size,
        original_bytes=content,
        extracted_text="x" * size,
        sha256=hashlib.sha256(content).hexdigest(),
    )


class _Stores:
    def __init__(self, budget_bytes: int | None) -> None:
        self.clock = FakeClock()
        self.ledger = SessionMemoryLedger(budget_bytes=budget_bytes, clock=self.clock)
        self.sessions = InMemorySessionStore(ledger=self.ledger)
        self.reports = InMemoryReportStore(
            max_reports_per_session=10, ledger=self.ledger
        )

    async def create(self) -> UUID:
        session = await self.sessions.create()
        self.clock.advance(1)
        return session.session_id

    def seed_demo(self) -> UUID:
        demo_id = uuid4()
        user = ConversationMessage(role=ConversationRole.USER, content="Q")
        assistant = ConversationMessage(role=ConversationRole.ASSISTANT, content="A")
        self.sessions.seed(demo_id, ((user, assistant),), read_only=True)
        return demo_id


async def _assert_gone(stores: _Stores, session_id: UUID, file_id: UUID) -> None:
    with pytest.raises(SessionNotFoundError):
        await stores.sessions.get(session_id)
    assert await stores.reports.list_for_session(session_id) == ()
    with pytest.raises(ReportNotFoundError):
        await stores.reports.resolve(session_id, (file_id,))
    assert stores.ledger.footprint(session_id) is None


@pytest.mark.anyio
async def test_longest_idle_session_is_evicted_first() -> None:
    """With room for three sessions, the fourth evicts the oldest."""
    stores = _Stores(budget_bytes=3 * SESSION_ALLOWANCE_BYTES)
    oldest, middle, newest = [await stores.create() for _ in range(3)]

    fourth = await stores.create()

    assert stores.ledger.idle_order() == (middle, newest, fourth)
    with pytest.raises(SessionNotFoundError):
        await stores.sessions.get(oldest)


@pytest.mark.anyio
async def test_a_recently_used_session_survives_eviction() -> None:
    """An access moves a session to the back of the queue."""
    stores = _Stores(budget_bytes=3 * SESSION_ALLOWANCE_BYTES)
    oldest, middle, newest = [await stores.create() for _ in range(3)]
    await stores.sessions.get(oldest)

    await stores.create()

    assert oldest in stores.ledger.idle_order()
    with pytest.raises(SessionNotFoundError):
        await stores.sessions.get(middle)
    assert newest in stores.ledger.idle_order()


@pytest.mark.anyio
async def test_eviction_removes_the_session_reports_with_it() -> None:
    """Turns and reports leave both stores, and the total drops by both."""
    stores = _Stores(budget_bytes=SESSION_ALLOWANCE_BYTES + 2 * KIB + 1)
    victim = await stores.create()
    stored = await stores.reports.add(victim, _report(KIB))
    assert stores.ledger.total_bytes == SESSION_ALLOWANCE_BYTES + 2 * KIB

    survivor = await stores.create()

    await _assert_gone(stores, victim, stored.file_id)
    assert stores.ledger.idle_order() == (survivor,)
    assert stores.ledger.total_bytes == SESSION_ALLOWANCE_BYTES


@pytest.mark.anyio
async def test_only_as_many_sessions_as_needed_are_evicted() -> None:
    """Eviction stops as soon as the incoming write fits."""
    stores = _Stores(budget_bytes=4 * SESSION_ALLOWANCE_BYTES)
    ids = [await stores.create() for _ in range(4)]

    await stores.create()

    assert stores.ledger.idle_order()[:3] == tuple(ids[1:])


@pytest.mark.anyio
async def test_large_upload_evicts_several_idle_sessions() -> None:
    """One write may need more than one victim."""
    stores = _Stores(budget_bytes=4 * SESSION_ALLOWANCE_BYTES)
    first, second, third = [await stores.create() for _ in range(3)]
    uploader = await stores.create()

    await stores.reports.add(uploader, _report(SESSION_ALLOWANCE_BYTES))

    assert stores.ledger.idle_order() == (third, uploader)
    for gone in (first, second):
        with pytest.raises(SessionNotFoundError):
            await stores.sessions.get(gone)


@pytest.mark.anyio
async def test_write_exactly_at_the_budget_fits_and_one_byte_more_evicts() -> None:
    """The budget is inclusive."""
    stores = _Stores(budget_bytes=SESSION_ALLOWANCE_BYTES + 2 * KIB)
    first = await stores.create()
    await stores.reports.add(first, _report(KIB))
    assert stores.ledger.total_bytes == stores.ledger.budget_bytes

    stores.ledger.ensure_room(0)
    assert stores.ledger.idle_order() == (first,)

    stores.ledger.ensure_room(1)
    assert stores.ledger.idle_order() == ()


@pytest.mark.anyio
async def test_empty_session_at_a_one_session_budget_evicts_the_previous() -> None:
    """Empty sessions cost the allowance, so they are bounded too."""
    stores = _Stores(budget_bytes=SESSION_ALLOWANCE_BYTES)
    first = await stores.create()

    second = await stores.create()

    assert stores.ledger.idle_order() == (second,)
    with pytest.raises(SessionNotFoundError):
        await stores.sessions.get(first)


@pytest.mark.anyio
async def test_item_larger_than_the_budget_is_refused_without_eviction() -> None:
    """Evicting everyone would not help, so nobody is evicted."""
    stores = _Stores(budget_bytes=4 * SESSION_ALLOWANCE_BYTES)
    others = [await stores.create() for _ in range(2)]
    uploader = await stores.create()

    with pytest.raises(StorageCapacityError):
        await stores.reports.add(uploader, _report(2 * SESSION_ALLOWANCE_BYTES + 1))

    assert stores.ledger.idle_order() == (*others, uploader)
    assert await stores.reports.list_for_session(uploader) == ()


@pytest.mark.anyio
async def test_too_few_candidates_means_no_session_is_evicted() -> None:
    """Planning happens before removal, so a doomed write evicts nothing."""
    stores = _Stores(budget_bytes=3 * SESSION_ALLOWANCE_BYTES)
    other = await stores.create()
    uploader = await stores.create()
    await stores.reports.add(uploader, _report(SESSION_ALLOWANCE_BYTES // 4))
    before = stores.ledger.total_bytes

    with pytest.raises(StorageCapacityError):
        await stores.reports.add(uploader, _report(SESSION_ALLOWANCE_BYTES))

    assert stores.ledger.idle_order() == (other, uploader)
    assert stores.ledger.total_bytes == before


@pytest.mark.anyio
async def test_requesting_session_is_never_evicted_for_its_own_upload() -> None:
    """When only the uploader could be evicted, the upload is refused."""
    stores = _Stores(budget_bytes=SESSION_ALLOWANCE_BYTES + 2 * KIB)
    uploader = await stores.create()
    await stores.reports.add(uploader, _report(KIB // 2))

    with pytest.raises(StorageCapacityError):
        await stores.reports.add(uploader, _report(KIB))

    assert stores.ledger.idle_order() == (uploader,)
    assert len(await stores.reports.list_for_session(uploader)) == 1


@pytest.mark.anyio
async def test_demo_session_is_never_evicted() -> None:
    """The demo is untracked, so a full budget with only the demo refuses."""
    stores = _Stores(budget_bytes=SESSION_ALLOWANCE_BYTES)
    demo_id = stores.seed_demo()
    await stores.create()
    await stores.sessions.get(demo_id)

    await stores.create()

    assert (await stores.sessions.get(demo_id)).read_only is True
    assert stores.ledger.footprint(demo_id) is None


@pytest.mark.anyio
async def test_session_in_the_middle_of_a_turn_is_never_evicted() -> None:
    """A held turn lock exempts the session; the write is refused instead."""
    stores = _Stores(
        budget_bytes=SESSION_ALLOWANCE_BYTES + TURN_REPLY_ALLOWANCE_BYTES + 10
    )
    busy = await stores.create()
    agent = _BlockingAgent()
    async with anyio.create_task_group() as group:
        group.start_soon(partial(stores.sessions.run_turn, busy, "hi", agent=agent))
        await agent.started.wait()

        with pytest.raises(StorageCapacityError):
            await stores.sessions.create()

        assert stores.ledger.idle_order() == (busy,)
        agent.release.set()

    assert (await stores.sessions.get(busy)).retained_turns == 1


@pytest.mark.anyio
async def test_turn_admission_evicts_idle_sessions_before_the_agent_runs() -> None:
    """A turn reserves room for its reply, evicting idle sessions if needed."""
    # Two sessions plus the reservation is one byte over, so one must go.
    budget = 2 * SESSION_ALLOWANCE_BYTES + len("hi") + TURN_REPLY_ALLOWANCE_BYTES - 1
    stores = _Stores(budget_bytes=budget)
    idle = await stores.create()
    talker = await stores.create()

    agent = _ShortAgent()
    await stores.sessions.run_turn(talker, "hi", agent=agent)

    assert agent.calls == 1
    assert stores.ledger.idle_order() == (talker,)
    with pytest.raises(SessionNotFoundError):
        await stores.sessions.get(idle)
    assert stores.ledger.reserved_bytes == 0


@pytest.mark.anyio
async def test_refused_turn_never_calls_the_agent() -> None:
    """No model call, trace or stored turn when the budget is full."""
    stores = _Stores(budget_bytes=SESSION_ALLOWANCE_BYTES + TURN_REPLY_ALLOWANCE_BYTES)
    talker = await stores.create()
    agent = _ShortAgent()

    with pytest.raises(StorageCapacityError):
        await stores.sessions.run_turn(talker, "hi", agent=agent)

    assert agent.calls == 0
    assert (await stores.sessions.get(talker)).retained_turns == 0
    assert stores.ledger.reserved_bytes == 0


@pytest.mark.anyio
async def test_failed_turn_releases_its_reservation() -> None:
    """The reply reservation is returned even when the agent raises."""
    stores = _Stores(budget_bytes=None)
    talker = await stores.create()

    with pytest.raises(RuntimeError):
        await stores.sessions.run_turn(talker, "hi", agent=_FailingAgent())

    assert stores.ledger.reserved_bytes == 0


@pytest.mark.anyio
async def test_reservation_is_held_while_the_turn_runs() -> None:
    """Room for the reply cannot be taken by other writes mid-turn."""
    stores = _Stores(budget_bytes=None)
    talker = await stores.create()
    agent = _BlockingAgent()
    async with anyio.create_task_group() as group:
        group.start_soon(partial(stores.sessions.run_turn, talker, "hi", agent=agent))
        await agent.started.wait()
        assert stores.ledger.reserved_bytes == len("hi") + TURN_REPLY_ALLOWANCE_BYTES
        agent.release.set()

    assert stores.ledger.reserved_bytes == 0


@pytest.mark.anyio
async def test_turn_waiting_for_the_lock_finds_its_session_evicted() -> None:
    """A queued turn re-checks its session after taking the lock."""
    stores = _Stores(budget_bytes=None)
    talker = await stores.create()
    first = _BlockingAgent()
    queued = _ShortAgent()
    outcome: list[type[BaseException]] = []

    async def queued_turn() -> None:
        try:
            await stores.sessions.run_turn(talker, "second", agent=queued)
        except SessionNotFoundError as error:
            outcome.append(type(error))

    async with anyio.create_task_group() as group:
        group.start_soon(
            partial(stores.sessions.run_turn, talker, "first", agent=first)
        )
        await first.started.wait()
        group.start_soon(queued_turn)
        await anyio.sleep(0)
        # Simulate an eviction that happens between the two turns.
        stores.sessions._remove(talker)  # pyright: ignore[reportPrivateUsage]
        first.release.set()

    assert outcome == [SessionNotFoundError]
    assert queued.calls == 0


@pytest.mark.anyio
async def test_upload_for_a_session_evicted_during_extraction_stores_nothing() -> None:
    """A report is never stored for a session the ledger no longer tracks."""
    stores = _Stores(budget_bytes=None)
    gone = uuid4()

    with pytest.raises(SessionNotFoundError):
        await stores.reports.add(gone, _report(10))

    assert await stores.reports.list_for_session(gone) == ()


@pytest.mark.anyio
async def test_purge_removes_reports_left_by_a_missing_session() -> None:
    """Purge drops every report a session owns and reports how many."""
    reports = InMemoryReportStore(max_reports_per_session=10)
    owner = uuid4()
    await reports.add(owner, _report(5))
    await reports.add(owner, _report(6))

    assert await reports.purge(owner) == 2
    assert await reports.list_for_session(owner) == ()
    assert await reports.purge(owner) == 0


@pytest.mark.anyio
async def test_eviction_and_refusal_logs_carry_sizes_but_no_content(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Each eviction is one structured line; report text never appears."""
    stores = _Stores(budget_bytes=SESSION_ALLOWANCE_BYTES + 2 * KIB)
    victim = await stores.create()
    await stores.reports.add(victim, _report(KIB))

    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        survivor = await stores.create()
        # Only the survivor itself could make room, and it is exempt.
        with pytest.raises(StorageCapacityError):
            await stores.reports.add(survivor, _report(KIB + 1))

    messages = [record.getMessage() for record in caplog.records]
    assert (
        f"toyapp_storage event=evict session_id={victim} reason=budget "
        f"bytes_freed={SESSION_ALLOWANCE_BYTES + 2 * KIB} reports_removed=1"
    ) in messages
    assert any("event=capacity_refused reason=nothing_evictable" in m for m in messages)
    assert any(m.startswith("toyapp_memory event=evict ") for m in messages)
    assert not any("xxxx" in m for m in messages)


def test_budget_must_be_positive() -> None:
    """A zero budget would refuse every write, so it is rejected."""
    with pytest.raises(ValueError, match="budget_bytes"):
        SessionMemoryLedger(budget_bytes=0)


def test_release_cannot_exceed_the_reservation() -> None:
    """Releasing more than was reserved is a bookkeeping bug."""
    ledger = SessionMemoryLedger()
    ledger.reserve(10)

    with pytest.raises(ValueError, match="reserved"):
        ledger.release(11)
    ledger.release(10)
    assert ledger.reserved_bytes == 0
