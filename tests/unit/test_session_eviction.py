"""Tests for budget eviction shared by the in-memory session and report stores."""

import logging
from functools import partial
from uuid import UUID, uuid4

import anyio
import pytest
from tests.support.memory_stores import (
    KIB,
    BlockingAgent,
    FailingAgent,
    MemoryStores,
    ShortAgent,
    text_report,
)

from agentic_eval_toyapp.redis_memory import (
    TURN_REPLY_ALLOWANCE_BYTES,
    StorageCapacityError,
)
from agentic_eval_toyapp.reports import InMemoryReportStore, ReportNotFoundError
from agentic_eval_toyapp.session_memory import (
    SESSION_ALLOWANCE_BYTES,
    SessionMemoryLedger,
    SessionNotFoundError,
)


async def _assert_gone(stores: MemoryStores, session_id: UUID, file_id: UUID) -> None:
    with pytest.raises(SessionNotFoundError):
        await stores.sessions.get(session_id)
    assert await stores.reports.list_for_session(session_id) == ()
    with pytest.raises(ReportNotFoundError):
        await stores.reports.resolve(session_id, (file_id,))
    assert stores.ledger.footprint(session_id) is None


@pytest.mark.anyio
async def test_longest_idle_session_is_evicted_first() -> None:
    """With room for three sessions, the fourth evicts the oldest."""
    stores = MemoryStores(budget_bytes=3 * SESSION_ALLOWANCE_BYTES)
    oldest, middle, newest = [await stores.create() for _ in range(3)]

    fourth = await stores.create()

    assert stores.ledger.idle_order() == (middle, newest, fourth)
    with pytest.raises(SessionNotFoundError):
        await stores.sessions.get(oldest)


@pytest.mark.anyio
async def test_a_recently_used_session_survives_eviction() -> None:
    """An access moves a session to the back of the queue."""
    stores = MemoryStores(budget_bytes=3 * SESSION_ALLOWANCE_BYTES)
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
    stores = MemoryStores(budget_bytes=SESSION_ALLOWANCE_BYTES + 2 * KIB + 1)
    victim = await stores.create()
    stored = await stores.reports.add(victim, text_report(KIB))
    assert stores.ledger.total_bytes == SESSION_ALLOWANCE_BYTES + 2 * KIB

    survivor = await stores.create()

    await _assert_gone(stores, victim, stored.file_id)
    assert stores.ledger.idle_order() == (survivor,)
    assert stores.ledger.total_bytes == SESSION_ALLOWANCE_BYTES


@pytest.mark.anyio
async def test_only_as_many_sessions_as_needed_are_evicted() -> None:
    """Eviction stops as soon as the incoming write fits."""
    stores = MemoryStores(budget_bytes=4 * SESSION_ALLOWANCE_BYTES)
    ids = [await stores.create() for _ in range(4)]

    await stores.create()

    assert stores.ledger.idle_order()[:3] == tuple(ids[1:])


@pytest.mark.anyio
async def test_large_upload_evicts_several_idle_sessions() -> None:
    """One write may need more than one victim."""
    stores = MemoryStores(budget_bytes=4 * SESSION_ALLOWANCE_BYTES)
    first, second, third = [await stores.create() for _ in range(3)]
    uploader = await stores.create()

    await stores.reports.add(uploader, text_report(SESSION_ALLOWANCE_BYTES))

    assert stores.ledger.idle_order() == (third, uploader)
    for gone in (first, second):
        with pytest.raises(SessionNotFoundError):
            await stores.sessions.get(gone)


@pytest.mark.anyio
async def test_write_exactly_at_the_budget_fits_and_one_byte_more_evicts() -> None:
    """The budget is inclusive."""
    stores = MemoryStores(budget_bytes=SESSION_ALLOWANCE_BYTES + 2 * KIB)
    first = await stores.create()
    await stores.reports.add(first, text_report(KIB))
    assert stores.ledger.total_bytes == stores.ledger.budget_bytes

    stores.ledger.ensure_room(0)
    assert stores.ledger.idle_order() == (first,)

    stores.ledger.ensure_room(1)
    assert stores.ledger.idle_order() == ()


@pytest.mark.anyio
async def test_empty_session_at_a_one_session_budget_evicts_the_previous() -> None:
    """Empty sessions cost the allowance, so they are bounded too."""
    stores = MemoryStores(budget_bytes=SESSION_ALLOWANCE_BYTES)
    first = await stores.create()

    second = await stores.create()

    assert stores.ledger.idle_order() == (second,)
    with pytest.raises(SessionNotFoundError):
        await stores.sessions.get(first)


@pytest.mark.anyio
async def test_item_larger_than_the_budget_is_refused_without_eviction() -> None:
    """Evicting everyone would not help, so nobody is evicted."""
    stores = MemoryStores(budget_bytes=4 * SESSION_ALLOWANCE_BYTES)
    others = [await stores.create() for _ in range(2)]
    uploader = await stores.create()

    with pytest.raises(StorageCapacityError):
        await stores.reports.add(uploader, text_report(2 * SESSION_ALLOWANCE_BYTES + 1))

    assert stores.ledger.idle_order() == (*others, uploader)
    assert await stores.reports.list_for_session(uploader) == ()


@pytest.mark.anyio
async def test_too_few_candidates_means_no_session_is_evicted() -> None:
    """Planning happens before removal, so a doomed write evicts nothing."""
    stores = MemoryStores(budget_bytes=3 * SESSION_ALLOWANCE_BYTES)
    other = await stores.create()
    uploader = await stores.create()
    await stores.reports.add(uploader, text_report(SESSION_ALLOWANCE_BYTES // 4))
    before = stores.ledger.total_bytes

    with pytest.raises(StorageCapacityError):
        await stores.reports.add(uploader, text_report(SESSION_ALLOWANCE_BYTES))

    assert stores.ledger.idle_order() == (other, uploader)
    assert stores.ledger.total_bytes == before


@pytest.mark.anyio
async def test_requesting_session_is_never_evicted_for_its_own_upload() -> None:
    """When only the uploader could be evicted, the upload is refused."""
    stores = MemoryStores(budget_bytes=SESSION_ALLOWANCE_BYTES + 2 * KIB)
    uploader = await stores.create()
    await stores.reports.add(uploader, text_report(KIB // 2))

    with pytest.raises(StorageCapacityError):
        await stores.reports.add(uploader, text_report(KIB))

    assert stores.ledger.idle_order() == (uploader,)
    assert len(await stores.reports.list_for_session(uploader)) == 1


@pytest.mark.anyio
async def test_demo_session_is_never_evicted() -> None:
    """The demo is untracked, so a full budget with only the demo refuses."""
    stores = MemoryStores(budget_bytes=SESSION_ALLOWANCE_BYTES)
    demo_id = stores.seed_demo()
    await stores.create()
    await stores.sessions.get(demo_id)

    await stores.create()

    assert (await stores.sessions.get(demo_id)).read_only is True
    assert stores.ledger.footprint(demo_id) is None


@pytest.mark.anyio
async def test_session_in_the_middle_of_a_turn_is_never_evicted() -> None:
    """A held turn lock exempts the session; the write is refused instead."""
    stores = MemoryStores(
        budget_bytes=SESSION_ALLOWANCE_BYTES + TURN_REPLY_ALLOWANCE_BYTES + 10
    )
    busy = await stores.create()
    agent = BlockingAgent()
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
    stores = MemoryStores(budget_bytes=budget)
    idle = await stores.create()
    talker = await stores.create()

    agent = ShortAgent()
    await stores.sessions.run_turn(talker, "hi", agent=agent)

    assert agent.calls == 1
    assert stores.ledger.idle_order() == (talker,)
    with pytest.raises(SessionNotFoundError):
        await stores.sessions.get(idle)
    assert stores.ledger.reserved_bytes == 0


@pytest.mark.anyio
async def test_refused_turn_never_calls_the_agent() -> None:
    """No model call, trace or stored turn when the budget is full."""
    stores = MemoryStores(
        budget_bytes=SESSION_ALLOWANCE_BYTES + TURN_REPLY_ALLOWANCE_BYTES
    )
    talker = await stores.create()
    agent = ShortAgent()

    with pytest.raises(StorageCapacityError):
        await stores.sessions.run_turn(talker, "hi", agent=agent)

    assert agent.calls == 0
    assert (await stores.sessions.get(talker)).retained_turns == 0
    assert stores.ledger.reserved_bytes == 0


@pytest.mark.anyio
async def test_failed_turn_releases_its_reservation() -> None:
    """The reply reservation is returned even when the agent raises."""
    stores = MemoryStores(budget_bytes=None)
    talker = await stores.create()

    with pytest.raises(RuntimeError):
        await stores.sessions.run_turn(talker, "hi", agent=FailingAgent())

    assert stores.ledger.reserved_bytes == 0


@pytest.mark.anyio
async def test_reservation_is_held_while_the_turn_runs() -> None:
    """Room for the reply cannot be taken by other writes mid-turn."""
    stores = MemoryStores(budget_bytes=None)
    talker = await stores.create()
    agent = BlockingAgent()
    async with anyio.create_task_group() as group:
        group.start_soon(partial(stores.sessions.run_turn, talker, "hi", agent=agent))
        await agent.started.wait()
        assert stores.ledger.reserved_bytes == len("hi") + TURN_REPLY_ALLOWANCE_BYTES
        agent.release.set()

    assert stores.ledger.reserved_bytes == 0


@pytest.mark.anyio
async def test_turn_waiting_for_the_lock_finds_its_session_evicted() -> None:
    """A queued turn re-checks its session after taking the lock."""
    stores = MemoryStores(budget_bytes=None)
    talker = await stores.create()
    first = BlockingAgent()
    queued = ShortAgent()
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
    stores = MemoryStores(budget_bytes=None)
    gone = uuid4()

    with pytest.raises(SessionNotFoundError):
        await stores.reports.add(gone, text_report(10))

    assert await stores.reports.list_for_session(gone) == ()


@pytest.mark.anyio
async def test_purge_removes_reports_left_by_a_missing_session() -> None:
    """Purge drops every report a session owns and reports how many."""
    reports = InMemoryReportStore(max_reports_per_session=10)
    owner = uuid4()
    await reports.add(owner, text_report(5))
    await reports.add(owner, text_report(6))

    assert await reports.purge(owner) == 2
    assert await reports.list_for_session(owner) == ()
    assert await reports.purge(owner) == 0


@pytest.mark.anyio
async def test_eviction_and_refusal_logs_carry_sizes_but_no_content(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Each eviction is one structured line; report text never appears."""
    stores = MemoryStores(budget_bytes=SESSION_ALLOWANCE_BYTES + 2 * KIB)
    victim = await stores.create()
    await stores.reports.add(victim, text_report(KIB))

    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        survivor = await stores.create()
        # Only the survivor itself could make room, and it is exempt.
        with pytest.raises(StorageCapacityError):
            await stores.reports.add(survivor, text_report(KIB + 1))

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
