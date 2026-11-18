"""Tests for lazy idle expiry of process-local sessions."""

import logging
from functools import partial
from uuid import UUID

import anyio
import pytest
from tests.support.memory_stores import (
    KIB,
    BlockingAgent,
    MemoryStores,
    ShortAgent,
    text_report,
)

from agentic_eval_toyapp.session_memory import (
    SESSION_ALLOWANCE_BYTES,
    SessionMemoryLedger,
    SessionNotFoundError,
)

TTL = 600.0


def _stores(budget_bytes: int | None = None) -> MemoryStores:
    return MemoryStores(budget_bytes=budget_bytes, idle_ttl_seconds=TTL)


async def _is_gone(stores: MemoryStores, session_id: UUID) -> bool:
    try:
        await stores.sessions.get(session_id)
    except SessionNotFoundError:
        return True
    return False


@pytest.mark.anyio
async def test_session_idle_for_exactly_the_ttl_is_expired() -> None:
    """The TTL is inclusive: idle for the full TTL means expired."""
    stores = _stores()
    idle = await stores.create()
    stores.clock.advance(TTL - 1)  # create() already advanced by 1

    await stores.sessions.create()

    assert await _is_gone(stores, idle)


@pytest.mark.anyio
async def test_session_just_inside_the_ttl_survives() -> None:
    """A session idle for a moment less than the TTL is kept."""
    stores = _stores()
    idle = await stores.create()
    stores.clock.advance(TTL - 1 - 0.001)

    await stores.sessions.create()

    assert not await _is_gone(stores, idle)


@pytest.mark.anyio
async def test_expired_session_takes_its_reports_and_bytes_with_it() -> None:
    """Expiry removes turns, reports and the session's estimate together."""
    stores = _stores()
    idle = await stores.create()
    await stores.reports.add(idle, text_report(KIB))
    stores.clock.advance(TTL)

    fresh = await stores.create()

    assert await _is_gone(stores, idle)
    assert await stores.reports.list_for_session(idle) == ()
    assert stores.ledger.footprint(idle) is None
    assert stores.ledger.idle_order() == (fresh,)
    assert stores.ledger.total_bytes == SESSION_ALLOWANCE_BYTES


@pytest.mark.anyio
async def test_one_sweep_expires_every_stale_session_and_stops_at_a_fresh_one() -> None:
    """All sessions past the TTL go in one pass; recent ones stay."""
    stores = _stores()
    stale = [await stores.create() for _ in range(3)]
    stores.clock.advance(TTL)
    recent = await stores.create()

    await stores.sessions.create()

    for session_id in stale:
        assert await _is_gone(stores, session_id)
    assert not await _is_gone(stores, recent)


@pytest.mark.anyio
async def test_a_read_keeps_a_session_from_expiring() -> None:
    """Any access renews the idle clock."""
    stores = _stores()
    session_id = await stores.create()
    stores.clock.advance(TTL - 10)
    await stores.sessions.get(session_id)
    stores.clock.advance(TTL - 10)

    await stores.sessions.create()

    assert not await _is_gone(stores, session_id)


@pytest.mark.anyio
async def test_upload_triggers_the_sweep() -> None:
    """Adding a report checks the budget, so it expires idle sessions."""
    stores = _stores()
    idle = await stores.create()
    uploader = await stores.create()
    stores.clock.advance(TTL)
    await stores.sessions.get(uploader)

    await stores.reports.add(uploader, text_report(10))

    assert await _is_gone(stores, idle)
    assert not await _is_gone(stores, uploader)


@pytest.mark.anyio
async def test_turn_triggers_the_sweep_but_never_expires_the_talker() -> None:
    """Starting a turn sweeps, skipping the requesting session."""
    stores = _stores()
    idle = await stores.create()
    talker = await stores.create()
    stores.clock.advance(TTL)
    # The talker is past the TTL too, but it is the requesting session.
    agent = ShortAgent()

    await stores.sessions.run_turn(talker, "hi", agent=agent)

    assert agent.calls == 1
    assert await _is_gone(stores, idle)
    assert stores.ledger.idle_order() == (talker,)


@pytest.mark.anyio
async def test_busy_session_is_skipped_and_the_sweep_continues_past_it() -> None:
    """A session mid-turn is never expired, but those behind it still are."""
    stores = _stores()
    busy = await stores.create()
    stale = await stores.create()
    agent = BlockingAgent()
    async with anyio.create_task_group() as group:
        group.start_soon(partial(stores.sessions.run_turn, busy, "hi", agent=agent))
        await agent.started.wait()
        # Make the busy session look idle again, ahead of the stale one.
        stores.ledger._entries.move_to_end(stale)  # pyright: ignore[reportPrivateUsage]
        stores.clock.advance(TTL * 2)

        await stores.sessions.create()

        assert not await _is_gone(stores, busy)
        assert await _is_gone(stores, stale)
        agent.release.set()


@pytest.mark.anyio
async def test_demo_session_never_expires() -> None:
    """Untracked demo sessions are outside the sweep."""
    stores = _stores()
    demo_id = stores.seed_demo()
    stores.clock.advance(TTL * 10)

    await stores.sessions.create()

    assert (await stores.sessions.get(demo_id)).read_only is True


@pytest.mark.anyio
async def test_expiry_frees_room_before_any_active_session_is_evicted() -> None:
    """Expired sessions are removed first, so active ones keep their place."""
    stores = _stores(budget_bytes=2 * SESSION_ALLOWANCE_BYTES)
    stale = await stores.create()
    stores.clock.advance(TTL)
    active = await stores.create()

    newcomer = await stores.create()

    assert await _is_gone(stores, stale)
    assert stores.ledger.idle_order() == (active, newcomer)


@pytest.mark.anyio
async def test_no_ttl_means_nothing_ever_expires() -> None:
    """A ledger without a TTL keeps idle sessions forever."""
    stores = MemoryStores(budget_bytes=None)
    idle = await stores.create()
    stores.clock.advance(10**9)

    await stores.sessions.create()

    assert not await _is_gone(stores, idle)


@pytest.mark.anyio
async def test_expiry_logs_reason_idle_with_sizes_and_no_content(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Each expiry is one structured line; content never appears."""
    stores = _stores()
    idle = await stores.create()
    await stores.reports.add(idle, text_report(KIB))
    stores.clock.advance(TTL)

    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        await stores.sessions.create()

    messages = [record.getMessage() for record in caplog.records]
    assert (
        f"toyapp_storage event=expire session_id={idle} reason=idle "
        f"bytes_freed={SESSION_ALLOWANCE_BYTES + 2 * KIB} reports_removed=1"
    ) in messages
    assert any(m.startswith("toyapp_memory event=expire ") for m in messages)
    assert not any("xxxx" in m for m in messages)


@pytest.mark.parametrize("ttl", [0.0, -1.0])
def test_ttl_must_be_positive(ttl: float) -> None:
    """A zero TTL would expire every session at once."""
    with pytest.raises(ValueError, match="idle_ttl_seconds"):
        SessionMemoryLedger(idle_ttl_seconds=ttl)
