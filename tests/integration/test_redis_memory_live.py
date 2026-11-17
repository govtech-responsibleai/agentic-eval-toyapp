"""Opt-in checks of the Lua eviction script against a real, disposable Redis.

These are the only tests that run the eviction script; offline tests cover the
budget maths and the stores' bookkeeping. Run them against a dedicated local
container, never a shared Redis, because one test changes `maxmemory`:

    docker run --rm -p 6390:6379 redis:8.2-alpine \\
        redis-server --maxmemory-policy noeviction
    TOYAPP_TEST_REDIS_URL=redis://127.0.0.1:6390/0 make test-live

Every test uses a unique key prefix and deletes only its own keys.
"""

import hashlib
import os
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from typing import Any, cast
from uuid import UUID, uuid4

import pytest
from redis.asyncio import Redis

from agentic_eval_toyapp.models import (
    AgentTurnResult,
    AgentTurnStatus,
    ClarifyAction,
    ConversationMessage,
    ReportMediaType,
    TerminationReason,
)
from agentic_eval_toyapp.redis_memory import (
    EvictionRecord,
    RedisMemoryClient,
    RedisMemoryGuard,
)
from agentic_eval_toyapp.reports import ProcessedReport, RedisReportStore, StoredReport
from agentic_eval_toyapp.sessions import RedisSessionStore

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not os.environ.get("TOYAPP_TEST_REDIS_URL"),
        reason="set TOYAPP_TEST_REDIS_URL to a disposable Redis to run",
    ),
]


class _Agent:
    async def run(
        self,
        user_message: str,
        *,
        history: Sequence[ConversationMessage] = (),
        session_id: str | None = None,
        reports: Sequence[StoredReport] = (),
    ) -> AgentTurnResult:
        del history, session_id, reports
        return AgentTurnResult(
            status=AgentTurnStatus.CLARIFICATION_REQUIRED,
            message=f"Clarify: {user_message}",
            termination_reason=TerminationReason.CLARIFICATION_REQUIRED,
            model_calls=1,
            searches=0,
            model_actions=(ClarifyAction(question=f"Clarify: {user_message}"),),
        )


def _report(size: int) -> ProcessedReport:
    text = "x" * size
    content = text.encode()
    return ProcessedReport(
        filename="r.txt",
        media_type=ReportMediaType.TEXT,
        byte_size=len(content),
        original_bytes=content,
        extracted_text=text,
        sha256=hashlib.sha256(content).hexdigest(),
    )


class _Stack:
    def __init__(self, client: Any, prefix: str, now: list[float]) -> None:
        self.client = client
        self.prefix = prefix
        self.now = now
        self.guard = RedisMemoryGuard(
            key_prefix=prefix, budget_bytes=1 << 40, clock=lambda: now[0]
        )
        self.sessions = RedisSessionStore(
            client,
            key_prefix=prefix,
            memory_guard=self.guard,
        )
        self.reports = RedisReportStore(
            client,
            key_prefix=prefix,
            max_reports_per_session=5,
            memory_guard=self.guard,
        )

    async def keys(self) -> set[str]:
        return {key async for key in self.client.scan_iter(f"{self.prefix}:*")}

    async def session_keys(self, session_id: UUID) -> set[str]:
        return {key for key in await self.keys() if str(session_id) in key}

    async def evict(
        self, needed: int, current: UUID | None = None
    ) -> tuple[EvictionRecord, ...]:
        return await self.guard._evict(  # pyright: ignore[reportPrivateUsage]
            cast(RedisMemoryClient, self.client), needed, current
        )

    async def new_session(self, at: float, *, reports: int = 0) -> UUID:
        self.now[0] = at
        session_id = (await self.sessions.create()).session_id
        for _ in range(reports):
            await self.reports.add(session_id, _report(2_000))
        return session_id


@asynccontextmanager
async def _stack() -> AsyncGenerator[_Stack]:
    client: Any = Redis.from_url(  # pyright: ignore[reportUnknownMemberType]
        os.environ["TOYAPP_TEST_REDIS_URL"], decode_responses=True
    )
    stack = _Stack(client, f"toyapp-live-{uuid4().hex}", [1_000.0])
    try:
        yield stack
    finally:
        keys = await stack.keys()
        if keys:
            await client.delete(*keys)
        await client.aclose()


@pytest.mark.anyio
async def test_eviction_removes_the_longest_idle_session_completely() -> None:
    async with _stack() as stack:
        oldest = await stack.new_session(1_000.0, reports=2)
        newer = await stack.new_session(2_000.0, reports=1)
        assert len(await stack.session_keys(oldest)) == 4

        (record,) = await stack.evict(needed=1)

        assert record.session_id == str(oldest)
        assert record.reason == "memory_budget"
        assert record.reports_removed == 2
        assert record.bytes_freed > 4_000
        assert await stack.session_keys(oldest) == set()
        assert str(oldest) not in await stack.client.zrange(stack.guard.lru_key, 0, -1)
        sizes = await stack.client.hkeys(stack.guard.bytes_key)
        assert all(not field.startswith(str(oldest)) for field in sizes)
        assert len(await stack.session_keys(newer)) == 3


@pytest.mark.anyio
async def test_eviction_continues_until_enough_is_freed_in_idle_order() -> None:
    async with _stack() as stack:
        ids = [await stack.new_session(1_000.0 + n, reports=1) for n in range(4)]

        records = await stack.evict(needed=5_000)

        assert [r.session_id for r in records] == [str(ids[0]), str(ids[1])]
        for kept in ids[2:]:
            assert await stack.session_keys(kept)


@pytest.mark.anyio
async def test_eviction_skips_the_current_and_locked_sessions() -> None:
    async with _stack() as stack:
        current = await stack.new_session(1_000.0)
        turn_locked = await stack.new_session(1_001.0)
        report_locked = await stack.new_session(1_002.0)
        idle = await stack.new_session(1_003.0)
        await stack.client.set(f"{stack.prefix}:session:{turn_locked}:turn-lock", "t")
        await stack.client.set(
            f"{stack.prefix}:reports:{report_locked}:mutation-lock", "t"
        )

        records = await stack.evict(needed=1 << 30, current=current)

        assert [r.session_id for r in records] == [str(idle)]
        for kept in (current, turn_locked, report_locked):
            assert f"{stack.prefix}:session:{kept}" in await stack.keys()


@pytest.mark.anyio
async def test_a_seeded_demo_session_is_never_a_candidate() -> None:
    async with _stack() as stack:
        demo = UUID("00000000-0000-0000-0000-00000000de00")
        await stack.sessions.seed(demo, (), read_only=True)
        await stack.sessions.get(demo)

        assert await stack.evict(needed=1 << 30) == ()
        assert f"{stack.prefix}:session:{demo}" in await stack.keys()


@pytest.mark.anyio
async def test_an_expired_session_is_swept_without_counting_as_freed() -> None:
    async with _stack() as stack:
        expired = await stack.new_session(1_000.0, reports=1)
        live = await stack.new_session(2_000.0)
        await stack.client.delete(f"{stack.prefix}:session:{expired}")

        records = await stack.evict(needed=1)

        assert [(r.session_id, r.reason) for r in records] == [
            (str(expired), "expired"),
            (str(live), "memory_budget"),
        ]
        assert records[0].bytes_freed == 0
        assert await stack.session_keys(expired) == set()


@pytest.mark.anyio
async def test_two_concurrent_passes_never_evict_a_session_twice() -> None:
    async with _stack() as stack:
        for n in range(6):
            await stack.new_session(1_000.0 + n)

        first, second = await stack.evict(needed=1), await stack.evict(needed=1)

        assert {r.session_id for r in first}.isdisjoint(r.session_id for r in second)


@pytest.mark.anyio
async def test_noeviction_with_a_small_maxmemory_keeps_accepting_writes() -> None:
    """Load past the budget: idle sessions go, and Redis never rejects a write."""
    async with _stack() as stack:
        info = await stack.client.info("memory")
        if info.get("maxmemory_policy") != "noeviction":
            pytest.skip("start the container with --maxmemory-policy noeviction")
        original = await stack.client.config_get("maxmemory")
        used = int(info["used_memory"])
        await stack.client.config_set("maxmemory", used + 16 * 1024 * 1024)
        guard = RedisMemoryGuard(key_prefix=stack.prefix, clock=lambda: stack.now[0])
        sessions = RedisSessionStore(
            stack.client,
            key_prefix=stack.prefix,
            memory_guard=guard,
        )
        reports = RedisReportStore(
            stack.client,
            key_prefix=stack.prefix,
            max_reports_per_session=5,
            memory_guard=guard,
        )
        created: list[UUID] = []
        try:
            # Any refusal, including a Redis OOM reply wrapped as a store
            # outage, propagates and fails the test.
            for n in range(40):
                stack.now[0] = 10_000.0 + n
                session_id = (await sessions.create()).session_id
                created.append(session_id)
                await reports.add(session_id, _report(200_000))
                await sessions.run_turn(session_id, "hello", agent=_Agent())
        finally:
            await stack.client.config_set("maxmemory", original["maxmemory"])

        survivors = [s for s in created if await stack.session_keys(s)]
        assert created[-1] in survivors
        assert len(survivors) < len(created)
        assert survivors == created[-len(survivors) :]
