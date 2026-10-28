"""Offline proofs that retrieval runs through a fixed single-slot limiter."""

import logging
import threading
import time
from collections.abc import Callable
from datetime import date

import anyio
import pytest
from anyio import CapacityLimiter

from agentic_eval_toyapp.agent import RagAgent
from agentic_eval_toyapp.models import (
    AgentTurnResult,
    AnswerAction,
    DocumentMetadata,
    RetrievedChunk,
    SearchAction,
)
from agentic_eval_toyapp.providers import ScriptedGenerationProvider
from agentic_eval_toyapp.retrieval import AsyncRetriever

_SECRET_QUERY = "secret-query-text"


def _chunk(chunk_id: str = "LEAVE-001#1") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="LEAVE-001",
        title="Leave Policy",
        text="Employees receive 18 days of annual leave.",
        metadata=DocumentMetadata(topic="leave", effective_date=date(2026, 1, 1)),
        similarity_score=0.9,
    )


class _OverlapRetriever:
    """Record how many searches are inside the synchronous fake at once."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.calls: list[tuple[str, int | None]] = []

    def search(
        self, query: str, *, top_k: int | None = None
    ) -> tuple[RetrievedChunk, ...]:
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.calls.append((query, top_k))
        time.sleep(0.005)
        with self._lock:
            self.active -= 1
        return (_chunk(query),)


class _GatedRetriever:
    """Hold the first search inside the worker until the test releases it."""

    def __init__(self, *, fail: bool = False) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.fail = fail
        self.calls = 0

    def search(
        self, query: str, *, top_k: int | None = None
    ) -> tuple[RetrievedChunk, ...]:
        del top_k
        self.calls += 1
        if self.calls == 1:
            self.entered.set()
            assert self.release.wait(timeout=5)
            if self.fail:
                raise RuntimeError("boom")
        return (_chunk(query),)


async def _wait_until(predicate: Callable[[], bool]) -> None:
    with anyio.fail_after(5):
        while not predicate():
            await anyio.sleep(0.001)


@pytest.mark.anyio
async def test_twenty_concurrent_searches_never_overlap() -> None:
    fake = _OverlapRetriever()
    adapter = AsyncRetriever(fake, CapacityLimiter(1))
    results: dict[int, tuple[RetrievedChunk, ...]] = {}

    async def run(index: int) -> None:
        results[index] = await adapter.search(f"query-{index}", top_k=index + 1)

    async with anyio.create_task_group() as group:
        for index in range(20):
            group.start_soon(run, index)

    assert fake.peak == 1
    assert len(fake.calls) == 20
    assert sorted(fake.calls) == sorted((f"query-{i}", i + 1) for i in range(20))
    assert all(results[i][0].chunk_id == f"query-{i}" for i in range(20))


@pytest.mark.anyio
async def test_search_passes_arguments_and_results_through() -> None:
    fake = _OverlapRetriever()
    adapter = AsyncRetriever(fake, CapacityLimiter(1))

    default = await adapter.search("one")
    explicit = await adapter.search("two", top_k=5)

    assert fake.calls == [("one", None), ("two", 5)]
    assert [default[0].chunk_id, explicit[0].chunk_id] == ["one", "two"]


@pytest.mark.anyio
async def test_slot_is_released_when_search_raises() -> None:
    fake = _GatedRetriever(fail=True)
    limiter = CapacityLimiter(1)
    adapter = AsyncRetriever(fake, limiter)
    outcome: list[BaseException] = []

    async def failing() -> None:
        try:
            await adapter.search("first")
        except RuntimeError as error:
            outcome.append(error)

    async with anyio.create_task_group() as group:
        group.start_soon(failing)
        await _wait_until(fake.entered.is_set)
        fake.release.set()

    assert len(outcome) == 1
    assert limiter.statistics().borrowed_tokens == 0
    assert (await adapter.search("second"))[0].chunk_id == "second"


@pytest.mark.anyio
async def test_cancelling_a_waiting_search_leaks_nothing() -> None:
    fake = _GatedRetriever()
    limiter = CapacityLimiter(1)
    adapter = AsyncRetriever(fake, limiter)

    async with anyio.create_task_group() as group:
        group.start_soon(adapter.search, "holder")
        await _wait_until(fake.entered.is_set)

        async with anyio.create_task_group() as inner:
            inner.start_soon(adapter.search, "waiter")
            await _wait_until(lambda: limiter.statistics().tasks_waiting == 1)
            inner.cancel_scope.cancel()

        assert limiter.statistics().tasks_waiting == 0
        fake.release.set()

    assert fake.calls == 1
    assert limiter.statistics().borrowed_tokens == 0
    assert (await adapter.search("later"))[0].chunk_id == "later"


@pytest.mark.anyio
async def test_contended_searches_log_queue_depth_and_wait(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake = _GatedRetriever()
    limiter = CapacityLimiter(1)
    adapter = AsyncRetriever(fake, limiter)

    with caplog.at_level(logging.INFO):
        async with anyio.create_task_group() as group:
            group.start_soon(adapter.search, _SECRET_QUERY)
            await _wait_until(fake.entered.is_set)
            group.start_soon(adapter.search, _SECRET_QUERY)
            await _wait_until(lambda: limiter.statistics().tasks_waiting == 1)
            group.start_soon(adapter.search, _SECRET_QUERY)
            await _wait_until(lambda: limiter.statistics().tasks_waiting == 2)
            fake.release.set()

    lines = [r.getMessage() for r in caplog.records if "toyapp_retrieval" in r.message]
    assert len(lines) == 2
    assert sorted("queue_depth=0" in line for line in lines) == [False, True]
    assert any("queue_depth=1" in line for line in lines)
    assert all("wait_ms=" in line for line in lines)
    assert all(_SECRET_QUERY not in line for line in lines)


@pytest.mark.anyio
async def test_uncontended_search_logs_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    adapter = AsyncRetriever(_OverlapRetriever(), CapacityLimiter(1))

    with caplog.at_level(logging.DEBUG):
        await adapter.search("alone")

    assert not [r for r in caplog.records if "toyapp_retrieval" in r.message]


@pytest.mark.anyio
async def test_adapter_runs_search_off_the_event_loop_thread() -> None:
    seen: list[int] = []

    class _Probe:
        def search(
            self, query: str, *, top_k: int | None = None
        ) -> tuple[RetrievedChunk, ...]:
            del query, top_k
            seen.append(threading.get_ident())
            return ()

    await AsyncRetriever(_Probe(), CapacityLimiter(1)).search("q")

    assert seen and seen[0] != threading.get_ident()


@pytest.mark.anyio
async def test_twenty_concurrent_agent_turns_never_overlap_searches() -> None:
    fake = _OverlapRetriever()
    adapter = AsyncRetriever(fake, CapacityLimiter(1))
    answer = AnswerAction(markdown="Answer. [q#1]", citations=("q#1",))

    def agent() -> RagAgent:
        return RagAgent(
            provider=ScriptedGenerationProvider(
                (SearchAction(query="annual leave"), answer)
            ),
            retriever=adapter,
            system_prompt="Search, then answer with citations.",
        )

    results: list[AgentTurnResult] = []

    async def turn() -> None:
        results.append(await agent().run("How much leave?"))

    async with anyio.create_task_group() as group:
        for _ in range(20):
            group.start_soon(turn)

    assert fake.peak == 1
    assert len(fake.calls) == 20
    assert len(results) == 20
    assert all(result.searches == 1 for result in results)
