"""Tests for turn admission and the shared model-call gate."""

import asyncio
from typing import Any

import anyio
import pytest

from agentic_eval_toyapp.backpressure import (
    RateLimitedCompletionClient,
    TurnAdmission,
)
from agentic_eval_toyapp.providers import StructuredCompletion

_TIMEOUT_SECONDS = 5.0


class _FakeTime:
    """Deterministic clock whose sleepers wake only when the test advances time."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []
        self._waiters: list[tuple[float, asyncio.Event]] = []

    def clock(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        event = asyncio.Event()
        self._waiters.append((self.now + seconds, event))
        await event.wait()

    async def settle(self) -> None:
        """Let every runnable task reach its next await."""
        for _ in range(20):
            await anyio.sleep(0)

    async def advance_until(self, done: "asyncio.Future[Any] | asyncio.Task[Any]"):
        """Wake the earliest sleeper until the awaited work has finished."""
        while True:
            await self.settle()
            if done.done():
                return
            assert self._waiters, "work is blocked on something other than time"
            self._waiters.sort(key=lambda waiter: waiter[0])
            wake_at, event = self._waiters.pop(0)
            self.now = max(self.now, wake_at)
            event.set()


class _RecordingDelegate:
    """Records when each call enters and leaves, optionally holding calls open."""

    def __init__(self, fake_time: _FakeTime, *, gated: bool = False) -> None:
        self._time = fake_time
        self.gate = asyncio.Event()
        if not gated:
            self.gate.set()
        self.start_times: list[float] = []
        self.in_flight = 0
        self.peak = 0
        self.calls = 0
        self.fail_with: Exception | None = None
        self.events: list[str] = []

    async def create(
        self,
        *,
        model: str,
        messages: tuple[dict[str, str], ...],
        response_schema: dict[str, Any],
        temperature: float,
        max_tokens: int,
    ) -> str | StructuredCompletion:
        del model, messages, response_schema, temperature, max_tokens
        self.events.append("delegate_entered")
        self.calls += 1
        self.start_times.append(self._time.now)
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await self.gate.wait()
            if self.fail_with is not None:
                raise self.fail_with
            return "ok"
        finally:
            self.in_flight -= 1


def _gate(
    delegate: _RecordingDelegate,
    fake_time: _FakeTime,
    *,
    per_minute: int = 60,
    concurrent: int = 16,
    unpaced: bool = False,
) -> RateLimitedCompletionClient:
    """Build a gate; ``unpaced`` makes sleeps instant so only the semaphore binds."""

    async def no_wait(_seconds: float) -> None:
        return None

    return RateLimitedCompletionClient(
        delegate,
        max_requests_per_minute=per_minute,
        max_concurrent_calls=concurrent,
        clock=fake_time.clock,
        sleep=no_wait if unpaced else fake_time.sleep,
    )


async def _call(client: RateLimitedCompletionClient) -> str | StructuredCompletion:
    return await client.create(
        model="m",
        messages=({"role": "user", "content": "hi"},),
        response_schema={},
        temperature=0.0,
        max_tokens=8,
    )


def test_turn_admission_rejects_a_limit_below_one() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        TurnAdmission(0)


def test_turn_admission_admits_exactly_up_to_the_limit() -> None:
    admission = TurnAdmission(2)

    assert admission.try_acquire() is True
    assert admission.try_acquire() is True
    assert admission.try_acquire() is False
    assert admission.pending == 2

    admission.release()

    assert admission.try_acquire() is True


def test_turn_admission_rejects_an_unmatched_release() -> None:
    with pytest.raises(RuntimeError, match="without a matching acquire"):
        TurnAdmission(1).release()


@pytest.mark.parametrize(
    ("per_minute", "concurrent"),
    [(0, 1), (1, 0), (-1, 1)],
)
def test_gate_rejects_limits_below_one(per_minute: int, concurrent: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        RateLimitedCompletionClient(
            _RecordingDelegate(_FakeTime()),
            max_requests_per_minute=per_minute,
            max_concurrent_calls=concurrent,
        )


def test_gate_can_be_built_without_a_running_loop() -> None:
    RateLimitedCompletionClient(
        _RecordingDelegate(_FakeTime()),
        max_requests_per_minute=60,
        max_concurrent_calls=1,
    )


@pytest.mark.anyio
async def test_call_starts_are_spaced_evenly_at_the_configured_rate() -> None:
    fake_time = _FakeTime()
    delegate = _RecordingDelegate(fake_time)
    client = _gate(delegate, fake_time, per_minute=120)  # one start per 0.5 s

    work = asyncio.gather(*(_call(client) for _ in range(6)))
    await asyncio.wait_for(fake_time.advance_until(work), _TIMEOUT_SECONDS)

    assert delegate.start_times == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])


@pytest.mark.anyio
async def test_starts_never_exceed_the_rate_in_any_one_minute_window() -> None:
    fake_time = _FakeTime()
    delegate = _RecordingDelegate(fake_time)
    client = _gate(delegate, fake_time, per_minute=180)

    work = asyncio.gather(*(_call(client) for _ in range(400)))
    await asyncio.wait_for(fake_time.advance_until(work), _TIMEOUT_SECONDS)

    starts = delegate.start_times
    # Any 181 consecutive starts must span at least one full minute.
    assert (
        min(starts[i + 180] - starts[i] for i in range(len(starts) - 180))
        >= 60.0 - 1e-9
    )


@pytest.mark.anyio
async def test_an_idle_gate_grants_no_burst_credit() -> None:
    fake_time = _FakeTime()
    delegate = _RecordingDelegate(fake_time)
    client = _gate(delegate, fake_time, per_minute=60)

    first = asyncio.ensure_future(_call(client))
    await asyncio.wait_for(fake_time.advance_until(first), _TIMEOUT_SECONDS)
    fake_time.now += 600.0  # a long idle gap

    burst = asyncio.gather(*(_call(client) for _ in range(3)))
    await asyncio.wait_for(fake_time.advance_until(burst), _TIMEOUT_SECONDS)

    assert delegate.start_times == pytest.approx([0.0, 600.0, 601.0, 602.0])


@pytest.mark.anyio
async def test_a_first_call_on_an_idle_gate_does_not_wait() -> None:
    fake_time = _FakeTime()
    delegate = _RecordingDelegate(fake_time)

    await asyncio.wait_for(_call(_gate(delegate, fake_time)), _TIMEOUT_SECONDS)

    assert fake_time.sleeps == []


@pytest.mark.anyio
async def test_never_more_than_the_limit_are_inside_the_delegate() -> None:
    concurrent_limit, extra = 3, 4
    fake_time = _FakeTime()
    delegate = _RecordingDelegate(fake_time, gated=True)
    client = _gate(delegate, fake_time, concurrent=concurrent_limit, unpaced=True)

    tasks = [
        asyncio.ensure_future(_call(client)) for _ in range(concurrent_limit + extra)
    ]
    await fake_time.settle()

    assert delegate.in_flight == concurrent_limit
    assert delegate.peak == concurrent_limit

    delegate.gate.set()
    work = asyncio.gather(*tasks)
    await asyncio.wait_for(fake_time.advance_until(work), _TIMEOUT_SECONDS)

    assert delegate.calls == concurrent_limit + extra
    assert delegate.peak == concurrent_limit


@pytest.mark.anyio
async def test_waiting_happens_before_the_delegate_is_entered() -> None:
    """The delegate (and its own timeout) must only see time after the wait."""
    fake_time = _FakeTime()
    delegate = _RecordingDelegate(fake_time)
    client = _gate(delegate, fake_time, per_minute=60)

    async def mark_sleep(seconds: float) -> None:
        delegate.events.append("slept")
        await fake_time.sleep(seconds)

    client = RateLimitedCompletionClient(
        delegate,
        max_requests_per_minute=60,
        max_concurrent_calls=4,
        clock=fake_time.clock,
        sleep=mark_sleep,
    )
    work = asyncio.gather(_call(client), _call(client))
    await asyncio.wait_for(fake_time.advance_until(work), _TIMEOUT_SECONDS)

    assert delegate.events == ["delegate_entered", "slept", "delegate_entered"]


@pytest.mark.anyio
async def test_a_delegate_error_releases_the_slot_and_propagates() -> None:
    fake_time = _FakeTime()
    delegate = _RecordingDelegate(fake_time)
    delegate.fail_with = RuntimeError("provider down")
    client = _gate(delegate, fake_time, concurrent=1, unpaced=True)

    with pytest.raises(RuntimeError, match="provider down"):
        await asyncio.wait_for(_call(client), _TIMEOUT_SECONDS)

    delegate.fail_with = None
    assert await asyncio.wait_for(_call(client), _TIMEOUT_SECONDS) == "ok"


@pytest.mark.anyio
async def test_cancelling_a_call_inside_the_delegate_releases_the_slot() -> None:
    fake_time = _FakeTime()
    delegate = _RecordingDelegate(fake_time, gated=True)
    client = _gate(delegate, fake_time, concurrent=1, unpaced=True)

    held = asyncio.ensure_future(_call(client))
    await fake_time.settle()
    assert delegate.in_flight == 1
    held.cancel()
    with pytest.raises(asyncio.CancelledError):
        await held

    delegate.gate.set()
    assert await asyncio.wait_for(_call(client), _TIMEOUT_SECONDS) == "ok"
    assert delegate.in_flight == 0


@pytest.mark.anyio
async def test_cancelling_a_call_waiting_for_its_start_releases_the_slot() -> None:
    fake_time = _FakeTime()
    delegate = _RecordingDelegate(fake_time)
    client = _gate(delegate, fake_time, per_minute=60, concurrent=1)

    await asyncio.wait_for(_call(client), _TIMEOUT_SECONDS)  # consumes t=0
    waiting = asyncio.ensure_future(_call(client))  # sleeps until t=1
    await fake_time.settle()
    assert fake_time.sleeps == [pytest.approx(1.0)]
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting

    follow_up = asyncio.ensure_future(_call(client))
    await asyncio.wait_for(fake_time.advance_until(follow_up), _TIMEOUT_SECONDS)

    # The cancelled call forfeited its slot rather than letting the next one
    # start early: the rate is never exceeded.
    assert delegate.start_times == pytest.approx([0.0, 2.0])
