"""Turn admission and model-call limits that keep overload explicit.

Overload is refused before a turn starts (``TurnAdmission``); calls inside an
admitted turn wait for a slot and are never rejected
(``RateLimitedCompletionClient``).
"""

import time
from collections.abc import Awaitable, Callable
from typing import Any

import anyio

from agentic_eval_toyapp.providers import (
    StructuredCompletion,
    StructuredCompletionClient,
)

RETRY_AFTER_SECONDS = 10

Clock = Callable[[], float]
Sleeper = Callable[[float], Awaitable[None]]


class ServerBusyError(RuntimeError):
    """Raised when the admission cap is reached and no turn may start."""


class TurnAdmission:
    """Count admitted turns against a fixed cap.

    ``try_acquire`` has no ``await`` between its check and its increment, so on
    the event loop thread it is atomic. Callers must pair every successful
    acquire with one ``release`` in a ``finally`` block.
    """

    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        self._limit = limit
        self._pending = 0

    @property
    def limit(self) -> int:
        """Return the maximum number of simultaneously admitted turns."""
        return self._limit

    @property
    def pending(self) -> int:
        """Return the number of turns currently admitted."""
        return self._pending

    def try_acquire(self) -> bool:
        """Admit one turn if the cap allows it."""
        if self._pending >= self._limit:
            return False
        self._pending += 1
        return True

    def release(self) -> None:
        """Return one admitted turn's slot."""
        if self._pending < 1:
            raise RuntimeError("release called without a matching acquire")
        self._pending -= 1


class RateLimitedCompletionClient:
    """Bound and pace calls made through one shared completion client.

    A semaphore caps simultaneous calls. Each call then reserves the next
    evenly spaced start time, so starts never exceed the configured rate and
    are not released in bursts. The semaphore is taken before the start time is
    reserved: reversing the order would let many already-paced calls pile up on
    the semaphore and start together when slots free. All waiting happens here,
    outside the delegate, so it never counts against the delegate's own
    per-call timeout.

    A call cancelled while waiting for its start time forfeits that slot; the
    next start stays at least one interval after the previous reservation, which
    errs on the side of staying under the rate.
    """

    def __init__(
        self,
        delegate: StructuredCompletionClient,
        *,
        max_requests_per_minute: int,
        max_concurrent_calls: int,
        clock: Clock = time.monotonic,
        sleep: Sleeper = anyio.sleep,
    ) -> None:
        if max_requests_per_minute < 1:
            raise ValueError("max_requests_per_minute must be at least 1")
        if max_concurrent_calls < 1:
            raise ValueError("max_concurrent_calls must be at least 1")
        self._delegate = delegate
        self._interval = 60.0 / max_requests_per_minute
        self._clock = clock
        self._sleep = sleep
        # anyio's semaphore binds to an event loop on first use, so building it
        # here does not require a running loop.
        self._semaphore = anyio.Semaphore(max_concurrent_calls)
        self._next_start: float | None = None

    async def create(
        self,
        *,
        model: str,
        messages: tuple[dict[str, str], ...],
        response_schema: dict[str, Any],
        temperature: float,
        max_tokens: int,
    ) -> str | StructuredCompletion:
        """Wait for a concurrency slot and start time, then call the delegate."""
        async with self._semaphore:
            delay = self._reserve_start()
            if delay > 0:
                await self._sleep(delay)
            return await self._delegate.create(
                model=model,
                messages=messages,
                response_schema=response_schema,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    def _reserve_start(self) -> float:
        """Reserve the next start time and return how long to wait for it."""
        now = self._clock()
        start = now if self._next_start is None else max(now, self._next_start)
        self._next_start = start + self._interval
        return start - now
