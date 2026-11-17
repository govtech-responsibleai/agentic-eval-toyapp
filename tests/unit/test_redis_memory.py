"""Offline tests for the Redis memory budget and eviction driver.

The Lua eviction script runs only against a real Redis; see
`tests/integration/test_redis_memory_live.py`. Here the fake `eval` records
its arguments and returns scripted results, so these tests cover the budget
maths, the eviction rounds, the refusal paths, and the log fields.
"""

import logging
from collections.abc import Callable, Mapping
from uuid import UUID

import pytest
from redis.exceptions import ConnectionError

from agentic_eval_toyapp.redis_memory import (
    KEY_OVERHEAD_BYTES,
    REPORT_FRAMING_BYTES,
    RedisMemoryGuard,
    StorageBudgetUnavailableError,
    StorageCapacityError,
    StorageUnavailableError,
    compute_budget,
    estimate_report_bytes,
)

_SESSION = UUID("00000000-0000-0000-0000-0000000000a1")


class _FakeMemoryClient:
    """Serve scripted `INFO memory` replies and record eviction calls."""

    def __init__(
        self,
        *,
        used: int = 0,
        maxmemory: int | str | None = 1_000,
        evict: Callable[[int], object] | None = None,
    ) -> None:
        self.used = used
        self.maxmemory = maxmemory
        self.info_calls = 0
        self.eval_calls: list[tuple[str, ...]] = []
        self.fail_info = False
        self.fail_eval = False
        self.evict: Callable[[int], object] = evict or (lambda _needed: [])

    async def info(self, section: str) -> Mapping[str, object]:
        assert section == "memory"
        self.info_calls += 1
        if self.fail_info:
            raise ConnectionError("fake INFO failure")
        info: dict[str, object] = {"used_memory": self.used}
        if self.maxmemory is not None:
            info["maxmemory"] = self.maxmemory
        return info

    async def eval(self, script: str, numkeys: int, *keys_and_args: str) -> object:
        assert numkeys == 2
        assert "ZRANGE" in script
        if self.fail_eval:
            raise ConnectionError("fake EVAL failure")
        self.eval_calls.append(keys_and_args)
        return self.evict(int(keys_and_args[3]))


class _RecordingPipeline:
    def __init__(self) -> None:
        self.commands: list[tuple[object, ...]] = []

    def zadd(self, name: str, mapping: Mapping[str, float]) -> object:
        self.commands.append(("zadd", name, dict(mapping)))
        return self

    def hset(self, name: str, key: str, value: str) -> object:
        self.commands.append(("hset", name, key, value))
        return self

    def hdel(self, name: str, *keys: str) -> object:
        self.commands.append(("hdel", name, *keys))
        return self

    async def execute(self) -> list[object]:
        return []


# --- budget derivation ---------------------------------------------------------


@pytest.mark.parametrize(
    ("maxmemory", "fallback", "expected"),
    [
        pytest.param(1_000, None, (750, "maxmemory"), id="maxmemory-only"),
        pytest.param(None, 500, (500, "fallback"), id="unreadable-uses-fallback"),
        pytest.param(0, 500, (500, "fallback"), id="unlimited-uses-fallback"),
        pytest.param(1_000, 600, (600, "fallback"), id="fallback-smaller"),
        pytest.param(1_000, 900, (750, "maxmemory"), id="maxmemory-smaller"),
        pytest.param(1, None, (1, "maxmemory"), id="never-rounds-to-zero"),
    ],
)
def test_budget_uses_the_smaller_available_limit(
    maxmemory: int | None, fallback: int | None, expected: tuple[int, str]
) -> None:
    assert (
        compute_budget(maxmemory=maxmemory, high_water=0.75, fallback_bytes=fallback)
        == expected
    )


@pytest.mark.parametrize("maxmemory", [0, None])
def test_budget_without_any_limit_is_a_misconfiguration(
    maxmemory: int | None,
) -> None:
    with pytest.raises(StorageBudgetUnavailableError, match="no fallback"):
        compute_budget(maxmemory=maxmemory, high_water=0.75, fallback_bytes=None)


@pytest.mark.anyio
async def test_usage_reads_info_memory_including_string_values() -> None:
    client = _FakeMemoryClient(used=120, maxmemory="2000")
    usage = await RedisMemoryGuard(high_water=0.5).usage(client)

    assert (usage.used_bytes, usage.budget_bytes, usage.source) == (
        120,
        1_000,
        "maxmemory",
    )


@pytest.mark.anyio
async def test_usage_with_unlimited_maxmemory_and_no_fallback_fails() -> None:
    with pytest.raises(StorageBudgetUnavailableError):
        await RedisMemoryGuard().usage(_FakeMemoryClient(maxmemory=0))
    with pytest.raises(StorageBudgetUnavailableError):
        await RedisMemoryGuard().usage(_FakeMemoryClient(maxmemory="not-a-number"))


@pytest.mark.anyio
async def test_usage_failures_are_reported_as_unavailable() -> None:
    client = _FakeMemoryClient()
    client.fail_info = True
    with pytest.raises(StorageUnavailableError, match="could not be read"):
        await RedisMemoryGuard().usage(client)

    class _NoUsedMemory(_FakeMemoryClient):
        async def info(self, section: str) -> Mapping[str, object]:
            del section
            return {"maxmemory": 1_000, "used_memory": True}

    with pytest.raises(StorageUnavailableError, match="used_memory"):
        await RedisMemoryGuard().usage(_NoUsedMemory())


# --- ensure_room -----------------------------------------------------------------


@pytest.mark.anyio
async def test_a_write_that_fits_evicts_nothing() -> None:
    client = _FakeMemoryClient(used=700, maxmemory=1_000)

    await RedisMemoryGuard().ensure_room(client, incoming_bytes=50)

    assert client.eval_calls == []


@pytest.mark.anyio
async def test_eviction_asks_for_exactly_the_overflow_and_exempts_the_caller() -> None:
    client = _FakeMemoryClient(used=740, maxmemory=1_000)

    def evict(needed: int) -> list[object]:
        client.used -= needed
        return [str(UUID(int=9)), "memory_budget", str(needed), "2"]

    client.evict = evict
    guard = RedisMemoryGuard(key_prefix="toy", scan_limit=10)

    await guard.ensure_room(client, incoming_bytes=30, current_session=_SESSION)

    assert client.eval_calls == [
        ("toy:sessions:lru", "toy:sessions:bytes", "toy", "20", "10", str(_SESSION))
    ]


@pytest.mark.anyio
async def test_create_has_no_exempt_session() -> None:
    client = _FakeMemoryClient(used=750, maxmemory=1_000)

    def evict(needed: int) -> list[object]:
        client.used -= needed
        return ["x", "memory_budget", str(needed), "0"]

    client.evict = evict
    await RedisMemoryGuard().ensure_room(client, incoming_bytes=1)

    assert client.eval_calls[0][-1] == ""


@pytest.mark.anyio
async def test_a_single_write_larger_than_the_budget_evicts_nothing() -> None:
    client = _FakeMemoryClient(used=0, maxmemory=1_000)

    with pytest.raises(StorageCapacityError):
        await RedisMemoryGuard().ensure_room(client, incoming_bytes=751)

    assert client.eval_calls == []


@pytest.mark.anyio
async def test_nothing_evictable_refuses_with_capacity() -> None:
    client = _FakeMemoryClient(used=750, maxmemory=1_000)

    with pytest.raises(StorageCapacityError):
        await RedisMemoryGuard().ensure_room(client, incoming_bytes=10)

    assert len(client.eval_calls) == 1


@pytest.mark.anyio
async def test_eviction_stops_after_the_round_limit() -> None:
    """Estimates may undercount; a bounded number of passes then gives up."""
    client = _FakeMemoryClient(used=750, maxmemory=1_000)
    client.evict = lambda _needed: ["x", "memory_budget", "1", "0"]

    with pytest.raises(StorageCapacityError):
        await RedisMemoryGuard(max_rounds=3).ensure_room(client, incoming_bytes=10)

    assert len(client.eval_calls) == 3


@pytest.mark.anyio
async def test_a_second_round_runs_when_the_first_frees_too_little() -> None:
    client = _FakeMemoryClient(used=800, maxmemory=1_000)
    freed = iter((30, 100))

    def evict(_needed: int) -> list[object]:
        amount = next(freed)
        client.used -= amount
        return ["x", "memory_budget", str(amount), "0"]

    client.evict = evict
    await RedisMemoryGuard().ensure_room(client, incoming_bytes=10)

    assert len(client.eval_calls) == 2


@pytest.mark.anyio
async def test_an_eviction_outage_is_unavailable_not_capacity() -> None:
    client = _FakeMemoryClient(used=750, maxmemory=1_000)
    client.fail_eval = True

    with pytest.raises(StorageUnavailableError, match="eviction failed"):
        await RedisMemoryGuard().ensure_room(client, incoming_bytes=10)


@pytest.mark.anyio
async def test_each_removed_session_is_logged_without_content(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = _FakeMemoryClient(used=760, maxmemory=1_000)

    def evict(_needed: int) -> list[object]:
        client.used = 0
        return [
            b"session-a",
            b"memory_budget",
            b"4096",
            b"3",
            "session-b",
            "expired",
            "0",
            "1",
            "trailing-partial-record",
        ]

    client.evict = evict
    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        await RedisMemoryGuard().ensure_room(client, incoming_bytes=10)

    storage = [r.getMessage() for r in caplog.records if "toyapp_storage" in r.message]
    assert storage == [
        "toyapp_storage event=evict session_id=session-a reason=memory_budget "
        "bytes_freed=4096 reports_removed=3",
        "toyapp_storage event=expire session_id=session-b reason=expired "
        "bytes_freed=0 reports_removed=1",
    ]


@pytest.mark.anyio
async def test_an_unexpected_script_reply_counts_as_nothing_evicted() -> None:
    client = _FakeMemoryClient(used=750, maxmemory=1_000)
    client.evict = lambda _needed: None

    with pytest.raises(StorageCapacityError):
        await RedisMemoryGuard().ensure_room(client, incoming_bytes=10)


@pytest.mark.anyio
async def test_refusal_is_logged_with_sizes_only(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with (
        caplog.at_level(logging.WARNING, logger="uvicorn.error"),
        pytest.raises(StorageCapacityError),
    ):
        await RedisMemoryGuard().ensure_room(
            _FakeMemoryClient(used=0, maxmemory=1_000), incoming_bytes=2_000
        )

    assert (
        "toyapp_storage event=capacity_refused reason=larger_than_budget "
        "used_bytes=0 budget_bytes=750 incoming_bytes=2000"
    ) in caplog.text


# --- bookkeeping -----------------------------------------------------------------


def test_track_queues_access_time_and_only_the_sizes_given() -> None:
    guard = RedisMemoryGuard(key_prefix="toy", clock=lambda: 42.0)
    pipeline = _RecordingPipeline()

    guard.track(pipeline, _SESSION)
    guard.track(pipeline, _SESSION, session_bytes=10, reports_bytes=20)
    guard.forget_reports(pipeline, _SESSION)

    member = str(_SESSION)
    assert pipeline.commands == [
        ("zadd", "toy:sessions:lru", {member: 42.0}),
        ("zadd", "toy:sessions:lru", {member: 42.0}),
        ("hset", "toy:sessions:bytes", f"{member}:session", "10"),
        ("hset", "toy:sessions:bytes", f"{member}:reports", "20"),
        ("hdel", "toy:sessions:bytes", f"{member}:reports"),
    ]


def test_report_size_estimate_counts_base64_text_and_pdf_pages() -> None:
    fixed = REPORT_FRAMING_BYTES + KEY_OVERHEAD_BYTES

    assert estimate_report_bytes(
        byte_size=3, extracted_text_length=10, has_pages=False
    ) == (4 + 10 + fixed)
    assert estimate_report_bytes(
        byte_size=4, extracted_text_length=10, has_pages=True
    ) == (8 + 20 + fixed)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"key_prefix": " : "}, "key_prefix"),
        ({"high_water": 0.0}, "high_water"),
        ({"high_water": 1.01}, "high_water"),
        ({"budget_bytes": 0}, "budget_bytes"),
        ({"scan_limit": 0}, "scan_limit"),
        ({"max_rounds": 0}, "max_rounds"),
    ],
)
def test_guard_rejects_unusable_configuration(
    kwargs: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        RedisMemoryGuard(**kwargs)  # pyright: ignore[reportArgumentType]
