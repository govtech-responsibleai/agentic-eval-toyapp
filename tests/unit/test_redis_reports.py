"""Offline tests for persistent Redis-backed session reports."""

import asyncio
import hashlib
import threading
from uuid import UUID, uuid4

import pytest
from anyio import to_thread
from redis.exceptions import ConnectionError

from agentic_eval_toyapp.models import ReportMediaType
from agentic_eval_toyapp.reports import (
    ProcessedReport,
    RedisReportStore,
    ReportCapacityError,
    ReportNotFoundError,
    ReportStoreBusyError,
    ReportStoreUnavailableError,
)


def _processed_report(text: str, *, filename: str = "report.txt") -> ProcessedReport:
    content = text.encode()
    return ProcessedReport(
        filename=filename,
        media_type=ReportMediaType.TEXT,
        byte_size=len(content),
        original_bytes=content,
        extracted_text=text,
        sha256=hashlib.sha256(content).hexdigest(),
    )


class _FakeLock:
    def __init__(self, client: "_FakeRedisClient") -> None:
        self._client = client
        self.acquired = False

    async def acquire(self) -> bool:
        self._client.ensure_available()
        self.acquired = self._client.lock_available
        return self.acquired

    async def release(self) -> None:
        self._client.ensure_available()
        if not self.acquired:
            raise AssertionError("cannot release an unowned fake lock")
        self.acquired = False


class _FakeRedisClient:
    def __init__(self) -> None:
        self.data: dict[str, str | bytes] = {}
        self.available = True
        self.lock_available = True
        self.locks: list[_FakeLock] = []
        self.closed = False

    def ensure_available(self) -> None:
        if not self.available:
            raise ConnectionError("fake Redis unavailable")

    async def ping(self) -> bool:
        self.ensure_available()
        return True

    async def set(self, name: str, value: str) -> object:
        self.ensure_available()
        await asyncio.sleep(0)
        self.data[name] = value
        return True

    async def get(self, name: str) -> str | bytes | None:
        self.ensure_available()
        await asyncio.sleep(0)
        return self.data.get(name)

    def lock(
        self,
        name: str,
        *,
        timeout: float,
        blocking_timeout: float,
    ) -> _FakeLock:
        del name, timeout, blocking_timeout
        lock = self._new_lock()
        self.locks.append(lock)
        return lock

    def _new_lock(self) -> _FakeLock:
        return _FakeLock(self)

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.anyio
async def test_redis_report_store_persists_all_report_content() -> None:
    """Reconstructed stores should recover bytes, metadata, and extracted text."""
    client = _FakeRedisClient()
    first_store = RedisReportStore(client, max_reports_per_session=2)
    session_id = uuid4()
    processed = _processed_report("Revenue increased by 12%.")

    metadata = await first_store.add(session_id, processed)
    reconstructed = RedisReportStore(client, max_reports_per_session=2)
    restored = (await reconstructed.resolve(session_id, (metadata.file_id,)))[0]

    assert await reconstructed.list_for_session(session_id) == (metadata,)
    assert restored.metadata == metadata
    assert restored.original_bytes == processed.original_bytes
    assert restored.extracted_text == processed.extracted_text


@pytest.mark.anyio
async def test_redis_report_store_preserves_order_ownership_capacity_and_deletion() -> (
    None
):
    """Persistent report behavior should match the in-memory store contract."""
    store = RedisReportStore(_FakeRedisClient(), max_reports_per_session=2)
    owner = uuid4()
    other = uuid4()
    first = await store.add(owner, _processed_report("first", filename="first.txt"))
    second = await store.add(owner, _processed_report("second", filename="second.txt"))

    assert await store.list_for_session(owner) == (first, second)
    assert await store.list_for_session(other) == ()
    with pytest.raises(ReportNotFoundError):
        await store.resolve(other, (first.file_id,))
    with pytest.raises(ReportNotFoundError):
        await store.delete(other, first.file_id)
    with pytest.raises(ReportCapacityError):
        await store.add(owner, _processed_report("third"))

    await store.delete(owner, first.file_id)
    third = await store.add(owner, _processed_report("third"))
    assert await store.list_for_session(owner) == (second, third)


@pytest.mark.anyio
async def test_redis_report_store_handles_collisions_busy_locks_and_outages() -> None:
    """Collision, concurrency, and connectivity failures should remain distinct."""
    first_id = UUID("00000000-0000-0000-0000-000000000001")
    second_id = UUID("00000000-0000-0000-0000-000000000002")
    generated_ids = iter((first_id, first_id, second_id))
    client = _FakeRedisClient()
    store = RedisReportStore(
        client,
        max_reports_per_session=2,
        id_factory=lambda: next(generated_ids),
    )
    session_id = uuid4()

    assert (await store.add(session_id, _processed_report("first"))).file_id == first_id
    assert (
        await store.add(session_id, _processed_report("second"))
    ).file_id == second_id

    client.lock_available = False
    with pytest.raises(ReportStoreBusyError):
        await store.delete(session_id, first_id)

    client.available = False
    with pytest.raises(ReportStoreUnavailableError):
        await store.check()
    with pytest.raises(ReportStoreUnavailableError):
        await store.list_for_session(session_id)


@pytest.mark.anyio
async def test_redis_report_store_releases_the_lock_on_error_and_cancellation() -> None:
    """A mutation that raises or is cancelled must still free the lock."""
    client = _FakeRedisClient()
    store = RedisReportStore(client, max_reports_per_session=1)
    session_id = uuid4()
    await store.add(session_id, _processed_report("first"))

    with pytest.raises(ReportCapacityError):
        await store.add(session_id, _processed_report("second"))
    assert [lock.acquired for lock in client.locks] == [False, False]

    gate = asyncio.Event()
    entered = asyncio.Event()

    async def held_mutation() -> None:
        async with store._session_lock(session_id):  # pyright: ignore[reportPrivateUsage]
            entered.set()
            await gate.wait()

    task = asyncio.create_task(held_mutation())
    await asyncio.wait_for(entered.wait(), 1.0)
    assert client.locks[-1].acquired is True
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert client.locks[-1].acquired is False


@pytest.mark.anyio
async def test_redis_report_store_waiter_holds_no_thread_with_the_pool_saturated() -> (
    None
):
    """A mutation waiting on a busy lock must not need a worker thread to wait."""
    proceed = asyncio.Event()
    started = asyncio.Event()

    class _BlockingLock(_FakeLock):
        async def acquire(self) -> bool:
            started.set()
            await proceed.wait()
            return await super().acquire()

    client = _FakeRedisClient()
    client._new_lock = lambda: _BlockingLock(client)  # pyright: ignore[reportPrivateUsage]
    store = RedisReportStore(client, max_reports_per_session=2)
    limiter = to_thread.current_default_thread_limiter()
    original = limiter.total_tokens
    limiter.total_tokens = 1
    try:
        gate = threading.Event()
        blocker = asyncio.create_task(to_thread.run_sync(gate.wait, 5.0))
        await asyncio.sleep(0.01)
        assert limiter.borrowed_tokens == 1

        waiter = asyncio.create_task(store.add(uuid4(), _processed_report("x")))
        await asyncio.wait_for(started.wait(), 1.0)
        # The waiter is parked on the lock, not in the saturated pool.
        assert limiter.borrowed_tokens == 1
        proceed.set()
        gate.set()
        await blocker
        await asyncio.wait_for(waiter, 1.0)
    finally:
        limiter.total_tokens = original


@pytest.mark.anyio
async def test_redis_report_store_client_lifecycle_is_lazy_and_idempotent() -> None:
    """from_url builds no client until open(); aclose closes only an owned client."""
    clients: list[_FakeRedisClient] = []

    def factory() -> _FakeRedisClient:
        clients.append(_FakeRedisClient())
        return clients[-1]

    store = RedisReportStore(client_factory=factory, max_reports_per_session=1)
    assert clients == []
    with pytest.raises(ReportStoreUnavailableError, match="not open"):
        await store.check()

    await store.open()
    await store.open()
    assert len(clients) == 1
    await store.check()
    await store.aclose()
    await store.aclose()
    assert clients[0].closed is True
    with pytest.raises(ReportStoreUnavailableError, match="not open"):
        await store.list_for_session(uuid4())

    injected = _FakeRedisClient()
    owned_elsewhere = RedisReportStore(injected, max_reports_per_session=1)
    await owned_elsewhere.open()
    await owned_elsewhere.aclose()
    assert injected.closed is False


@pytest.mark.parametrize(
    "payload",
    [
        "not JSON",
        '{"version":2,"reports":[]}',
        '{"version":1,"reports":[{"metadata":null}]}',
        b"\xff",
    ],
)
@pytest.mark.anyio
async def test_redis_report_store_rejects_corrupt_persistent_data(
    payload: str | bytes,
) -> None:
    """Malformed persistent data must fail closed instead of leaking partial state."""
    client = _FakeRedisClient()
    store = RedisReportStore(client, max_reports_per_session=2)
    session_id = uuid4()
    client.data[f"agentic-eval-toyapp:reports:{session_id}"] = payload

    with pytest.raises(ReportStoreUnavailableError, match="invalid report data"):
        await store.list_for_session(session_id)


def test_redis_report_store_validates_configuration() -> None:
    """Persistent report storage should reject unusable configuration."""
    client = _FakeRedisClient()
    with pytest.raises(ValueError, match="max_reports_per_session"):
        RedisReportStore(client, max_reports_per_session=0)
    with pytest.raises(ValueError, match="key_prefix"):
        RedisReportStore(client, max_reports_per_session=1, key_prefix=":")
    with pytest.raises(ValueError, match="lock_timeout_seconds"):
        RedisReportStore(
            client,
            max_reports_per_session=1,
            lock_timeout_seconds=0,
        )
    with pytest.raises(ValueError, match="exactly one"):
        RedisReportStore(max_reports_per_session=1)
