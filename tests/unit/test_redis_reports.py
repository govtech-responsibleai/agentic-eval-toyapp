"""Offline tests for persistent Redis-backed session reports."""

import asyncio
import base64
import hashlib
import json
import threading
from collections.abc import Callable, Mapping
from typing import cast
from uuid import UUID, uuid4

import pytest
from anyio import to_thread
from redis.exceptions import ConnectionError
from tests.support.fake_redis_memory import FakeMemoryState

from agentic_eval_toyapp.models import ReportMediaType, ReportUploadResponse
from agentic_eval_toyapp.redis_memory import RedisMemoryGuard, StorageCapacityError
from agentic_eval_toyapp.reports import (
    ExtractedPage,
    ProcessedReport,
    RedisReportStore,
    ReportCapacityError,
    ReportNotFoundError,
    ReportStoreBusyError,
    ReportStoreUnavailableError,
    StoredReport,
)

_PREFIX = "agentic-eval-toyapp"


def _index_key(session_id: UUID) -> str:
    return f"{_PREFIX}:reports:{session_id}:index"


def _content_key(session_id: UUID, file_id: UUID) -> str:
    return f"{_PREFIX}:report:{session_id}:{file_id}"


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


def _pdf_report() -> ProcessedReport:
    content = b"%PDF-1.7 fake bytes that the store never parses"
    pages = (
        ExtractedPage(page_number=1, text="Page one."),
        ExtractedPage(page_number=3, text="Page three."),
    )
    return ProcessedReport(
        filename="report.pdf",
        media_type=ReportMediaType.PDF,
        byte_size=len(content),
        original_bytes=content,
        extracted_text="\n\n".join(page.text for page in pages),
        sha256=hashlib.sha256(content).hexdigest(),
        page_count=3,
        pages=pages,
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


class _FakePipeline:
    """Queue commands and apply all of them, or none, on `execute`."""

    def __init__(self, client: "_FakeRedisClient", *, transaction: bool) -> None:
        if not transaction:
            raise AssertionError("report writes must use a MULTI/EXEC pipeline")
        self._client = client
        self._commands: list[tuple[str, tuple[str, ...], int | None]] = []
        self._bookkeeping: list[tuple[str, tuple[object, ...]]] = []

    def set(self, name: str, value: str, *, ex: int | None = None) -> object:
        self._commands.append(("set", (name, value), ex))
        return self

    def expire(self, name: str, time: int) -> object:
        self._commands.append(("expire", (name,), time))
        return self

    def zadd(self, name: str, mapping: Mapping[str, float]) -> object:
        self._bookkeeping.append(("zadd", (name, dict(mapping))))
        return self

    def hset(self, name: str, key: str, value: str) -> object:
        self._bookkeeping.append(("hset", (name, key, value)))
        return self

    def hdel(self, name: str, *keys: str) -> object:
        self._bookkeeping.append(("hdel", (name, *keys)))
        return self

    def delete(self, *names: str) -> object:
        self._commands.append(("delete", names, None))
        return self

    async def execute(self) -> list[object]:
        self._client.ensure_available()
        await asyncio.sleep(0)
        if self._client.fail_next_execute:
            self._client.fail_next_execute = False
            raise ConnectionError("fake EXEC failure")
        self._client.executed += 1
        results: list[object] = []
        data, ttls = self._client.data, self._client.ttls
        for command, arguments, seconds in self._commands:
            if command == "set":
                name, value = arguments
                data[name] = value
                # Like Redis, a SET without EX clears any existing TTL.
                if seconds is None:
                    ttls.pop(name, None)
                else:
                    ttls[name] = seconds
                results.append(True)
            elif command == "expire":
                (name,) = arguments
                assert seconds is not None
                if name in data:
                    ttls[name] = seconds
                results.append(name in data)
            else:
                removed = [name for name in arguments if name in data]
                for name in removed:
                    del data[name]
                    ttls.pop(name, None)
                results.append(len(removed))
        for command, arguments in self._bookkeeping:
            results.append(self._client.memory.apply(command, arguments))
        return results


class _FakeRedisClient:
    def __init__(self) -> None:
        self.data: dict[str, str | bytes] = {}
        self.ttls: dict[str, int] = {}
        self.memory = FakeMemoryState()
        self.reads: list[str] = []
        self.available = True
        self.lock_available = True
        self.fail_next_execute = False
        self.executed = 0
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
        self.reads.append(name)
        return self.data.get(name)

    def pipeline(self, transaction: bool = True) -> _FakePipeline:
        return _FakePipeline(self, transaction=transaction)

    async def info(self, section: str) -> Mapping[str, object]:
        self.ensure_available()
        return await self.memory.info(section)

    async def eval(self, script: str, numkeys: int, *keys_and_args: str) -> object:
        self.ensure_available()
        return await self.memory.eval(script, numkeys, *keys_and_args)

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
    with pytest.raises(ValueError, match="report_load_concurrency"):
        RedisReportStore(client, max_reports_per_session=1, report_load_concurrency=0)
    with pytest.raises(ValueError, match="idle_ttl_seconds"):
        RedisReportStore(client, max_reports_per_session=1, idle_ttl_seconds=0)


def _stored(
    session_id: UUID, processed: ProcessedReport, file_id: UUID
) -> StoredReport:
    del session_id
    return StoredReport(
        metadata=ReportUploadResponse(
            file_id=file_id,
            filename=processed.filename,
            media_type=processed.media_type,
            byte_size=processed.byte_size,
            extracted_text_length=len(processed.extracted_text),
            sha256=processed.sha256,
            page_count=processed.page_count,
        ),
        original_bytes=processed.original_bytes,
        extracted_text=processed.extracted_text,
        pages=processed.pages,
    )


def _load_json(client: _FakeRedisClient, key: str) -> dict[str, object]:
    raw = client.data[key]
    assert isinstance(raw, str)
    return cast(dict[str, object], json.loads(raw))


@pytest.mark.anyio
async def test_add_writes_a_metadata_index_and_one_content_key_per_report() -> None:
    """Each report gets its own content key; the index holds metadata only."""
    client = _FakeRedisClient()
    store = RedisReportStore(client, max_reports_per_session=3)
    session_id = uuid4()
    first = await store.add(session_id, _processed_report("first"))
    second = await store.add(session_id, _pdf_report())

    assert set(client.data) == {
        _index_key(session_id),
        _content_key(session_id, first.file_id),
        _content_key(session_id, second.file_id),
    }
    index = _load_json(client, _index_key(session_id))
    assert index == {
        "version": 2,
        "reports": [
            first.model_dump(mode="json"),
            second.model_dump(mode="json"),
        ],
    }
    content = _load_json(client, _content_key(session_id, second.file_id))
    assert content["version"] == 2
    assert content["file_id"] == str(second.file_id)
    assert base64.b64decode(cast(str, content["original_content_base64"])) == (
        _pdf_report().original_bytes
    )
    # Every report write is one MULTI/EXEC transaction.
    assert client.executed == 2


@pytest.mark.anyio
async def test_list_reads_only_the_index_even_when_content_is_absent() -> None:
    """Listing must never load report content, so missing content is invisible."""
    client = _FakeRedisClient()
    store = RedisReportStore(client, max_reports_per_session=2)
    session_id = uuid4()
    first = await store.add(session_id, _processed_report("first"))
    second = await store.add(session_id, _processed_report("second"))
    client.data[_content_key(session_id, first.file_id)] = "corrupt"
    del client.data[_content_key(session_id, second.file_id)]
    client.reads.clear()

    assert await store.list_for_session(session_id) == (first, second)
    assert client.reads == [_index_key(session_id)]


@pytest.mark.anyio
async def test_list_for_a_session_without_reports_returns_nothing() -> None:
    client = _FakeRedisClient()
    store = RedisReportStore(client, max_reports_per_session=1)
    session_id = uuid4()

    assert await store.list_for_session(session_id) == ()
    assert client.reads == [_index_key(session_id)]


@pytest.mark.anyio
async def test_resolve_loads_only_the_requested_reports_in_requested_order() -> None:
    """A turn attaching one file must load that file's content key alone."""
    client = _FakeRedisClient()
    store = RedisReportStore(client, max_reports_per_session=3)
    session_id = uuid4()
    first = await store.add(session_id, _processed_report("first"))
    second = await store.add(session_id, _pdf_report())
    third = await store.add(session_id, _processed_report("third"))
    client.reads.clear()

    (only,) = await store.resolve(session_id, (second.file_id,))
    assert only.metadata == second
    assert only.pages == _pdf_report().pages
    assert only.extracted_text == _pdf_report().extracted_text
    assert client.reads == [
        _index_key(session_id),
        _content_key(session_id, second.file_id),
    ]

    client.reads.clear()
    resolved = await store.resolve(
        session_id, (third.file_id, first.file_id, third.file_id)
    )
    assert [report.metadata for report in resolved] == [third, first, third]
    assert client.reads == [
        _index_key(session_id),
        _content_key(session_id, third.file_id),
        _content_key(session_id, first.file_id),
    ]


@pytest.mark.anyio
async def test_resolve_with_no_ids_reads_no_content() -> None:
    client = _FakeRedisClient()
    store = RedisReportStore(client, max_reports_per_session=1)
    session_id = uuid4()
    await store.add(session_id, _processed_report("first"))
    client.reads.clear()

    assert await store.resolve(session_id, ()) == ()
    assert all(not key.startswith(f"{_PREFIX}:report:") for key in client.reads)


@pytest.mark.anyio
async def test_unknown_and_cross_session_ids_fail_before_loading_content() -> None:
    """Ownership comes from the index; no content key is read for a foreign ID."""
    client = _FakeRedisClient()
    store = RedisReportStore(client, max_reports_per_session=2)
    owner = uuid4()
    other = uuid4()
    owned = await store.add(owner, _processed_report("owned"))
    await store.add(other, _processed_report("other"))
    client.reads.clear()

    with pytest.raises(ReportNotFoundError):
        await store.resolve(other, (owned.file_id,))
    with pytest.raises(ReportNotFoundError):
        await store.resolve(owner, (owned.file_id, uuid4()))
    assert all(not key.startswith(f"{_PREFIX}:report:") for key in client.reads)


@pytest.mark.anyio
async def test_resolve_treats_a_missing_content_key_as_a_missing_report() -> None:
    client = _FakeRedisClient()
    store = RedisReportStore(client, max_reports_per_session=1)
    session_id = uuid4()
    metadata = await store.add(session_id, _processed_report("first"))
    del client.data[_content_key(session_id, metadata.file_id)]

    with pytest.raises(ReportNotFoundError):
        await store.resolve(session_id, (metadata.file_id,))


def _set_field(name: str, value: object) -> Callable[[dict[str, object]], object]:
    def mutate(payload: dict[str, object]) -> object:
        payload[name] = value
        return payload

    return mutate


def _replace_with(value: object) -> Callable[[dict[str, object]], object]:
    def mutate(payload: dict[str, object]) -> object:
        del payload
        return value

    return mutate


def _without(name: str) -> Callable[[dict[str, object]], object]:
    def mutate(payload: dict[str, object]) -> object:
        del payload[name]
        return payload

    return mutate


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(_replace_with("not an object"), id="not-an-object"),
        pytest.param(_set_field("version", 1), id="old-version"),
        pytest.param(_set_field("file_id", str(UUID(int=7))), id="file-id-mismatch"),
        pytest.param(_set_field("file_id", "not-a-uuid"), id="file-id-invalid"),
        pytest.param(_set_field("original_content_base64", "!!!"), id="invalid-base64"),
        pytest.param(
            _set_field("original_content_base64", base64.b64encode(b"other").decode()),
            id="size-mismatch",
        ),
        pytest.param(
            _set_field(
                "original_content_base64",
                base64.b64encode(
                    b"%PDF-1.7 fake bytes that the store never parsez"
                ).decode(),
            ),
            id="hash-mismatch",
        ),
        pytest.param(_set_field("extracted_text", ""), id="empty-text"),
        pytest.param(_set_field("extracted_text", "Page one."), id="text-length"),
        pytest.param(
            _set_field("pages", [{"page_number": 1, "text": "Page one."}]),
            id="pages-disagree-with-text",
        ),
        pytest.param(
            _set_field("pages", [{"page_number": 0, "text": "x"}]), id="page-zero"
        ),
        pytest.param(
            _set_field(
                "pages",
                [
                    {"page_number": 1, "text": "Page one."},
                    {"page_number": 4, "text": "Page three."},
                ],
            ),
            id="page-beyond-count",
        ),
        pytest.param(
            _set_field("pages", [{"page_number": True, "text": "Page one."}]),
            id="page-number-bool",
        ),
        pytest.param(_set_field("pages", []), id="pdf-without-pages"),
        pytest.param(_set_field("unexpected", 1), id="extra-field"),
        pytest.param(_without("extracted_text"), id="missing-field"),
    ],
)
@pytest.mark.anyio
async def test_resolve_rejects_corrupt_report_content(
    mutate: Callable[[dict[str, object]], object],
) -> None:
    """Content that disagrees with its index entry must fail closed."""
    client = _FakeRedisClient()
    store = RedisReportStore(client, max_reports_per_session=1)
    session_id = uuid4()
    metadata = await store.add(session_id, _pdf_report())
    key = _content_key(session_id, metadata.file_id)
    client.data[key] = json.dumps(mutate(_load_json(client, key)))

    with pytest.raises(ReportStoreUnavailableError, match="invalid report data"):
        await store.resolve(session_id, (metadata.file_id,))


@pytest.mark.anyio
async def test_resolve_rejects_non_utf8_content_bytes() -> None:
    client = _FakeRedisClient()
    store = RedisReportStore(client, max_reports_per_session=1)
    session_id = uuid4()
    metadata = await store.add(session_id, _processed_report("first"))
    client.data[_content_key(session_id, metadata.file_id)] = b"\xff"

    with pytest.raises(ReportStoreUnavailableError, match="invalid report data"):
        await store.resolve(session_id, (metadata.file_id,))


@pytest.mark.anyio
async def test_a_text_report_with_stored_pages_is_rejected() -> None:
    client = _FakeRedisClient()
    store = RedisReportStore(client, max_reports_per_session=1)
    session_id = uuid4()
    metadata = await store.add(session_id, _processed_report("first"))
    key = _content_key(session_id, metadata.file_id)
    payload = _load_json(client, key)
    payload["pages"] = [{"page_number": 1, "text": "first"}]
    client.data[key] = json.dumps(payload)

    with pytest.raises(ReportStoreUnavailableError, match="invalid report data"):
        await store.resolve(session_id, (metadata.file_id,))


def _metadata_json(file_id: UUID) -> dict[str, object]:
    return _stored(uuid4(), _processed_report("x"), file_id).metadata.model_dump(
        mode="json"
    )


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param("not JSON", id="not-json"),
        pytest.param(b"\xff", id="not-utf8"),
        pytest.param("[]", id="not-an-object"),
        pytest.param('{"version":1,"reports":[]}', id="old-version"),
        pytest.param('{"version":2,"reports":{}}', id="reports-not-a-list"),
        pytest.param('{"version":2,"reports":[{"file_id":null}]}', id="bad-metadata"),
        pytest.param('{"version":2,"reports":[],"extra":1}', id="extra-field"),
        pytest.param(
            json.dumps({"version": 2, "reports": [_metadata_json(UUID(int=1))] * 2}),
            id="duplicate-file-id",
        ),
        pytest.param(
            json.dumps(
                {
                    "version": 2,
                    "reports": [_metadata_json(UUID(int=n)) for n in (1, 2, 3)],
                }
            ),
            id="over-capacity",
        ),
    ],
)
@pytest.mark.anyio
async def test_list_rejects_a_corrupt_index(payload: str | bytes) -> None:
    """A malformed index must fail closed instead of leaking partial state."""
    client = _FakeRedisClient()
    store = RedisReportStore(client, max_reports_per_session=2)
    session_id = uuid4()
    client.data[_index_key(session_id)] = payload

    with pytest.raises(ReportStoreUnavailableError, match="invalid report data"):
        await store.list_for_session(session_id)


@pytest.mark.anyio
async def test_a_version_one_collection_under_the_old_key_is_ignored() -> None:
    """Old single-value data is not read; flushing Redis is a deploy step."""
    client = _FakeRedisClient()
    store = RedisReportStore(client, max_reports_per_session=1)
    session_id = uuid4()
    client.data[f"{_PREFIX}:reports:{session_id}"] = '{"version":1,"reports":[]}'

    assert await store.list_for_session(session_id) == ()
    await store.add(session_id, _processed_report("fresh"))
    assert len(await store.list_for_session(session_id)) == 1


@pytest.mark.anyio
async def test_a_failed_add_transaction_leaves_no_keys_and_frees_the_lock() -> None:
    client = _FakeRedisClient()
    store = RedisReportStore(client, max_reports_per_session=2)
    session_id = uuid4()
    client.fail_next_execute = True

    with pytest.raises(ReportStoreUnavailableError):
        await store.add(session_id, _processed_report("first"))
    assert client.data == {}
    assert [lock.acquired for lock in client.locks] == [False]


@pytest.mark.anyio
async def test_deleting_the_last_report_removes_its_content_and_the_index() -> None:
    client = _FakeRedisClient()
    store = RedisReportStore(client, max_reports_per_session=2)
    session_id = uuid4()
    first = await store.add(session_id, _processed_report("first"))
    second = await store.add(session_id, _processed_report("second"))

    await store.delete(session_id, first.file_id)
    assert set(client.data) == {
        _index_key(session_id),
        _content_key(session_id, second.file_id),
    }
    await store.delete(session_id, second.file_id)
    assert client.data == {}
    assert await store.list_for_session(session_id) == ()


@pytest.mark.anyio
async def test_deleting_an_unknown_report_writes_nothing() -> None:
    client = _FakeRedisClient()
    store = RedisReportStore(client, max_reports_per_session=1)
    session_id = uuid4()
    await store.add(session_id, _processed_report("first"))
    before = dict(client.data)
    executed = client.executed

    with pytest.raises(ReportNotFoundError):
        await store.delete(session_id, uuid4())
    with pytest.raises(ReportNotFoundError):
        await store.delete(uuid4(), uuid4())
    assert client.data == before
    assert client.executed == executed


@pytest.mark.anyio
async def test_seed_writes_every_report_in_one_transaction() -> None:
    client = _FakeRedisClient()
    store = RedisReportStore(client, max_reports_per_session=2)
    session_id = uuid4()
    reports = (
        _stored(session_id, _processed_report("a"), UUID(int=1)),
        _stored(session_id, _pdf_report(), UUID(int=2)),
    )

    await store.seed(session_id, reports)

    assert client.executed == 1
    assert set(client.data) == {
        _index_key(session_id),
        _content_key(session_id, UUID(int=1)),
        _content_key(session_id, UUID(int=2)),
    }
    assert await store.resolve(session_id, (UUID(int=2), UUID(int=1))) == (
        reports[1],
        reports[0],
    )


@pytest.mark.anyio
async def test_seed_refuses_existing_data_and_writes_nothing_when_empty() -> None:
    client = _FakeRedisClient()
    store = RedisReportStore(client, max_reports_per_session=2)
    session_id = uuid4()

    await store.seed(session_id, ())
    assert client.data == {}
    assert client.executed == 0

    await store.add(session_id, _processed_report("a"))
    with pytest.raises(ReportStoreUnavailableError, match="already contains"):
        await store.seed(
            session_id, (_stored(session_id, _processed_report("b"), UUID(int=9)),)
        )


@pytest.mark.anyio
async def test_content_loads_are_bounded_by_report_load_concurrency() -> None:
    """At most N content loads may run at once across all sessions."""
    release = asyncio.Event()
    in_flight = 0
    peak = 0

    class _GatedClient(_FakeRedisClient):
        async def get(self, name: str) -> str | bytes | None:
            nonlocal in_flight, peak
            if not name.startswith(f"{_PREFIX}:report:"):
                return await super().get(name)
            in_flight += 1
            peak = max(peak, in_flight)
            try:
                await release.wait()
                return await super().get(name)
            finally:
                in_flight -= 1

    client = _GatedClient()
    store = RedisReportStore(
        client, max_reports_per_session=1, report_load_concurrency=2
    )
    sessions = [uuid4() for _ in range(4)]
    file_ids = [
        (await store.add(session_id, _processed_report(f"r{n}"))).file_id
        for n, session_id in enumerate(sessions)
    ]

    tasks = [
        asyncio.create_task(store.resolve(session_id, (file_id,)))
        for session_id, file_id in zip(sessions, file_ids, strict=True)
    ]
    for _ in range(20):
        await asyncio.sleep(0)
    assert in_flight == 2
    release.set()
    results = await asyncio.wait_for(asyncio.gather(*tasks), 1.0)

    assert peak == 2
    assert [result[0].metadata.file_id for result in results] == file_ids


_TTL = 3_600


@pytest.mark.anyio
async def test_add_writes_every_report_key_with_the_idle_ttl() -> None:
    client = _FakeRedisClient()
    store = RedisReportStore(client, max_reports_per_session=2, idle_ttl_seconds=_TTL)
    session_id = uuid4()
    first = await store.add(session_id, _processed_report("first"))
    client.ttls[_index_key(session_id)] = 5
    second = await store.add(session_id, _pdf_report())

    assert client.ttls == {
        _index_key(session_id): _TTL,
        _content_key(session_id, first.file_id): _TTL,
        _content_key(session_id, second.file_id): _TTL,
    }


@pytest.mark.anyio
async def test_delete_rewrites_the_index_with_the_idle_ttl() -> None:
    client = _FakeRedisClient()
    store = RedisReportStore(client, max_reports_per_session=2, idle_ttl_seconds=_TTL)
    session_id = uuid4()
    first = await store.add(session_id, _processed_report("first"))
    second = await store.add(session_id, _processed_report("second"))
    client.ttls[_index_key(session_id)] = 5

    await store.delete(session_id, first.file_id)

    assert client.ttls == {
        _index_key(session_id): _TTL,
        _content_key(session_id, second.file_id): _TTL,
    }


@pytest.mark.anyio
async def test_touch_refreshes_the_index_and_every_content_key() -> None:
    client = _FakeRedisClient()
    store = RedisReportStore(client, max_reports_per_session=3, idle_ttl_seconds=_TTL)
    session_id = uuid4()
    other = uuid4()
    reports = [await store.add(session_id, _processed_report(t)) for t in "ab"]
    foreign = await store.add(other, _processed_report("c"))
    for key in client.ttls:
        client.ttls[key] = 5
    client.reads.clear()

    await store.touch(session_id)

    assert client.ttls == {
        _index_key(session_id): _TTL,
        _content_key(session_id, reports[0].file_id): _TTL,
        _content_key(session_id, reports[1].file_id): _TTL,
        _index_key(other): 5,
        _content_key(other, foreign.file_id): 5,
    }
    # Refreshing reads the small index only, never report content.
    assert client.reads == [_index_key(session_id)]


@pytest.mark.anyio
async def test_touch_of_a_session_without_reports_writes_nothing() -> None:
    client = _FakeRedisClient()
    store = RedisReportStore(client, max_reports_per_session=1, idle_ttl_seconds=_TTL)

    await store.touch(uuid4())

    assert client.data == {}
    assert client.executed == 0


@pytest.mark.anyio
async def test_touch_ignores_a_missing_content_key_and_creates_nothing() -> None:
    client = _FakeRedisClient()
    store = RedisReportStore(client, max_reports_per_session=2, idle_ttl_seconds=_TTL)
    session_id = uuid4()
    gone = await store.add(session_id, _processed_report("gone"))
    kept = await store.add(session_id, _processed_report("kept"))
    del client.data[_content_key(session_id, gone.file_id)]
    del client.ttls[_content_key(session_id, gone.file_id)]

    await store.touch(session_id)

    assert set(client.data) == {
        _index_key(session_id),
        _content_key(session_id, kept.file_id),
    }
    assert set(client.ttls) == set(client.data)


@pytest.mark.anyio
async def test_touch_reports_an_outage_as_unavailable() -> None:
    client = _FakeRedisClient()
    store = RedisReportStore(client, max_reports_per_session=1, idle_ttl_seconds=_TTL)
    session_id = uuid4()
    await store.add(session_id, _processed_report("first"))
    client.fail_next_execute = True

    with pytest.raises(ReportStoreUnavailableError):
        await store.touch(session_id)


@pytest.mark.anyio
async def test_seeded_demo_reports_have_no_ttl() -> None:
    client = _FakeRedisClient()
    store = RedisReportStore(client, max_reports_per_session=1, idle_ttl_seconds=_TTL)
    session_id = uuid4()

    await store.seed(
        session_id, (_stored(session_id, _processed_report("demo"), UUID(int=1)),)
    )

    assert len(client.data) == 2
    assert client.ttls == {}


def _guarded_store(
    client: _FakeRedisClient, *, budget: int, max_reports: int = 3
) -> RedisReportStore:
    client.memory.maxmemory = budget
    return RedisReportStore(
        client,
        max_reports_per_session=max_reports,
        idle_ttl_seconds=_TTL,
        memory_guard=RedisMemoryGuard(high_water=1.0, clock=lambda: 7.0),
    )


@pytest.mark.anyio
async def test_guarded_add_and_delete_keep_the_report_size_current() -> None:
    client = _FakeRedisClient()
    store = _guarded_store(client, budget=1 << 30)
    session_id = uuid4()
    field = f"{session_id}:reports"

    first = await store.add(session_id, _processed_report("first"))
    one = int(client.memory.sizes[field])
    second = await store.add(session_id, _pdf_report())
    two = int(client.memory.sizes[field])
    assert two > one > len(client.data[_content_key(session_id, first.file_id)])
    assert client.memory.lru == {str(session_id): 7.0}

    await store.delete(session_id, second.file_id)
    assert int(client.memory.sizes[field]) == one
    await store.delete(session_id, first.file_id)
    assert field not in client.memory.sizes


@pytest.mark.anyio
async def test_guarded_add_refuses_a_report_larger_than_the_budget() -> None:
    client = _FakeRedisClient()
    store = _guarded_store(client, budget=600)
    session_id = uuid4()

    with pytest.raises(StorageCapacityError):
        await store.add(session_id, _processed_report("x" * 1_000))

    assert client.data == {}
    assert client.memory.eval_calls == []
    assert [lock.acquired for lock in client.locks] == [False]


@pytest.mark.anyio
async def test_guarded_add_evicts_others_but_never_its_own_session() -> None:
    client = _FakeRedisClient()
    store = _guarded_store(client, budget=1 << 20)
    session_id = uuid4()
    client.memory.used_memory = 1 << 20

    def evict(_args: tuple[str, ...]) -> object:
        client.memory.used_memory = 0
        return ["idle", "memory_budget", "9000", "1"]

    client.memory.evict = evict
    await store.add(session_id, _processed_report("first"))

    (call,) = client.memory.eval_calls
    assert call[:3] == (
        f"{_PREFIX}:sessions:lru",
        f"{_PREFIX}:sessions:bytes",
        _PREFIX,
    )
    assert call[-1] == str(session_id)


@pytest.mark.anyio
async def test_guarded_seed_does_not_track_demo_reports() -> None:
    client = _FakeRedisClient()
    store = _guarded_store(client, budget=1 << 30)
    session_id = uuid4()

    await store.seed(
        session_id, (_stored(session_id, _processed_report("demo"), UUID(int=3)),)
    )

    assert client.memory.lru == {}
    assert client.memory.sizes == {}
