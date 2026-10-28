"""Offline tests for persistent Redis-backed session reports."""

import hashlib
from uuid import UUID, uuid4

import pytest
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
        self._acquired = False

    def acquire(self) -> bool:
        self._client.ensure_available()
        self._acquired = self._client.lock_available
        return self._acquired

    def release(self) -> None:
        self._client.ensure_available()
        if not self._acquired:
            raise AssertionError("cannot release an unowned fake lock")
        self._acquired = False


class _FakeRedisClient:
    def __init__(self) -> None:
        self.data: dict[str, str | bytes] = {}
        self.available = True
        self.lock_available = True

    def ensure_available(self) -> None:
        if not self.available:
            raise ConnectionError("fake Redis unavailable")

    def ping(self) -> bool:
        self.ensure_available()
        return True

    def set(self, name: str, value: str) -> object:
        self.ensure_available()
        self.data[name] = value
        return True

    def get(self, name: str) -> str | bytes | None:
        self.ensure_available()
        return self.data.get(name)

    def lock(
        self,
        name: str,
        *,
        timeout: float,
        blocking_timeout: float,
    ) -> _FakeLock:
        del name, timeout, blocking_timeout
        return _FakeLock(self)


def test_redis_report_store_persists_all_report_content() -> None:
    """Reconstructed stores should recover bytes, metadata, and extracted text."""
    client = _FakeRedisClient()
    first_store = RedisReportStore(client, max_reports_per_session=2)
    session_id = uuid4()
    processed = _processed_report("Revenue increased by 12%.")

    metadata = first_store.add(session_id, processed)
    reconstructed = RedisReportStore(client, max_reports_per_session=2)
    restored = reconstructed.resolve(session_id, (metadata.file_id,))[0]

    assert reconstructed.list_for_session(session_id) == (metadata,)
    assert restored.metadata == metadata
    assert restored.original_bytes == processed.original_bytes
    assert restored.extracted_text == processed.extracted_text


def test_redis_report_store_preserves_order_ownership_capacity_and_deletion() -> None:
    """Persistent report behavior should match the in-memory store contract."""
    store = RedisReportStore(_FakeRedisClient(), max_reports_per_session=2)
    owner = uuid4()
    other = uuid4()
    first = store.add(owner, _processed_report("first", filename="first.txt"))
    second = store.add(owner, _processed_report("second", filename="second.txt"))

    assert store.list_for_session(owner) == (first, second)
    assert store.list_for_session(other) == ()
    with pytest.raises(ReportNotFoundError):
        store.resolve(other, (first.file_id,))
    with pytest.raises(ReportNotFoundError):
        store.delete(other, first.file_id)
    with pytest.raises(ReportCapacityError):
        store.add(owner, _processed_report("third"))

    store.delete(owner, first.file_id)
    third = store.add(owner, _processed_report("third"))
    assert store.list_for_session(owner) == (second, third)


def test_redis_report_store_handles_collisions_busy_locks_and_outages() -> None:
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

    assert store.add(session_id, _processed_report("first")).file_id == first_id
    assert store.add(session_id, _processed_report("second")).file_id == second_id

    client.lock_available = False
    with pytest.raises(ReportStoreBusyError):
        store.delete(session_id, first_id)

    client.available = False
    with pytest.raises(ReportStoreUnavailableError):
        store.check()
    with pytest.raises(ReportStoreUnavailableError):
        store.list_for_session(session_id)


@pytest.mark.parametrize(
    "payload",
    [
        "not JSON",
        '{"version":2,"reports":[]}',
        '{"version":1,"reports":[{"metadata":null}]}',
        b"\xff",
    ],
)
def test_redis_report_store_rejects_corrupt_persistent_data(
    payload: str | bytes,
) -> None:
    """Malformed persistent data must fail closed instead of leaking partial state."""
    client = _FakeRedisClient()
    store = RedisReportStore(client, max_reports_per_session=2)
    session_id = uuid4()
    client.data[f"agentic-eval-toyapp:reports:{session_id}"] = payload

    with pytest.raises(ReportStoreUnavailableError, match="invalid report data"):
        store.list_for_session(session_id)


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
