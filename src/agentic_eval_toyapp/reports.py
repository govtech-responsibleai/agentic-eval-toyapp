"""Bounded report extraction and memory or Redis-backed storage."""

import base64
import binascii
import hashlib
from collections import defaultdict
from collections.abc import AsyncGenerator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import PurePath
from threading import Lock
from typing import Literal, Protocol, cast
from uuid import UUID, uuid4

import anyio
from anyio import to_thread
from pydantic import BaseModel, ConfigDict, Field
from pypdf import PdfReader
from redis.asyncio import Redis
from redis.exceptions import LockError, RedisError

from agentic_eval_toyapp.models import ReportMediaType, ReportUploadResponse


class UnsupportedReportTypeError(ValueError):
    """Raised when filename, media type, or file signature is unsupported."""


class ReportTooLargeError(ValueError):
    """Raised when a report exceeds an accepted resource bound."""


class ReportProcessingError(ValueError):
    """Raised when a supported report cannot yield safe usable text."""


class ReportNotFoundError(LookupError):
    """Raised for both missing and cross-session report references."""


class ReportCapacityError(RuntimeError):
    """Raised when a session has reached its retained-report bound."""


class ReportAttachmentLimitError(ValueError):
    """Raised when one message references too many reports."""


class ReportStoreUnavailableError(RuntimeError):
    """Raised when the configured persistent report store is unavailable."""


class ReportStoreBusyError(RuntimeError):
    """Raised when another worker holds a report mutation lock too long."""


@dataclass(frozen=True, slots=True)
class ExtractedPage:
    """Text extracted from one one-indexed PDF page."""

    page_number: int
    text: str


@dataclass(frozen=True, slots=True)
class ProcessedReport:
    """Validated report content before a session-scoped ID is allocated."""

    filename: str
    media_type: ReportMediaType
    byte_size: int
    original_bytes: bytes = field(repr=False)
    extracted_text: str
    sha256: str
    page_count: int | None = None
    pages: tuple[ExtractedPage, ...] = ()


@dataclass(frozen=True, slots=True)
class StoredReport:
    """Immutable original and extracted content retained for one session."""

    metadata: ReportUploadResponse
    original_bytes: bytes = field(repr=False)
    extracted_text: str
    pages: tuple[ExtractedPage, ...] = ()


class ReportStore(Protocol):
    """Storage boundary required by session-scoped report routes."""

    @property
    def max_reports_per_session(self) -> int:
        """Return the maximum retained reports for one session."""
        ...

    @property
    def readiness_detail(self) -> str:
        """Return a non-sensitive description of the configured backend."""
        ...

    async def open(self) -> None:
        """Acquire loop-bound resources; call from the running event loop."""
        ...

    async def aclose(self) -> None:
        """Release resources acquired by `open`."""
        ...

    async def check(self) -> None:
        """Raise when the backing store cannot serve requests."""
        ...

    async def add(
        self, session_id: UUID, report: ProcessedReport
    ) -> ReportUploadResponse:
        """Persist one processed report under its owning session."""
        ...

    async def list_for_session(
        self, session_id: UUID
    ) -> tuple[ReportUploadResponse, ...]:
        """Return safe metadata for a session's reports in upload order."""
        ...

    async def delete(self, session_id: UUID, file_id: UUID) -> None:
        """Delete one report owned by the supplied session."""
        ...

    async def resolve(
        self,
        session_id: UUID,
        file_ids: Sequence[UUID],
    ) -> tuple[StoredReport, ...]:
        """Resolve selected reports only for their owning session."""
        ...

    async def touch(self, session_id: UUID) -> None:
        """Renew the idle expiry of every report the session owns."""
        ...


class ReportProcessor:
    """Validate untrusted report bytes and deterministically extract text."""

    _MEDIA_TYPES_BY_SUFFIX: dict[str, frozenset[str]] = {
        ".txt": frozenset({ReportMediaType.TEXT.value}),
        ".md": frozenset(
            {
                ReportMediaType.MARKDOWN.value,
                "text/x-markdown",
                ReportMediaType.TEXT.value,
            }
        ),
        ".pdf": frozenset({ReportMediaType.PDF.value}),
    }

    def __init__(
        self,
        *,
        max_bytes: int,
        max_pdf_pages: int,
        max_extracted_characters: int,
    ) -> None:
        if min(max_bytes, max_pdf_pages, max_extracted_characters) < 1:
            raise ValueError("report processing limits must be positive")
        self._max_bytes = max_bytes
        self._max_pdf_pages = max_pdf_pages
        self._max_extracted_characters = max_extracted_characters

    @property
    def max_bytes(self) -> int:
        """Return the maximum accepted raw upload size."""
        return self._max_bytes

    def process(
        self,
        *,
        filename: str,
        media_type: str,
        content: bytes,
    ) -> ProcessedReport:
        """Return bounded original bytes, metadata, and extracted content."""
        safe_filename = self._safe_filename(filename)
        normalised_media_type = media_type.partition(";")[0].strip().lower()
        suffix = PurePath(safe_filename).suffix.lower()
        accepted_media_types = self._MEDIA_TYPES_BY_SUFFIX.get(suffix)
        if (
            accepted_media_types is None
            or normalised_media_type not in accepted_media_types
        ):
            raise UnsupportedReportTypeError(
                "filename extension and media type are not a supported pair"
            )
        if not content:
            raise ReportProcessingError("report is empty")
        if len(content) > self._max_bytes:
            raise ReportTooLargeError("report exceeds the byte limit")

        digest = hashlib.sha256(content).hexdigest()
        if suffix == ".pdf":
            if not content.startswith(b"%PDF-"):
                raise UnsupportedReportTypeError("PDF signature is missing")
            text, page_count, pages = self._extract_pdf(content)
            selected_media_type = ReportMediaType.PDF
        else:
            text = self._extract_utf8(content)
            page_count = None
            pages = ()
            selected_media_type = (
                ReportMediaType.MARKDOWN if suffix == ".md" else ReportMediaType.TEXT
            )

        return ProcessedReport(
            filename=safe_filename,
            media_type=selected_media_type,
            byte_size=len(content),
            original_bytes=bytes(content),
            extracted_text=text,
            sha256=digest,
            page_count=page_count,
            pages=pages,
        )

    @staticmethod
    def _safe_filename(filename: str) -> str:
        """Return a display-only basename without accepting control characters."""
        candidate = filename.replace("\\", "/").rsplit("/", maxsplit=1)[-1].strip()
        if not candidate or len(candidate) > 255:
            raise UnsupportedReportTypeError("filename is missing or too long")
        if any(ord(character) < 32 for character in candidate):
            raise UnsupportedReportTypeError("filename contains control characters")
        return candidate

    def _extract_utf8(self, content: bytes) -> str:
        try:
            decoded = content.decode("utf-8-sig", errors="strict")
        except UnicodeDecodeError as error:
            raise ReportProcessingError("text report is not valid UTF-8") from error
        return self._normalise_and_bound(decoded)

    def _extract_pdf(
        self,
        content: bytes,
    ) -> tuple[str, int, tuple[ExtractedPage, ...]]:
        try:
            reader = PdfReader(BytesIO(content), strict=True)
            if reader.is_encrypted:
                raise ReportProcessingError("encrypted PDFs are not supported")
            page_count = len(reader.pages)
            if page_count < 1:
                raise ReportProcessingError("PDF contains no pages")
            if page_count > self._max_pdf_pages:
                raise ReportTooLargeError("PDF exceeds the page limit")

            pages: list[ExtractedPage] = []
            total_characters = 0
            for page_number, page in enumerate(reader.pages, start=1):
                page_text = self._normalise_text(page.extract_text() or "")
                if not page_text:
                    continue
                total_characters += len(page_text)
                if total_characters > self._max_extracted_characters:
                    raise ReportTooLargeError(
                        "report exceeds the extracted-character limit"
                    )
                pages.append(ExtractedPage(page_number=page_number, text=page_text))
        except (ReportProcessingError, ReportTooLargeError):
            raise
        except Exception as error:
            raise ReportProcessingError("PDF could not be parsed") from error

        if not pages:
            raise ReportProcessingError("PDF contains no extractable text")
        text = "\n\n".join(page.text for page in pages)
        return text, page_count, tuple(pages)

    def _normalise_and_bound(self, text: str) -> str:
        normalised = self._normalise_text(text)
        if not normalised:
            raise ReportProcessingError("report contains no usable text")
        if len(normalised) > self._max_extracted_characters:
            raise ReportTooLargeError("report exceeds the extracted-character limit")
        return normalised

    @staticmethod
    def _normalise_text(text: str) -> str:
        return text.replace("\r\n", "\n").replace("\r", "\n").strip()


class InMemoryReportStore:
    """Retain bounded complete reports and enforce session ownership."""

    def __init__(
        self,
        *,
        max_reports_per_session: int,
        id_factory: Callable[[], UUID] = uuid4,
    ) -> None:
        if max_reports_per_session < 1:
            raise ValueError("max_reports_per_session must be at least 1")
        self._max_reports_per_session = max_reports_per_session
        self._id_factory = id_factory
        self._records: dict[UUID, StoredReport] = {}
        self._owners: dict[UUID, UUID] = {}
        self._session_file_ids: dict[UUID, list[UUID]] = defaultdict(list)
        self._lock = Lock()

    @property
    def max_reports_per_session(self) -> int:
        """Return the maximum retained reports for one session."""
        return self._max_reports_per_session

    @property
    def readiness_detail(self) -> str:
        """Describe the process-local store without exposing internal state."""
        return (
            f"retains up to {self.max_reports_per_session} complete reports per session"
        )

    async def open(self) -> None:
        """The process-local store has no loop-bound resources."""

    async def aclose(self) -> None:
        """The process-local store has nothing to release."""

    async def check(self) -> None:
        """The process-local store has no external dependency to check."""

    async def add(
        self, session_id: UUID, report: ProcessedReport
    ) -> ReportUploadResponse:
        """Allocate a globally unique file ID and retain the complete report."""
        with self._lock:
            session_file_ids = self._session_file_ids[session_id]
            if len(session_file_ids) >= self._max_reports_per_session:
                raise ReportCapacityError("session report capacity reached")

            file_id: UUID | None = None
            for _ in range(100):
                candidate = self._id_factory()
                if candidate not in self._records:
                    file_id = candidate
                    break
            if file_id is None:
                raise RuntimeError("could not allocate a unique file ID")

            metadata = ReportUploadResponse(
                file_id=file_id,
                filename=report.filename,
                media_type=report.media_type,
                byte_size=report.byte_size,
                extracted_text_length=len(report.extracted_text),
                sha256=report.sha256,
                page_count=report.page_count,
            )
            self._records[file_id] = StoredReport(
                metadata=metadata,
                original_bytes=report.original_bytes,
                extracted_text=report.extracted_text,
                pages=report.pages,
            )
            self._owners[file_id] = session_id
            session_file_ids.append(file_id)
            return metadata

    async def list_for_session(
        self, session_id: UUID
    ) -> tuple[ReportUploadResponse, ...]:
        """List retained report metadata in upload order, without report text."""
        with self._lock:
            return tuple(
                self._records[file_id].metadata
                for file_id in self._session_file_ids.get(session_id, ())
            )

    async def delete(self, session_id: UUID, file_id: UUID) -> None:
        """Remove one owned report's original bytes, metadata, and extracted text."""
        with self._lock:
            if self._owners.get(file_id) != session_id:
                raise ReportNotFoundError from None
            del self._records[file_id]
            del self._owners[file_id]
            self._session_file_ids[session_id].remove(file_id)

    async def resolve(
        self,
        session_id: UUID,
        file_ids: Sequence[UUID],
    ) -> tuple[StoredReport, ...]:
        """Resolve reports only when every ID belongs to the requesting session."""
        with self._lock:
            resolved: list[StoredReport] = []
            for file_id in file_ids:
                if self._owners.get(file_id) != session_id:
                    raise ReportNotFoundError from None
                resolved.append(self._records[file_id])
            return tuple(resolved)

    async def touch(self, session_id: UUID) -> None:
        """Process-local reports have no idle expiry to renew."""
        del session_id

    def seed(
        self,
        session_id: UUID,
        reports: Sequence[StoredReport],
    ) -> None:
        """Insert reports with predetermined IDs; fail on any collision.

        Intended for deterministic startup seeding.  Not part of the
        ``ReportStore`` protocol.
        """
        with self._lock:
            for report in reports:
                if report.metadata.file_id in self._records:
                    raise ReportCapacityError(
                        f"file_id collision: {report.metadata.file_id} "
                        f"already exists in the report store"
                    )
            for report in reports:
                fid = report.metadata.file_id
                self._records[fid] = report
                self._owners[fid] = session_id
                self._session_file_ids[session_id].append(fid)


class _RedisReportLock(Protocol):
    """Subset of the redis.asyncio distributed lock used by the report store."""

    async def acquire(self) -> bool:
        """Acquire the configured lock within its blocking timeout."""
        ...

    async def release(self) -> None:
        """Release a lock owned by this client."""
        ...


class _RedisReportPipeline(Protocol):
    """Subset of a redis.asyncio MULTI/EXEC pipeline; commands buffer locally."""

    def set(self, name: str, value: str, *, ex: int | None = None) -> object:
        """Queue one string write, with an expiry in seconds when given."""
        ...

    def expire(self, name: str, time: int) -> object:
        """Queue a TTL reset; Redis ignores it when the key does not exist."""
        ...

    def delete(self, *names: str) -> object:
        """Queue removal of the named keys."""
        ...

    async def execute(self) -> list[object]:
        """Apply every queued command atomically, or raise and apply none."""
        ...


class _RedisReportClient(Protocol):
    """Small redis.asyncio surface kept injectable for offline tests."""

    async def ping(self) -> bool:
        """Check connectivity to the Redis service."""
        ...

    async def get(self, name: str) -> str | bytes | None:
        """Return one serialized report index or report content value."""
        ...

    def pipeline(self, transaction: bool = True) -> _RedisReportPipeline:
        """Return a pipeline that wraps its commands in MULTI/EXEC."""
        ...

    def lock(
        self,
        name: str,
        *,
        timeout: float,
        blocking_timeout: float,
    ) -> _RedisReportLock:
        """Return a distributed lock for one session's reports."""
        ...

    async def aclose(self) -> None:
        """Close the client and its connection pool."""
        ...


_REPORT_PAYLOAD_VERSION = 2


class _StoredPagePayload(BaseModel):
    """One persisted PDF page; strict so booleans are not accepted as numbers."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    page_number: int = Field(ge=1)
    text: str = Field(min_length=1)


class _ReportIndexPayload(BaseModel):
    """Per-session report metadata in upload order, without any content."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal[2]
    reports: tuple[ReportUploadResponse, ...]


class _ReportContentPayload(BaseModel):
    """One report's original bytes and extracted text, keyed by its file ID."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal[2]
    file_id: UUID
    original_content_base64: str
    extracted_text: str = Field(min_length=1)
    pages: tuple[_StoredPagePayload, ...]


class RedisReportStore:
    """Persist report metadata and each report's content under separate keys.

    `<prefix>:reports:<session>:index` holds every report's public metadata in
    upload order and decides ownership. `<prefix>:report:<session>:<file>`
    holds one report's original bytes, extracted text and pages. Listing reads
    only the index, and resolving reads only the requested content keys, so a
    request never loads reports it does not use. Mutations hold a per-session
    distributed lock and write their keys in one MULTI/EXEC transaction.

    Every write of a session's report keys sets the idle TTL with `SET ... EX`
    and `touch` renews all of them together. Seeded demo reports are written
    without a TTL; callers do not touch read-only sessions.

    Redis calls and lock waits are awaited on the event loop. Only the JSON
    and base64 work on content values, which can reach several megabytes, runs
    in a worker thread. Content loads are bounded by `report_load_concurrency`
    so concurrent requests cannot hold unbounded report content in memory.
    A store built by `from_url` creates its loop-bound client in `open()` and
    closes it in `aclose()`; a client passed in is owned by the caller.
    """

    _PAYLOAD_VERSION = _REPORT_PAYLOAD_VERSION

    def __init__(
        self,
        client: _RedisReportClient | None = None,
        *,
        client_factory: Callable[[], _RedisReportClient] | None = None,
        max_reports_per_session: int,
        key_prefix: str = "agentic-eval-toyapp",
        lock_timeout_seconds: float = 1_800.0,
        lock_blocking_timeout_seconds: float = 10.0,
        report_load_concurrency: int = 4,
        idle_ttl_seconds: int = 2_592_000,
        id_factory: Callable[[], UUID] = uuid4,
    ) -> None:
        if (client is None) == (client_factory is None):
            raise ValueError("provide exactly one of client or client_factory")
        if max_reports_per_session < 1:
            raise ValueError("max_reports_per_session must be at least 1")
        normalised_key_prefix = key_prefix.strip().rstrip(":")
        if not normalised_key_prefix:
            raise ValueError("key_prefix must not be blank")
        if lock_timeout_seconds <= 0:
            raise ValueError("lock_timeout_seconds must be positive")
        if lock_blocking_timeout_seconds < 0:
            raise ValueError("lock_blocking_timeout_seconds must not be negative")
        if report_load_concurrency < 1:
            raise ValueError("report_load_concurrency must be at least 1")
        if idle_ttl_seconds < 1:
            raise ValueError("idle_ttl_seconds must be at least 1")
        self._client = client
        self._client_factory = client_factory
        self._owns_client = False
        self._max_reports_per_session = max_reports_per_session
        self._key_prefix = normalised_key_prefix
        self._lock_timeout_seconds = lock_timeout_seconds
        self._lock_blocking_timeout_seconds = lock_blocking_timeout_seconds
        self._idle_ttl_seconds = idle_ttl_seconds
        self._id_factory = id_factory
        # anyio's semaphore binds to an event loop on first use, so building it
        # here does not require a running loop.
        self._load_semaphore = anyio.Semaphore(report_load_concurrency)

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        max_reports_per_session: int,
        key_prefix: str = "agentic-eval-toyapp",
        socket_timeout_seconds: float = 5.0,
        lock_timeout_seconds: float = 1_800.0,
        lock_blocking_timeout_seconds: float = 10.0,
        report_load_concurrency: int = 4,
        idle_ttl_seconds: int = 2_592_000,
    ) -> "RedisReportStore":
        """Describe a Redis client that `open()` creates on the running loop."""

        def build_client() -> _RedisReportClient:
            client = Redis.from_url(  # pyright: ignore[reportUnknownMemberType]
                url,
                decode_responses=True,
                socket_connect_timeout=socket_timeout_seconds,
                socket_timeout=socket_timeout_seconds,
            )
            return cast(_RedisReportClient, client)

        return cls(
            client_factory=build_client,
            max_reports_per_session=max_reports_per_session,
            key_prefix=key_prefix,
            lock_timeout_seconds=lock_timeout_seconds,
            lock_blocking_timeout_seconds=lock_blocking_timeout_seconds,
            report_load_concurrency=report_load_concurrency,
            idle_ttl_seconds=idle_ttl_seconds,
        )

    @property
    def max_reports_per_session(self) -> int:
        """Return the maximum retained reports for one session."""
        return self._max_reports_per_session

    @property
    def readiness_detail(self) -> str:
        """Describe Redis persistence without exposing its connection URL."""
        return (
            "Redis reachable; retains original and extracted content for up to "
            f"{self.max_reports_per_session} reports per session"
        )

    async def open(self) -> None:
        """Create the loop-bound client once; later calls change nothing."""
        if self._client is None and self._client_factory is not None:
            self._client = self._client_factory()
            self._owns_client = True

    async def aclose(self) -> None:
        """Close a client this store created; an injected client is left open."""
        client, self._client = self._client, None
        owned, self._owns_client = self._owns_client, False
        if client is not None and owned:
            try:
                await client.aclose()
            except RedisError as error:
                raise ReportStoreUnavailableError(
                    "Redis client could not be closed"
                ) from error

    def _require_client(self) -> _RedisReportClient:
        if self._client is None:
            raise ReportStoreUnavailableError(
                "Redis client is not open; run the application lifespan"
            )
        return self._client

    async def check(self) -> None:
        """Verify Redis connectivity without reading application data."""
        client = self._require_client()
        try:
            if not await client.ping():
                raise ReportStoreUnavailableError("Redis ping did not succeed")
        except RedisError as error:
            raise ReportStoreUnavailableError("Redis is unavailable") from error

    async def add(
        self, session_id: UUID, report: ProcessedReport
    ) -> ReportUploadResponse:
        """Persist one report's content and index entry in one transaction."""
        async with self._session_lock(session_id):
            index = await self._read_index(session_id)
            if len(index) >= self.max_reports_per_session:
                raise ReportCapacityError("session report capacity reached")
            existing_ids = {metadata.file_id for metadata in index}
            file_id: UUID | None = None
            for _ in range(100):
                candidate = self._id_factory()
                if candidate not in existing_ids:
                    file_id = candidate
                    break
            if file_id is None:
                raise RuntimeError("could not allocate a unique file ID")
            metadata = ReportUploadResponse(
                file_id=file_id,
                filename=report.filename,
                media_type=report.media_type,
                byte_size=report.byte_size,
                extracted_text_length=len(report.extracted_text),
                sha256=report.sha256,
                page_count=report.page_count,
            )
            stored = StoredReport(
                metadata=metadata,
                original_bytes=report.original_bytes,
                extracted_text=report.extracted_text,
                pages=report.pages,
            )
            content = await to_thread.run_sync(self._serialize_content, stored)
            pipeline = self._require_client().pipeline(transaction=True)
            pipeline.set(
                self._content_key(session_id, file_id),
                content,
                ex=self._idle_ttl_seconds,
            )
            pipeline.set(
                self._index_key(session_id),
                self._serialize_index((*index, metadata)),
                ex=self._idle_ttl_seconds,
            )
            await self._execute(pipeline)
            return metadata

    async def list_for_session(
        self, session_id: UUID
    ) -> tuple[ReportUploadResponse, ...]:
        """List report metadata from the index without loading any content."""
        return await self._read_index(session_id)

    async def delete(self, session_id: UUID, file_id: UUID) -> None:
        """Delete one report's content and index entry from its owning session."""
        async with self._session_lock(session_id):
            index = await self._read_index(session_id)
            retained = tuple(
                metadata for metadata in index if metadata.file_id != file_id
            )
            if len(retained) == len(index):
                raise ReportNotFoundError from None
            pipeline = self._require_client().pipeline(transaction=True)
            pipeline.delete(self._content_key(session_id, file_id))
            if retained:
                pipeline.set(
                    self._index_key(session_id),
                    self._serialize_index(retained),
                    ex=self._idle_ttl_seconds,
                )
            else:
                pipeline.delete(self._index_key(session_id))
            await self._execute(pipeline)

    async def resolve(
        self,
        session_id: UUID,
        file_ids: Sequence[UUID],
    ) -> tuple[StoredReport, ...]:
        """Load only the requested reports, after checking session ownership."""
        if not file_ids:
            return ()
        metadata_by_id = {
            metadata.file_id: metadata
            for metadata in await self._read_index(session_id)
        }
        if any(file_id not in metadata_by_id for file_id in file_ids):
            raise ReportNotFoundError from None
        loaded: dict[UUID, StoredReport] = {}
        for file_id in file_ids:
            if file_id not in loaded:
                loaded[file_id] = await self._read_content(
                    session_id, metadata_by_id[file_id]
                )
        return tuple(loaded[file_id] for file_id in file_ids)

    async def touch(self, session_id: UUID) -> None:
        """Renew the idle TTL of the index and every content key in one EXEC.

        Only the small index is read. `EXPIRE` never creates a key, so a report
        removed concurrently is not resurrected; a concurrent `add` writes its
        own keys with the TTL.
        """
        index = await self._read_index(session_id)
        if not index:
            return
        pipeline = self._require_client().pipeline(transaction=True)
        pipeline.expire(self._index_key(session_id), self._idle_ttl_seconds)
        for metadata in index:
            pipeline.expire(
                self._content_key(session_id, metadata.file_id),
                self._idle_ttl_seconds,
            )
        await self._execute(pipeline)

    async def seed(
        self,
        session_id: UUID,
        reports: Sequence[StoredReport],
    ) -> None:
        """Write predetermined reports if no data exists for this session.

        Intended for deterministic startup seeding.  Not part of the
        ``ReportStore`` protocol.
        """
        if await self._read_index(session_id):
            raise ReportStoreUnavailableError(
                "report store already contains data for the demo session"
            )
        if not reports:
            return
        pipeline = self._require_client().pipeline(transaction=True)
        for report in reports:
            content = await to_thread.run_sync(self._serialize_content, report)
            pipeline.set(
                self._content_key(session_id, report.metadata.file_id), content
            )
        pipeline.set(
            self._index_key(session_id),
            self._serialize_index(tuple(report.metadata for report in reports)),
        )
        await self._execute(pipeline)

    def _index_key(self, session_id: UUID) -> str:
        return f"{self._key_prefix}:reports:{session_id}:index"

    def _content_key(self, session_id: UUID, file_id: UUID) -> str:
        return f"{self._key_prefix}:report:{session_id}:{file_id}"

    def _lock_key(self, session_id: UUID) -> str:
        return f"{self._key_prefix}:reports:{session_id}:mutation-lock"

    async def _get(self, key: str) -> str | None:
        client = self._require_client()
        try:
            raw = await client.get(key)
        except RedisError as error:
            raise ReportStoreUnavailableError("Redis is unavailable") from error
        if isinstance(raw, bytes):
            try:
                return raw.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ReportStoreUnavailableError(
                    "Redis contains invalid report data"
                ) from error
        return raw

    async def _read_index(self, session_id: UUID) -> tuple[ReportUploadResponse, ...]:
        raw = await self._get(self._index_key(session_id))
        if raw is None:
            return ()
        # The index is metadata only, so it is parsed on the event loop.
        return self._deserialize_index(raw)

    async def _read_content(
        self, session_id: UUID, metadata: ReportUploadResponse
    ) -> StoredReport:
        async with self._load_semaphore:
            raw = await self._get(self._content_key(session_id, metadata.file_id))
            if raw is None:
                # The index lists a report whose content key is gone; treat the
                # report as deleted rather than failing the request.
                raise ReportNotFoundError from None
            return await to_thread.run_sync(self._deserialize_content, raw, metadata)

    @staticmethod
    async def _execute(pipeline: _RedisReportPipeline) -> None:
        try:
            await pipeline.execute()
        except RedisError as error:
            raise ReportStoreUnavailableError("Redis is unavailable") from error

    @asynccontextmanager
    async def _session_lock(self, session_id: UUID) -> AsyncGenerator[None, None]:
        client = self._require_client()
        try:
            lock = client.lock(
                self._lock_key(session_id),
                timeout=self._lock_timeout_seconds,
                blocking_timeout=self._lock_blocking_timeout_seconds,
            )
            acquired = await lock.acquire()
        except (LockError, RedisError) as error:
            raise ReportStoreUnavailableError("Redis locking is unavailable") from error
        if not acquired:
            raise ReportStoreBusyError
        try:
            yield
        finally:
            try:
                # Shielded so a cancelled request still frees the distributed lock.
                with anyio.CancelScope(shield=True):
                    await lock.release()
            except (LockError, RedisError) as error:
                raise ReportStoreUnavailableError(
                    "Redis report lock could not be released"
                ) from error

    def _serialize_index(self, reports: tuple[ReportUploadResponse, ...]) -> str:
        return _ReportIndexPayload(
            version=_REPORT_PAYLOAD_VERSION, reports=reports
        ).model_dump_json()

    def _deserialize_index(self, raw: str) -> tuple[ReportUploadResponse, ...]:
        try:
            payload = _ReportIndexPayload.model_validate_json(raw)
            if len(payload.reports) > self.max_reports_per_session:
                raise ValueError("report capacity exceeded in stored data")
            if len({metadata.file_id for metadata in payload.reports}) != len(
                payload.reports
            ):
                raise ValueError("duplicate stored file ID")
        except ValueError as error:
            raise ReportStoreUnavailableError(
                "Redis contains invalid report data"
            ) from error
        return payload.reports

    @staticmethod
    def _serialize_content(report: StoredReport) -> str:
        return _ReportContentPayload(
            version=_REPORT_PAYLOAD_VERSION,
            file_id=report.metadata.file_id,
            original_content_base64=base64.b64encode(report.original_bytes).decode(
                "ascii"
            ),
            extracted_text=report.extracted_text,
            pages=tuple(
                _StoredPagePayload(page_number=page.page_number, text=page.text)
                for page in report.pages
            ),
        ).model_dump_json()

    def _deserialize_content(
        self, raw: str, metadata: ReportUploadResponse
    ) -> StoredReport:
        try:
            payload = _ReportContentPayload.model_validate_json(raw)
            if payload.file_id != metadata.file_id:
                raise ValueError("stored content belongs to another file ID")
            original_bytes = base64.b64decode(
                payload.original_content_base64, validate=True
            )
            pages = tuple(
                ExtractedPage(page_number=page.page_number, text=page.text)
                for page in payload.pages
            )
            self._validate_stored_report(
                metadata=metadata,
                original_bytes=original_bytes,
                extracted_text=payload.extracted_text,
                pages=pages,
            )
        except (binascii.Error, ValueError) as error:
            raise ReportStoreUnavailableError(
                "Redis contains invalid report data"
            ) from error
        return StoredReport(
            metadata=metadata,
            original_bytes=original_bytes,
            extracted_text=payload.extracted_text,
            pages=pages,
        )

    @staticmethod
    def _validate_stored_report(
        *,
        metadata: ReportUploadResponse,
        original_bytes: bytes,
        extracted_text: str,
        pages: Sequence[ExtractedPage],
    ) -> None:
        if metadata.byte_size != len(original_bytes):
            raise ValueError("stored original content size does not match metadata")
        if metadata.sha256 != hashlib.sha256(original_bytes).hexdigest():
            raise ValueError("stored original content hash does not match metadata")
        if metadata.extracted_text_length != len(extracted_text):
            raise ValueError("stored extracted text length does not match metadata")
        if metadata.media_type is ReportMediaType.PDF:
            if not pages or metadata.page_count is None:
                raise ValueError("stored PDF pages are missing")
            if any(page.page_number > metadata.page_count for page in pages):
                raise ValueError("stored PDF page exceeds page count")
            if extracted_text != "\n\n".join(page.text for page in pages):
                raise ValueError("stored PDF text does not match its pages")
        elif pages:
            raise ValueError("stored non-PDF report contains pages")
