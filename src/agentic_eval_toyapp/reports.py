"""Bounded report extraction and memory or Redis-backed storage."""

import base64
import binascii
import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import PurePath
from threading import Lock
from typing import Protocol, cast
from uuid import UUID, uuid4

from pypdf import PdfReader
from redis import Redis
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

    def check(self) -> None:
        """Raise when the backing store cannot serve requests."""
        ...

    def add(self, session_id: UUID, report: ProcessedReport) -> ReportUploadResponse:
        """Persist one processed report under its owning session."""
        ...

    def list_for_session(self, session_id: UUID) -> tuple[ReportUploadResponse, ...]:
        """Return safe metadata for a session's reports in upload order."""
        ...

    def delete(self, session_id: UUID, file_id: UUID) -> None:
        """Delete one report owned by the supplied session."""
        ...

    def resolve(
        self,
        session_id: UUID,
        file_ids: Sequence[UUID],
    ) -> tuple[StoredReport, ...]:
        """Resolve selected reports only for their owning session."""
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

    def check(self) -> None:
        """The process-local store has no external dependency to check."""

    def add(self, session_id: UUID, report: ProcessedReport) -> ReportUploadResponse:
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

    def list_for_session(self, session_id: UUID) -> tuple[ReportUploadResponse, ...]:
        """List retained report metadata in upload order, without report text."""
        with self._lock:
            return tuple(
                self._records[file_id].metadata
                for file_id in self._session_file_ids.get(session_id, ())
            )

    def delete(self, session_id: UUID, file_id: UUID) -> None:
        """Remove one owned report's original bytes, metadata, and extracted text."""
        with self._lock:
            if self._owners.get(file_id) != session_id:
                raise ReportNotFoundError from None
            del self._records[file_id]
            del self._owners[file_id]
            self._session_file_ids[session_id].remove(file_id)

    def resolve(
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


class _RedisReportLock(Protocol):
    """Subset of the redis-py distributed lock used by the report store."""

    def acquire(self) -> bool:
        """Acquire the configured lock within its blocking timeout."""
        ...

    def release(self) -> None:
        """Release a lock owned by this client."""
        ...


class _RedisReportClient(Protocol):
    """Small redis-py surface kept injectable for offline tests."""

    def ping(self) -> bool:
        """Check connectivity to the Redis service."""
        ...

    def set(self, name: str, value: str) -> object:
        """Persist one serialized session report collection."""
        ...

    def get(self, name: str) -> str | bytes | None:
        """Return one serialized session report collection."""
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


class RedisReportStore:
    """Persist original bytes, metadata, and extracted report text in Redis."""

    _PAYLOAD_VERSION = 1

    def __init__(
        self,
        client: _RedisReportClient,
        *,
        max_reports_per_session: int,
        key_prefix: str = "agentic-eval-toyapp",
        lock_timeout_seconds: float = 1_800.0,
        lock_blocking_timeout_seconds: float = 10.0,
        id_factory: Callable[[], UUID] = uuid4,
    ) -> None:
        if max_reports_per_session < 1:
            raise ValueError("max_reports_per_session must be at least 1")
        normalised_key_prefix = key_prefix.strip().rstrip(":")
        if not normalised_key_prefix:
            raise ValueError("key_prefix must not be blank")
        if lock_timeout_seconds <= 0:
            raise ValueError("lock_timeout_seconds must be positive")
        if lock_blocking_timeout_seconds < 0:
            raise ValueError("lock_blocking_timeout_seconds must not be negative")
        self._client = client
        self._max_reports_per_session = max_reports_per_session
        self._key_prefix = normalised_key_prefix
        self._lock_timeout_seconds = lock_timeout_seconds
        self._lock_blocking_timeout_seconds = lock_blocking_timeout_seconds
        self._id_factory = id_factory

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
    ) -> "RedisReportStore":
        """Construct a decoded Redis client without opening a connection yet."""
        client = Redis.from_url(  # pyright: ignore[reportUnknownMemberType]
            url,
            decode_responses=True,
            socket_connect_timeout=socket_timeout_seconds,
            socket_timeout=socket_timeout_seconds,
        )
        return cls(
            cast(_RedisReportClient, client),
            max_reports_per_session=max_reports_per_session,
            key_prefix=key_prefix,
            lock_timeout_seconds=lock_timeout_seconds,
            lock_blocking_timeout_seconds=lock_blocking_timeout_seconds,
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

    def check(self) -> None:
        """Verify Redis connectivity without reading application data."""
        try:
            if not self._client.ping():
                raise ReportStoreUnavailableError("Redis ping did not succeed")
        except RedisError as error:
            raise ReportStoreUnavailableError("Redis is unavailable") from error

    def add(self, session_id: UUID, report: ProcessedReport) -> ReportUploadResponse:
        """Persist one complete report under a session-scoped mutation lock."""
        with self._session_lock(session_id):
            reports = self._read_reports(session_id)
            if len(reports) >= self.max_reports_per_session:
                raise ReportCapacityError("session report capacity reached")
            existing_ids = {stored.metadata.file_id for stored in reports}
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
            reports.append(
                StoredReport(
                    metadata=metadata,
                    original_bytes=report.original_bytes,
                    extracted_text=report.extracted_text,
                    pages=report.pages,
                )
            )
            self._write_reports(session_id, reports)
            return metadata

    def list_for_session(self, session_id: UUID) -> tuple[ReportUploadResponse, ...]:
        """List persistent report metadata without returning private contents."""
        return tuple(report.metadata for report in self._read_reports(session_id))

    def delete(self, session_id: UUID, file_id: UUID) -> None:
        """Delete one persistent report only from its owning session."""
        with self._session_lock(session_id):
            reports = self._read_reports(session_id)
            retained = [
                report for report in reports if report.metadata.file_id != file_id
            ]
            if len(retained) == len(reports):
                raise ReportNotFoundError from None
            self._write_reports(session_id, retained)

    def resolve(
        self,
        session_id: UUID,
        file_ids: Sequence[UUID],
    ) -> tuple[StoredReport, ...]:
        """Resolve persistent reports only for their owning session."""
        reports_by_id = {
            report.metadata.file_id: report for report in self._read_reports(session_id)
        }
        try:
            return tuple(reports_by_id[file_id] for file_id in file_ids)
        except KeyError:
            raise ReportNotFoundError from None

    def _session_key(self, session_id: UUID) -> str:
        return f"{self._key_prefix}:reports:{session_id}"

    def _lock_key(self, session_id: UUID) -> str:
        return f"{self._session_key(session_id)}:mutation-lock"

    def _read_reports(self, session_id: UUID) -> list[StoredReport]:
        try:
            raw = self._client.get(self._session_key(session_id))
        except RedisError as error:
            raise ReportStoreUnavailableError("Redis is unavailable") from error
        if raw is None:
            return []
        try:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ReportStoreUnavailableError(
                "Redis contains invalid report data"
            ) from error
        return self._deserialize(raw)

    def _write_reports(
        self,
        session_id: UUID,
        reports: Sequence[StoredReport],
    ) -> None:
        try:
            stored = self._client.set(
                self._session_key(session_id),
                self._serialize(reports),
            )
        except RedisError as error:
            raise ReportStoreUnavailableError("Redis is unavailable") from error
        if not stored:
            raise ReportStoreUnavailableError("Redis did not persist the reports")

    @contextmanager
    def _session_lock(self, session_id: UUID) -> Generator[None, None, None]:
        try:
            lock = self._client.lock(
                self._lock_key(session_id),
                timeout=self._lock_timeout_seconds,
                blocking_timeout=self._lock_blocking_timeout_seconds,
            )
            acquired = lock.acquire()
        except (LockError, RedisError) as error:
            raise ReportStoreUnavailableError("Redis locking is unavailable") from error
        if not acquired:
            raise ReportStoreBusyError
        try:
            yield
        finally:
            try:
                lock.release()
            except (LockError, RedisError) as error:
                raise ReportStoreUnavailableError(
                    "Redis report lock could not be released"
                ) from error

    def _serialize(self, reports: Sequence[StoredReport]) -> str:
        return json.dumps(
            {
                "version": self._PAYLOAD_VERSION,
                "reports": [
                    {
                        "metadata": report.metadata.model_dump(mode="json"),
                        "original_content_base64": base64.b64encode(
                            report.original_bytes
                        ).decode("ascii"),
                        "extracted_text": report.extracted_text,
                        "pages": [
                            {
                                "page_number": page.page_number,
                                "text": page.text,
                            }
                            for page in report.pages
                        ],
                    }
                    for report in reports
                ],
            },
            separators=(",", ":"),
        )

    def _deserialize(self, raw: str) -> list[StoredReport]:
        try:
            decoded = cast(object, json.loads(raw))
            if not isinstance(decoded, dict):
                raise ValueError("report payload must be an object")
            payload = cast(dict[str, object], decoded)
            if payload.get("version") != self._PAYLOAD_VERSION:
                raise ValueError("unsupported report payload version")
            decoded_reports = payload.get("reports")
            if not isinstance(decoded_reports, list):
                raise ValueError("reports must be a list")
            report_items = cast(list[object], decoded_reports)
            if len(report_items) > self.max_reports_per_session:
                raise ValueError("report capacity exceeded in stored data")
            reports: list[StoredReport] = []
            seen_ids: set[UUID] = set()
            for decoded_report in report_items:
                if not isinstance(decoded_report, dict):
                    raise ValueError("stored report must be an object")
                report_payload = cast(dict[str, object], decoded_report)
                metadata = ReportUploadResponse.model_validate(
                    report_payload.get("metadata")
                )
                if metadata.file_id in seen_ids:
                    raise ValueError("duplicate stored file ID")
                seen_ids.add(metadata.file_id)
                encoded_content = report_payload.get("original_content_base64")
                if not isinstance(encoded_content, str):
                    raise ValueError("stored original content must be text")
                original_bytes = base64.b64decode(encoded_content, validate=True)
                extracted_text = report_payload.get("extracted_text")
                if not isinstance(extracted_text, str) or not extracted_text:
                    raise ValueError("stored extracted text must not be empty")
                decoded_pages = report_payload.get("pages")
                if not isinstance(decoded_pages, list):
                    raise ValueError("stored pages must be a list")
                pages: list[ExtractedPage] = []
                for decoded_page in cast(list[object], decoded_pages):
                    if not isinstance(decoded_page, dict):
                        raise ValueError("stored page must be an object")
                    page_payload = cast(dict[str, object], decoded_page)
                    page_number = page_payload.get("page_number")
                    text = page_payload.get("text")
                    if (
                        not isinstance(page_number, int)
                        or isinstance(page_number, bool)
                        or page_number < 1
                        or not isinstance(text, str)
                        or not text
                    ):
                        raise ValueError("stored page is invalid")
                    pages.append(ExtractedPage(page_number=page_number, text=text))
                self._validate_stored_report(
                    metadata=metadata,
                    original_bytes=original_bytes,
                    extracted_text=extracted_text,
                    pages=pages,
                )
                reports.append(
                    StoredReport(
                        metadata=metadata,
                        original_bytes=original_bytes,
                        extracted_text=extracted_text,
                        pages=tuple(pages),
                    )
                )
            return reports
        except (binascii.Error, TypeError, UnicodeDecodeError, ValueError) as error:
            raise ReportStoreUnavailableError(
                "Redis contains invalid report data"
            ) from error

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
