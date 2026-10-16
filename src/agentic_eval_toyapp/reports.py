"""Bounded report extraction and session-scoped process-local storage."""

import hashlib
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import PurePath
from threading import Lock
from uuid import UUID, uuid4

from pypdf import PdfReader

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
    extracted_text: str
    sha256: str
    page_count: int | None = None
    pages: tuple[ExtractedPage, ...] = ()


@dataclass(frozen=True, slots=True)
class StoredReport:
    """Immutable extracted content retained for one owning session."""

    metadata: ReportUploadResponse
    extracted_text: str
    pages: tuple[ExtractedPage, ...] = ()


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
        """Return bounded extracted content without retaining raw bytes."""
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
    """Retain bounded extracted reports and enforce session ownership."""

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
        self._session_file_ids: dict[UUID, set[UUID]] = defaultdict(set)
        self._lock = Lock()

    @property
    def max_reports_per_session(self) -> int:
        """Return the maximum retained reports for one session."""
        return self._max_reports_per_session

    def add(self, session_id: UUID, report: ProcessedReport) -> ReportUploadResponse:
        """Allocate a globally unique file ID and retain extracted content."""
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
                extracted_text=report.extracted_text,
                pages=report.pages,
            )
            self._owners[file_id] = session_id
            session_file_ids.add(file_id)
            return metadata

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
