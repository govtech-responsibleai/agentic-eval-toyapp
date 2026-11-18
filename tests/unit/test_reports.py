"""Tests for bounded report processing and session-scoped storage."""

import hashlib
from collections.abc import Iterator
from io import BytesIO
from uuid import UUID, uuid4

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
from tests.support.fake_clock import FakeClock

from agentic_eval_toyapp.models import ReportMediaType
from agentic_eval_toyapp.reports import (
    ExtractedPage,
    InMemoryReportStore,
    ProcessedReport,
    ReportCapacityError,
    ReportNotFoundError,
    ReportProcessingError,
    ReportProcessor,
    ReportTooLargeError,
    UnsupportedReportTypeError,
)
from agentic_eval_toyapp.session_memory import SessionMemoryLedger, report_footprint


def _processor(
    *,
    max_bytes: int = 5_000,
    max_pdf_pages: int = 5,
    max_extracted_characters: int = 1_000,
) -> ReportProcessor:
    return ReportProcessor(
        max_bytes=max_bytes,
        max_pdf_pages=max_pdf_pages,
        max_extracted_characters=max_extracted_characters,
    )


def _text_pdf(*page_texts: str) -> bytes:
    writer = PdfWriter()
    for text in page_texts:
        page = writer.add_blank_page(width=612, height=792)
        if not text:
            continue
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        font_reference = writer._add_object(font)  # pyright: ignore[reportPrivateUsage]
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_reference})}
        )
        stream = DecodedStreamObject()
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream.set_data(f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(  # pyright: ignore[reportPrivateUsage]
            stream
        )
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def _processed_report(text: str = "Findings") -> ProcessedReport:
    content = text.encode()
    return ProcessedReport(
        filename="report.txt",
        media_type=ReportMediaType.TEXT,
        byte_size=len(content),
        original_bytes=content,
        extracted_text=text,
        sha256=hashlib.sha256(content).hexdigest(),
    )


def test_text_and_markdown_processing_is_reproducible() -> None:
    """UTF-8 text should be normalised while its raw-byte checksum stays stable."""
    processor = _processor()
    content = "\ufeff  Revenue rose 12%.\r\n第二行  \r\n".encode()

    text_report = processor.process(
        filename="C:\\fakepath\\report.txt",
        media_type="text/plain; charset=utf-8",
        content=content,
    )
    markdown_report = processor.process(
        filename="summary.md",
        media_type="text/markdown",
        content=b"# Summary\n\nFinding",
    )

    assert text_report.filename == "report.txt"
    assert text_report.original_bytes == content
    assert text_report.extracted_text == "Revenue rose 12%.\n第二行"
    assert text_report.sha256 == hashlib.sha256(content).hexdigest()
    assert text_report.pages == ()
    assert markdown_report.media_type is ReportMediaType.MARKDOWN


def test_text_pdf_preserves_page_numbers_and_combined_text() -> None:
    """PDF extraction should retain source pages for future report references."""
    content = _text_pdf(
        "Revenue increased by 12 percent.",
        "",
        "Risk remains high.",
    )
    report = _processor().process(
        filename="quarterly.pdf",
        media_type="application/pdf",
        content=content,
    )

    assert report.media_type is ReportMediaType.PDF
    assert report.original_bytes == content
    assert report.extracted_text == (
        "Revenue increased by 12 percent.\n\nRisk remains high."
    )
    assert report.page_count == 3
    assert [(page.page_number, page.text) for page in report.pages] == [
        (1, "Revenue increased by 12 percent."),
        (3, "Risk remains high."),
    ]


@pytest.mark.parametrize(
    ("filename", "media_type", "content"),
    [
        ("report.csv", "text/csv", b"a,b"),
        ("report.txt", "application/pdf", b"plain text"),
        ("report.pdf", "text/plain", b"%PDF-not-really"),
        ("report.pdf", "application/pdf", b"not a PDF"),
        ("\x00report.txt", "text/plain", b"content"),
    ],
)
def test_processor_rejects_unsupported_or_mismatched_types(
    filename: str,
    media_type: str,
    content: bytes,
) -> None:
    """Extension, MIME type, signature, and filename must agree."""
    with pytest.raises(UnsupportedReportTypeError):
        _processor().process(
            filename=filename,
            media_type=media_type,
            content=content,
        )


@pytest.mark.parametrize(
    "content",
    [b"", b"   \r\n", b"\xff\xfeinvalid"],
)
def test_processor_rejects_empty_or_invalid_text(content: bytes) -> None:
    """Text reports must contain non-whitespace valid UTF-8."""
    with pytest.raises(ReportProcessingError):
        _processor().process(
            filename="report.txt",
            media_type="text/plain",
            content=content,
        )


def test_processor_rejects_malformed_image_only_and_encrypted_pdfs() -> None:
    """Only unencrypted PDFs containing extractable text are accepted."""
    malformed = b"%PDF-1.7\nmalformed"
    image_only = _text_pdf("")
    encrypted_writer = PdfWriter()
    encrypted_writer.add_blank_page(width=612, height=792)
    encrypted_writer.encrypt("secret")
    encrypted_output = BytesIO()
    encrypted_writer.write(encrypted_output)

    for content in (malformed, image_only, encrypted_output.getvalue()):
        with pytest.raises(ReportProcessingError):
            _processor().process(
                filename="report.pdf",
                media_type="application/pdf",
                content=content,
            )


def test_processor_enforces_raw_page_and_text_bounds() -> None:
    """All three configured resource bounds should fail closed."""
    with pytest.raises(ReportTooLargeError, match="byte"):
        _processor(max_bytes=3).process(
            filename="report.txt",
            media_type="text/plain",
            content=b"four",
        )
    with pytest.raises(ReportTooLargeError, match="page"):
        _processor(max_pdf_pages=1).process(
            filename="report.pdf",
            media_type="application/pdf",
            content=_text_pdf("one", "two"),
        )
    with pytest.raises(ReportTooLargeError, match="character"):
        _processor(max_extracted_characters=3).process(
            filename="report.txt",
            media_type="text/plain",
            content=b"four",
        )


@pytest.mark.parametrize(
    "limits",
    [
        {"max_bytes": 0, "max_pdf_pages": 1, "max_extracted_characters": 1},
        {"max_bytes": 1, "max_pdf_pages": 0, "max_extracted_characters": 1},
        {"max_bytes": 1, "max_pdf_pages": 1, "max_extracted_characters": 0},
    ],
)
def test_processor_rejects_non_positive_limits(limits: dict[str, int]) -> None:
    """Invalid bounds must fail during dependency construction."""
    with pytest.raises(ValueError, match="positive"):
        ReportProcessor(**limits)


@pytest.mark.anyio
async def test_store_enforces_ownership_and_retains_original_bytes() -> None:
    """Only the owning session may resolve complete immutable content."""
    store = InMemoryReportStore(max_reports_per_session=2)
    owner = uuid4()
    other = uuid4()
    metadata = await store.add(owner, _processed_report("Known finding"))

    stored = (await store.resolve(owner, (metadata.file_id,)))[0]

    assert stored.extracted_text == "Known finding"
    assert stored.original_bytes == b"Known finding"
    with pytest.raises(ReportNotFoundError):
        await store.resolve(other, (metadata.file_id,))
    with pytest.raises(ReportNotFoundError):
        await store.resolve(owner, (uuid4(),))


@pytest.mark.anyio
async def test_store_retries_id_collisions_and_enforces_session_capacity() -> None:
    """ID collisions cannot overwrite data and capacity cannot silently evict it."""
    first_id = UUID("00000000-0000-0000-0000-000000000001")
    second_id = UUID("00000000-0000-0000-0000-000000000002")

    def generated_ids() -> Iterator[UUID]:
        yield first_id
        yield first_id
        yield second_id

    ids = generated_ids()
    store = InMemoryReportStore(
        max_reports_per_session=2,
        id_factory=lambda: next(ids),
    )
    session_id = uuid4()

    first = await store.add(session_id, _processed_report("first"))
    second = await store.add(session_id, _processed_report("second"))

    assert first.file_id == first_id
    assert second.file_id == second_id
    assert (await store.resolve(session_id, (first_id,)))[0].extracted_text == "first"
    with pytest.raises(ReportCapacityError):
        await store.add(session_id, _processed_report("third"))
    with pytest.raises(ValueError, match="at least 1"):
        InMemoryReportStore(max_reports_per_session=0)


@pytest.mark.anyio
async def test_store_lists_and_deletes_only_owned_reports() -> None:
    """Deletion releases capacity without leaking another session's metadata."""
    store = InMemoryReportStore(max_reports_per_session=2)
    owner = uuid4()
    other = uuid4()
    first = await store.add(owner, _processed_report("first"))
    second = await store.add(owner, _processed_report("second"))
    foreign = await store.add(other, _processed_report("foreign"))

    assert await store.list_for_session(owner) == (first, second)
    assert await store.list_for_session(other) == (foreign,)
    assert await store.list_for_session(uuid4()) == ()
    with pytest.raises(ReportNotFoundError):
        await store.delete(other, first.file_id)
    with pytest.raises(ReportNotFoundError):
        await store.delete(owner, uuid4())

    await store.delete(owner, first.file_id)
    assert await store.list_for_session(owner) == (second,)
    assert await store.list_for_session(other) == (foreign,)
    with pytest.raises(ReportNotFoundError):
        await store.resolve(owner, (first.file_id,))
    with pytest.raises(ReportNotFoundError):
        await store.delete(owner, first.file_id)
    third = await store.add(owner, _processed_report("third"))
    assert await store.list_for_session(owner) == (second, third)


def _pdf_like_report() -> ProcessedReport:
    pages = (
        ExtractedPage(page_number=1, text="alpha"),
        ExtractedPage(page_number=3, text="gamma!"),
    )
    content = b"%PDF-fake-original-bytes"
    return ProcessedReport(
        filename="report.pdf",
        media_type=ReportMediaType.PDF,
        byte_size=len(content),
        original_bytes=content,
        extracted_text="alpha\n\ngamma!",
        sha256=hashlib.sha256(content).hexdigest(),
        page_count=3,
        pages=pages,
    )


def _tracked_report_store() -> tuple[InMemoryReportStore, SessionMemoryLedger, UUID]:
    ledger = SessionMemoryLedger(clock=FakeClock())
    session_id = uuid4()
    ledger.track(session_id)
    return (
        InMemoryReportStore(max_reports_per_session=10, ledger=ledger),
        ledger,
        session_id,
    )


def _report_bytes(ledger: SessionMemoryLedger, session_id: UUID) -> int:
    footprint = ledger.footprint(session_id)
    assert footprint is not None
    return footprint.report_bytes


@pytest.mark.anyio
async def test_added_text_report_counts_original_bytes_and_text() -> None:
    """A text report has no pages, so its text is counted once."""
    store, ledger, session_id = _tracked_report_store()

    await store.add(session_id, _processed_report("Findings"))

    assert _report_bytes(ledger, session_id) == len(b"Findings") + len("Findings")


@pytest.mark.anyio
async def test_added_pdf_report_counts_page_text_on_top_of_extracted_text() -> None:
    """A PDF stores joined text and every page, so both count."""
    store, ledger, session_id = _tracked_report_store()
    report = _pdf_like_report()

    await store.add(session_id, report)

    assert _report_bytes(ledger, session_id) == (
        len(report.original_bytes)
        + len(report.extracted_text)
        + len("alpha")
        + len("gamma!")
    )


@pytest.mark.anyio
async def test_deleting_reports_returns_the_estimate_to_its_start() -> None:
    """Add then delete subtracts exactly what was added."""
    store, ledger, session_id = _tracked_report_store()
    before = ledger.total_bytes
    text = await store.add(session_id, _processed_report("Findings"))
    pdf = await store.add(session_id, _pdf_like_report())

    await store.delete(session_id, text.file_id)
    assert _report_bytes(ledger, session_id) == report_footprint(
        b"%PDF-fake-original-bytes", "alpha\n\ngamma!", ("alpha", "gamma!")
    )
    await store.delete(session_id, pdf.file_id)

    assert _report_bytes(ledger, session_id) == 0
    assert ledger.total_bytes == before


@pytest.mark.anyio
async def test_rejected_report_operations_leave_the_estimate_unchanged() -> None:
    """A capacity refusal or a foreign delete changes nothing."""
    ledger = SessionMemoryLedger(clock=FakeClock())
    owner, other = uuid4(), uuid4()
    ledger.track(owner)
    ledger.track(other)
    store = InMemoryReportStore(max_reports_per_session=1, ledger=ledger)
    stored = await store.add(owner, _processed_report())
    total = ledger.total_bytes

    with pytest.raises(ReportCapacityError):
        await store.add(owner, _processed_report())
    with pytest.raises(ReportNotFoundError):
        await store.delete(other, stored.file_id)

    assert ledger.total_bytes == total
    assert _report_bytes(ledger, other) == 0


@pytest.mark.anyio
async def test_seeded_reports_are_never_counted() -> None:
    """Demo reports stay outside the ledger."""
    ledger = SessionMemoryLedger(clock=FakeClock())
    store = InMemoryReportStore(max_reports_per_session=10, ledger=ledger)
    demo_id = uuid4()
    source = InMemoryReportStore(max_reports_per_session=10)
    await source.add(demo_id, _processed_report())
    seeded = await source.resolve_available(demo_id)

    store.seed(demo_id, seeded)

    assert ledger.footprint(demo_id) is None
    assert ledger.total_bytes == 0
