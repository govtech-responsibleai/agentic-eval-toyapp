"""Edge cases for complete report source spans and stable locators."""

from io import BytesIO
from pathlib import Path
from uuid import uuid4

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from agentic_eval_toyapp.coordinator import CoordinatorAgent
from agentic_eval_toyapp.models import ReportMediaType, SectionReportLocator
from agentic_eval_toyapp.orchestration_models import (
    CoordinatorAttachment,
    CoordinatorAttachmentPage,
)
from agentic_eval_toyapp.report_sources import source_blocks
from agentic_eval_toyapp.reports import InMemoryReportStore, ReportProcessor


def _attachment(
    text: str,
    *,
    media_type: ReportMediaType = ReportMediaType.TEXT,
    pages: tuple[CoordinatorAttachmentPage, ...] = (),
) -> CoordinatorAttachment:
    return CoordinatorAttachment(
        file_id=uuid4(),
        filename="report.pdf" if media_type is ReportMediaType.PDF else "report.md",
        media_type=media_type,
        byte_size=max(1, len(text.encode())),
        extracted_text=text,
        pages=pages,
    )


def _text_pdf(*page_texts: str) -> bytes:
    """Build an extractable in-memory PDF with one optional blank page."""
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


def test_long_unicode_and_unbroken_text_remains_one_complete_source() -> None:
    """Without chunking, the complete Unicode report is model input."""
    text = "收入增长" * 60 + "X" * 10_000
    report = _attachment(text)

    blocks = source_blocks((report,))

    assert len(blocks) == 1
    assert blocks[0].text == text
    assert blocks[0].locator.kind == "section"
    assert blocks[0].locator.section == "Document"


def test_large_markdown_report_preserves_all_sections() -> None:
    """A documentation-sized upload must retain every labeled source span."""
    text = "\n".join(
        f"## Section {index}\n" + (f"Finding {index} is documented. " * 65)
        for index in range(34)
    )
    processor = ReportProcessor(
        max_bytes=5 * 1024 * 1024,
        max_pdf_pages=50,
        max_extracted_characters=100_000,
    )
    processed = processor.process(
        filename="architecture.md",
        media_type="text/markdown",
        content=text.encode(),
    )

    blocks = source_blocks(
        (
            CoordinatorAttachment(
                file_id=uuid4(),
                filename=processed.filename,
                media_type=processed.media_type,
                byte_size=processed.byte_size,
                extracted_text=processed.extracted_text,
            ),
        )
    )

    assert len(text) > 50_000
    assert len(blocks) == 34
    locator = blocks[-1].locator
    assert isinstance(locator, SectionReportLocator)
    assert locator.section == "Section 33"
    assert "Finding 33 is documented." in blocks[-1].text


def test_pdf_pages_keep_original_numbers_and_ignore_blank_pages() -> None:
    """PDF evidence points to the original page, not its extracted-text rank."""
    pages = (
        CoordinatorAttachmentPage(page_number=1, text="Revenue rose."),
        CoordinatorAttachmentPage(page_number=3, text="Risk increased."),
    )
    report = _attachment(
        "Revenue rose.\n\nRisk increased.",
        media_type=ReportMediaType.PDF,
        pages=pages,
    )

    blocks = source_blocks((report,))

    assert [
        block.locator.page_number for block in blocks if block.locator.kind == "page"
    ] == [1, 3]
    assert [block.text for block in blocks] == ["Revenue rose.", "Risk increased."]


def test_duplicate_markdown_headings_get_distinct_locators() -> None:
    """Repeated headings must not create ambiguous reference matches."""
    report = _attachment(
        "# Findings\nOne.\n# Findings\nTwo.", media_type=ReportMediaType.MARKDOWN
    )

    blocks = source_blocks((report,))

    assert [
        block.locator.section for block in blocks if block.locator.kind == "section"
    ] == [
        "Findings",
        "Findings (2)",
    ]


@pytest.mark.parametrize(
    ("filename", "media_type"),
    [
        ("long.txt", "text/plain"),
        ("long.md", "text/markdown"),
        ("long.md", "text/x-markdown"),
        ("long.md", "text/plain"),
        ("long.pdf", "application/pdf"),
    ],
)
def test_every_accepted_type_reaches_sources_without_text_loss(
    filename: str,
    media_type: str,
) -> None:
    """Extraction, session storage, and source labeling retain all nonblank lines."""
    revenue = "Revenue rose by 18 percent in the quarter. " * 20
    limitation = "The figures are unaudited management estimates. " * 12
    if filename.endswith(".pdf"):
        content = _text_pdf(revenue, "", limitation)
    elif filename.endswith(".md"):
        content = f"# Revenue\n{revenue}\n# Limitations\n{limitation}".encode()
    else:
        content = f"{revenue}\n{limitation}".encode()

    processed = ReportProcessor(
        max_bytes=100_000,
        max_pdf_pages=5,
        max_extracted_characters=10_000,
    ).process(filename=filename, media_type=media_type, content=content)
    session_id = uuid4()
    store = InMemoryReportStore(max_reports_per_session=1)
    metadata = store.add(session_id, processed)
    attachment = CoordinatorAgent.attachment_from_report(
        store.resolve(session_id, (metadata.file_id,))[0]
    )
    blocks = source_blocks((attachment,))

    assert blocks == source_blocks((attachment,))
    assert all(block.file_id == metadata.file_id for block in blocks)
    extracted_lines = [
        line.strip() for line in processed.extracted_text.splitlines() if line.strip()
    ]
    source_lines = [
        line.strip()
        for block in blocks
        for line in block.text.splitlines()
        if line.strip()
    ]
    assert source_lines == extracted_lines
    if filename.endswith(".pdf"):
        assert [
            block.locator.page_number
            for block in blocks
            if block.locator.kind == "page"
        ] == [1, 3]
    elif filename.endswith(".md"):
        assert [
            block.locator.section for block in blocks if block.locator.kind == "section"
        ] == [
            "Revenue",
            "Limitations",
        ]
    else:
        assert [
            block.locator.section for block in blocks if block.locator.kind == "section"
        ] == ["Document"]


def test_demo_report_includes_late_sections_in_one_source_set() -> None:
    """The checked-in report must expose its full contents to one analysis call."""
    fixture = Path(__file__).parents[1] / "fixtures" / "test_report.md"
    processed = ReportProcessor(
        max_bytes=5 * 1024 * 1024,
        max_pdf_pages=50,
        max_extracted_characters=100_000,
    ).process(
        filename=fixture.name,
        media_type="text/markdown",
        content=fixture.read_bytes(),
    )
    attachment = CoordinatorAttachment(
        file_id=uuid4(),
        filename=processed.filename,
        media_type=processed.media_type,
        byte_size=processed.byte_size,
        extracted_text=processed.extracted_text,
    )

    blocks = source_blocks((attachment,))

    assert len(processed.extracted_text) > 12_000
    assert len(blocks) == 12
    assert any(
        block.locator.section == "Data quality and stated limitations"
        for block in blocks
        if block.locator.kind == "section"
    )
    assert blocks[-1].locator.kind == "section"
    assert blocks[-1].locator.section == "Main findings and follow-up questions"
    assert "broader decision" in blocks[-1].text
