"""Stable report source spans for model input and reference validation."""

from dataclasses import dataclass
from uuid import UUID

from agentic_eval_toyapp.models import (
    PageReportLocator,
    ReportLocator,
    ReportMediaType,
    SectionReportLocator,
)
from agentic_eval_toyapp.orchestration_models import CoordinatorAttachment


@dataclass(frozen=True, slots=True)
class SourceBlock:
    """Complete text for one report page, section, or plain-text document."""

    file_id: UUID
    locator: ReportLocator
    text: str


def _source_sections(attachment: CoordinatorAttachment) -> tuple[SourceBlock, ...]:
    if attachment.media_type is ReportMediaType.PDF:
        return tuple(
            SourceBlock(
                file_id=attachment.file_id,
                locator=PageReportLocator(page_number=page.page_number),
                text=page.text,
            )
            for page in attachment.pages
            if page.text.strip()
        )
    if attachment.media_type is ReportMediaType.TEXT:
        return (
            SourceBlock(
                file_id=attachment.file_id,
                locator=SectionReportLocator(section="Document"),
                text=attachment.extracted_text,
            ),
        )

    sections: list[SourceBlock] = []
    current_name = "Document"
    current_lines: list[str] = []
    seen_names: dict[str, int] = {}

    def append_section() -> None:
        text = "\n".join(current_lines).strip()
        if not text:
            return
        count = seen_names.get(current_name, 0) + 1
        seen_names[current_name] = count
        name = current_name if count == 1 else f"{current_name} ({count})"
        sections.append(
            SourceBlock(
                file_id=attachment.file_id,
                locator=SectionReportLocator(section=name),
                text=text,
            )
        )

    for line in attachment.extracted_text.split("\n"):
        if line.startswith("#") and line.lstrip("#").startswith(" "):
            append_section()
            current_name = line.lstrip("# ").strip() or "Untitled"
            current_lines = [line]
        else:
            current_lines.append(line)
    append_section()
    return tuple(sections)


def source_blocks(
    attachments: tuple[CoordinatorAttachment, ...],
) -> tuple[SourceBlock, ...]:
    """Preserve all extracted report content with stable page/section locators."""
    return tuple(
        block for attachment in attachments for block in _source_sections(attachment)
    )
