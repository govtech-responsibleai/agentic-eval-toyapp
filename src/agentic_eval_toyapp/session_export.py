"""Portable session export as JSON or ZIP."""

import io
import json
import zipfile
from dataclasses import dataclass
from uuid import UUID

from agentic_eval_toyapp.reports import ReportStore, StoredReport
from agentic_eval_toyapp.sessions import SessionStore

FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class ExportedFile:
    """File metadata in the portable session format."""

    file_id: str
    filename: str
    media_type: str
    byte_size: int
    sha256: str
    extracted_text: str
    extracted_text_length: int
    page_count: int | None = None
    pages: tuple[dict[str, object], ...] = ()


@dataclass(frozen=True, slots=True)
class PortableSession:
    """Versioned portable representation of one session."""

    format_version: int
    session_id: str
    turns: tuple[dict[str, str], ...]
    files: tuple[ExportedFile, ...]


def _build_exported_file(report: StoredReport) -> ExportedFile:
    """Build an ExportedFile from a StoredReport."""
    return ExportedFile(
        file_id=str(report.metadata.file_id),
        filename=report.metadata.filename,
        media_type=report.metadata.media_type.value,
        byte_size=report.metadata.byte_size,
        sha256=report.metadata.sha256,
        extracted_text=report.extracted_text,
        extracted_text_length=report.metadata.extracted_text_length,
        page_count=report.metadata.page_count,
        pages=tuple(
            {"page_number": page.page_number, "text": page.text}
            for page in report.pages
        ),
    )


def _portable_session_to_dict(session: PortableSession) -> dict[str, object]:
    """Serialize a PortableSession to a JSON-compatible dict."""
    return {
        "format_version": session.format_version,
        "session_id": session.session_id,
        "turns": list(session.turns),
        "files": [
            {
                "file_id": f.file_id,
                "filename": f.filename,
                "media_type": f.media_type,
                "byte_size": f.byte_size,
                "sha256": f.sha256,
                "extracted_text": f.extracted_text,
                "extracted_text_length": f.extracted_text_length,
                "page_count": f.page_count,
                "pages": list(f.pages),
            }
            for f in session.files
        ],
    }


async def export_session(
    session_id: UUID,
    session_store: SessionStore,
    report_store: ReportStore,
) -> tuple[bytes, str, str]:
    """Export a session as bytes, content_type, and filename.

    Returns JSON when no files exist, ZIP when files are present.
    """
    session = await session_store.get(session_id)

    file_metadata = await report_store.list_for_session(session_id)
    file_ids = [meta.file_id for meta in file_metadata]

    reports: tuple[StoredReport, ...] = ()
    if file_ids:
        reports = await report_store.resolve(session_id, file_ids)

    exported_files = tuple(_build_exported_file(r) for r in reports)

    portable = PortableSession(
        format_version=FORMAT_VERSION,
        session_id=str(session.session_id),
        turns=tuple(
            {"role": msg.role.value, "content": msg.content} for msg in session.messages
        ),
        files=exported_files,
    )

    session_dict = _portable_session_to_dict(portable)
    session_json = json.dumps(session_dict, indent=2, ensure_ascii=False).encode(
        "utf-8"
    )

    id_str = str(session_id)

    if not reports:
        return session_json, "application/json", f"session-{id_str}.json"

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("session.json", session_json)
        for report in reports:
            entry_path = f"uploaded/{report.metadata.file_id}"
            zf.writestr(entry_path, report.original_bytes)
    zip_bytes = buffer.getvalue()

    return zip_bytes, "application/zip", f"session-{id_str}.zip"
