"""Load, validate, and seed the bundled demonstration session."""

from __future__ import annotations

import hashlib
import json
import logging
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import cast
from uuid import UUID

from agentic_eval_toyapp.models import (
    ConversationMessage,
    ConversationRole,
    ReportMediaType,
    ReportUploadResponse,
)
from agentic_eval_toyapp.reports import (
    ExtractedPage,
    InMemoryReportStore,
    RedisReportStore,
    ReportStore,
    StoredReport,
)
from agentic_eval_toyapp.session_export import FORMAT_VERSION
from agentic_eval_toyapp.sessions import (
    InMemorySessionStore,
    RedisSessionStore,
    SessionStore,
)

LOGGER = logging.getLogger("uvicorn.error")

DEMO_RESOURCE = Path(__file__).parent / "resources" / "demo-session.json"

_ZIP_SIGNATURE = b"PK\x03\x04"


class DemoLoadError(RuntimeError):
    """Raised when the bundled demo session is invalid or inconsistent."""


@dataclass(frozen=True, slots=True)
class LoadedDemo:
    """Validated demo session ready to be seeded into stores."""

    session_id: UUID
    turns: tuple[tuple[ConversationMessage, ConversationMessage], ...]
    reports: tuple[StoredReport, ...]
    langfuse_project_url: str | None = None


def load_demo(path: Path = DEMO_RESOURCE) -> LoadedDemo:
    """Load and validate the demo session resource.

    Detects ZIP by checking the first four bytes for the PK\\x03\\x04
    signature; otherwise treats the content as JSON.

    Raises ``DemoLoadError`` for any structural or integrity problem.
    """
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise DemoLoadError(f"cannot read demo resource: {error}") from error

    if not raw:
        raise DemoLoadError("demo resource is empty")

    is_zip = raw[:4] == _ZIP_SIGNATURE

    if is_zip:
        return _load_from_zip(raw)
    return _load_from_json(raw, allow_files=False)


def _parse_session_json(data: bytes) -> dict[str, object]:
    """Decode and structurally validate a session JSON payload."""
    try:
        decoded = json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise DemoLoadError(f"demo JSON is malformed: {error}") from error

    if not isinstance(decoded, dict):
        raise DemoLoadError("demo JSON root must be an object")
    return cast(dict[str, object], decoded)


def _validate_common(payload: dict[str, object]) -> tuple[UUID, list[dict[str, str]]]:
    """Validate format_version, session_id, and turns structure."""
    if payload.get("format_version") != FORMAT_VERSION:
        raise DemoLoadError(
            f"unsupported format_version "
            f"(expected {FORMAT_VERSION}, got {payload.get('format_version')!r})"
        )

    raw_sid = payload.get("session_id")
    if not isinstance(raw_sid, str):
        raise DemoLoadError("session_id must be a string")
    try:
        session_id = UUID(raw_sid)
    except ValueError as error:
        raise DemoLoadError(f"session_id is not a valid UUID: {error}") from error

    raw_turns = payload.get("turns")
    if not isinstance(raw_turns, list):
        raise DemoLoadError("turns must be a list")

    turns: list[dict[str, str]] = []
    for index, raw_turn in enumerate(cast(list[object], raw_turns)):
        if not isinstance(raw_turn, dict):
            raise DemoLoadError(f"turn {index} must be an object")
        turn = cast(dict[str, str], raw_turn)
        role = turn.get("role")
        content = turn.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise DemoLoadError(f"turn {index} must have string role and content")
        if not content.strip():
            raise DemoLoadError(f"turn {index} has empty content")
        turns.append(turn)

    if len(turns) % 2 != 0:
        raise DemoLoadError("turns must form complete user/assistant pairs")

    for pair_index in range(0, len(turns), 2):
        if turns[pair_index].get("role") != "user":
            raise DemoLoadError(
                f"turn {pair_index} must be 'user', "
                f"got {turns[pair_index].get('role')!r}"
            )
        if turns[pair_index + 1].get("role") != "assistant":
            raise DemoLoadError(
                f"turn {pair_index + 1} must be 'assistant', "
                f"got {turns[pair_index + 1].get('role')!r}"
            )

    return session_id, turns


def _optional_langfuse_project_url(payload: dict[str, object]) -> str | None:
    """Read the optional ``metadata.langfuse_project_url`` override."""
    metadata = payload.get("metadata")
    if metadata is None:
        return None
    if not isinstance(metadata, dict):
        raise DemoLoadError("metadata must be an object")
    raw_url = cast(dict[str, object], metadata).get("langfuse_project_url")
    if raw_url is None:
        return None
    if not isinstance(raw_url, str):
        raise DemoLoadError("metadata.langfuse_project_url must be a string")
    url = raw_url.strip().rstrip("/")
    if not url.startswith(("http://", "https://")):
        raise DemoLoadError(
            "metadata.langfuse_project_url must start with http:// or https://"
        )
    return url


def _build_turn_pairs(
    turns: list[dict[str, str]],
) -> tuple[tuple[ConversationMessage, ConversationMessage], ...]:
    """Convert validated turn dicts to ConversationMessage pairs."""
    pairs: list[tuple[ConversationMessage, ConversationMessage]] = []
    for pair_index in range(0, len(turns), 2):
        user = ConversationMessage(
            role=ConversationRole.USER,
            content=turns[pair_index]["content"],
        )
        assistant = ConversationMessage(
            role=ConversationRole.ASSISTANT,
            content=turns[pair_index + 1]["content"],
        )
        pairs.append((user, assistant))
    return tuple(pairs)


def _load_from_json(data: bytes, *, allow_files: bool) -> LoadedDemo:
    """Load a demo from a plain JSON payload."""
    payload = _parse_session_json(data)
    session_id, turns = _validate_common(payload)

    raw_files = payload.get("files")
    if not isinstance(raw_files, list):
        raise DemoLoadError("files must be a list")

    if raw_files and not allow_files:
        raise DemoLoadError(
            "JSON demo declares files but cannot carry file bytes; "
            "use ZIP format for sessions with files"
        )

    return LoadedDemo(
        session_id=session_id,
        turns=_build_turn_pairs(turns),
        reports=(),
        langfuse_project_url=_optional_langfuse_project_url(payload),
    )


def _load_from_zip(data: bytes) -> LoadedDemo:
    """Load a demo from a ZIP archive with session.json and uploaded/ entries."""
    try:
        zf = zipfile.ZipFile(BytesIO(data))
    except zipfile.BadZipFile as error:
        raise DemoLoadError(f"demo ZIP is corrupt: {error}") from error

    with zf:
        if "session.json" not in zf.namelist():
            raise DemoLoadError("demo ZIP must contain session.json")

        session_data = zf.read("session.json")
        payload = _parse_session_json(session_data)
        session_id, turns = _validate_common(payload)

        raw_files = payload.get("files")
        if not isinstance(raw_files, list):
            raise DemoLoadError("files must be a list")

        declared_ids: set[str] = set()
        reports: list[StoredReport] = []

        for file_index, raw_file in enumerate(cast(list[object], raw_files)):
            if not isinstance(raw_file, dict):
                raise DemoLoadError(f"file {file_index} must be an object")
            file_entry = cast(dict[str, object], raw_file)
            report = _validate_and_build_report(file_entry, file_index, zf)
            file_id_str = str(report.metadata.file_id)
            if file_id_str in declared_ids:
                raise DemoLoadError(f"duplicate file_id {file_id_str}")
            declared_ids.add(file_id_str)
            reports.append(report)

        # Verify every uploaded/ entry is declared
        uploaded_entries = {
            name.removeprefix("uploaded/")
            for name in zf.namelist()
            if name.startswith("uploaded/")
        }
        if uploaded_entries != declared_ids:
            extra = uploaded_entries - declared_ids
            missing = declared_ids - uploaded_entries
            parts: list[str] = []
            if extra:
                parts.append(f"undeclared entries: {extra}")
            if missing:
                parts.append(f"missing entries: {missing}")
            raise DemoLoadError(
                f"ZIP uploaded/ entries do not match declared files: {'; '.join(parts)}"
            )

    return LoadedDemo(
        session_id=session_id,
        turns=_build_turn_pairs(turns),
        reports=tuple(reports),
        langfuse_project_url=_optional_langfuse_project_url(payload),
    )


def _validate_and_build_report(
    entry: dict[str, object],
    index: int,
    zf: zipfile.ZipFile,
) -> StoredReport:
    """Validate one file entry and build a StoredReport from the ZIP."""
    raw_fid = entry.get("file_id")
    if not isinstance(raw_fid, str):
        raise DemoLoadError(f"file {index}: file_id must be a string")
    try:
        file_id = UUID(raw_fid)
    except ValueError as error:
        raise DemoLoadError(
            f"file {index}: file_id is not a valid UUID: {error}"
        ) from error

    filename = entry.get("filename")
    if not isinstance(filename, str) or not filename.strip():
        raise DemoLoadError(f"file {index}: filename must be a non-empty string")

    raw_media = entry.get("media_type")
    if not isinstance(raw_media, str):
        raise DemoLoadError(f"file {index}: media_type must be a string")
    try:
        media_type = ReportMediaType(raw_media)
    except ValueError as error:
        raise DemoLoadError(
            f"file {index}: unsupported media_type {raw_media!r}: {error}"
        ) from error

    raw_byte_size = entry.get("byte_size")
    if not isinstance(raw_byte_size, int) or isinstance(raw_byte_size, bool):
        raise DemoLoadError(f"file {index}: byte_size must be an integer")

    raw_sha256 = entry.get("sha256")
    if not isinstance(raw_sha256, str) or len(raw_sha256) != 64:
        raise DemoLoadError(f"file {index}: sha256 must be a 64-character hex string")

    raw_extracted = entry.get("extracted_text")
    if not isinstance(raw_extracted, str) or not raw_extracted.strip():
        raise DemoLoadError(f"file {index}: extracted_text must be a non-empty string")

    raw_etl = entry.get("extracted_text_length")
    if not isinstance(raw_etl, int) or isinstance(raw_etl, bool):
        raise DemoLoadError(f"file {index}: extracted_text_length must be an integer")

    raw_page_count = entry.get("page_count")
    page_count: int | None = None
    if raw_page_count is not None:
        if not isinstance(raw_page_count, int) or isinstance(raw_page_count, bool):
            raise DemoLoadError(f"file {index}: page_count must be an integer or null")
        page_count = raw_page_count

    # Build pages
    raw_pages = entry.get("pages", [])
    if not isinstance(raw_pages, list):
        raise DemoLoadError(f"file {index}: pages must be a list")
    pages: list[ExtractedPage] = []
    for page_data in cast(list[object], raw_pages):
        if not isinstance(page_data, dict):
            raise DemoLoadError(f"file {index}: each page must be an object")
        typed_page = cast(dict[str, object], page_data)
        pn = typed_page.get("page_number")
        pt = typed_page.get("text")
        if (
            not isinstance(pn, int)
            or isinstance(pn, bool)
            or pn < 1
            or not isinstance(pt, str)
            or not pt.strip()
        ):
            raise DemoLoadError(f"file {index}: invalid page entry")
        pages.append(ExtractedPage(page_number=pn, text=pt))

    # Read and verify bytes from ZIP
    zip_path = f"uploaded/{file_id}"
    try:
        original_bytes = zf.read(zip_path)
    except KeyError:
        raise DemoLoadError(f"file {index}: missing ZIP entry {zip_path}") from None

    if len(original_bytes) != raw_byte_size:
        raise DemoLoadError(
            f"file {index}: byte_size mismatch "
            f"(declared {raw_byte_size}, actual {len(original_bytes)})"
        )

    actual_hash = hashlib.sha256(original_bytes).hexdigest()
    if actual_hash != raw_sha256:
        raise DemoLoadError(
            f"file {index}: sha256 mismatch "
            f"(declared {raw_sha256}, actual {actual_hash})"
        )

    metadata = ReportUploadResponse(
        file_id=file_id,
        filename=filename,
        media_type=media_type,
        byte_size=raw_byte_size,
        extracted_text_length=raw_etl,
        sha256=raw_sha256,
        page_count=page_count,
    )

    return StoredReport(
        metadata=metadata,
        original_bytes=original_bytes,
        extracted_text=raw_extracted,
        pages=tuple(pages),
    )


async def seed_demo(
    demo: LoadedDemo,
    session_store: SessionStore,
    report_store: ReportStore,
) -> bool:
    """Seed the demo session into both stores atomically.

    Returns ``True`` when the demo was freshly seeded, ``False`` when it
    already existed.  Raises on a partial conflict (session exists but
    file IDs collide with another session).
    """
    session_seeded = await _seed_session(demo, session_store)
    if not session_seeded:
        return False

    try:
        await _seed_reports(demo, report_store)
    except Exception:
        # Atomic rollback: remove the session we just created.
        await _rollback_session(demo.session_id, session_store)
        raise

    return True


async def _seed_session(
    demo: LoadedDemo,
    store: SessionStore,
) -> bool:
    """Insert the demo session; return False if it already exists."""
    if isinstance(store, InMemorySessionStore):
        return store.seed(demo.session_id, demo.turns, read_only=True)
    if isinstance(store, RedisSessionStore):
        return await store.seed(demo.session_id, demo.turns, read_only=True)
    raise DemoLoadError(f"unsupported session store type: {type(store).__name__}")


async def _seed_reports(
    demo: LoadedDemo,
    store: ReportStore,
) -> None:
    """Insert all demo reports atomically."""
    if not demo.reports:
        return

    if isinstance(store, InMemoryReportStore):
        store.seed(demo.session_id, demo.reports)
    elif isinstance(store, RedisReportStore):
        await store.seed(demo.session_id, demo.reports)
    else:
        raise DemoLoadError(f"unsupported report store type: {type(store).__name__}")


async def _rollback_session(
    session_id: UUID,
    store: SessionStore,
) -> None:
    """Best-effort removal of a session that was partially seeded."""
    try:
        if isinstance(store, InMemorySessionStore):
            store.unseed(session_id)
        elif isinstance(store, RedisSessionStore):
            await store.unseed(session_id)
    except Exception:
        LOGGER.warning(
            "demo_rollback_failed session_id=%s",
            session_id,
            exc_info=True,
        )
