"""Unit tests for the portable session export module."""

import hashlib
import io
import json
import zipfile
from collections.abc import Sequence
from uuid import UUID, uuid4

import pytest

from agentic_eval_toyapp.models import (
    AgentTurnResult,
    ConversationMessage,
    ConversationRole,
    ReportMediaType,
    ReportUploadResponse,
    SessionResponse,
)
from agentic_eval_toyapp.reports import ProcessedReport, StoredReport
from agentic_eval_toyapp.session_export import (
    FORMAT_VERSION,
    export_session,
)
from agentic_eval_toyapp.sessions import AgentRunner, SessionNotFoundError


class _StubSessionStore:
    """Minimal session store that returns a canned session."""

    def __init__(self, sessions: dict[UUID, SessionResponse] | None = None) -> None:
        self._sessions = sessions or {}

    @property
    def max_turns(self) -> int:
        return 10

    @property
    def readiness_detail(self) -> str:
        return "stub"

    async def open(self) -> None:
        pass

    async def aclose(self) -> None:
        pass

    async def check(self) -> None:
        pass

    async def create(self) -> SessionResponse:
        sid = uuid4()
        session = SessionResponse(session_id=sid, retained_turns=0)
        self._sessions[sid] = session
        return session

    async def get(self, session_id: UUID) -> SessionResponse:
        if session_id not in self._sessions:
            raise SessionNotFoundError
        return self._sessions[session_id]

    async def run_turn(
        self,
        session_id: UUID,
        user_message: str,
        *,
        agent: AgentRunner,
        reports: Sequence[StoredReport] = (),
    ) -> AgentTurnResult:
        raise NotImplementedError


class _StubReportStore:
    """Minimal report store that returns canned reports."""

    def __init__(
        self,
        reports: dict[UUID, list[StoredReport]] | None = None,
    ) -> None:
        self._reports = reports or {}

    @property
    def max_reports_per_session(self) -> int:
        return 10

    @property
    def readiness_detail(self) -> str:
        return "stub"

    async def open(self) -> None:
        pass

    async def aclose(self) -> None:
        pass

    async def check(self) -> None:
        pass

    async def add(
        self, session_id: UUID, report: ProcessedReport
    ) -> ReportUploadResponse:
        raise NotImplementedError

    async def list_for_session(
        self, session_id: UUID
    ) -> tuple[ReportUploadResponse, ...]:
        return tuple(r.metadata for r in self._reports.get(session_id, []))

    async def delete(self, session_id: UUID, file_id: UUID) -> None:
        raise NotImplementedError

    async def resolve(
        self, session_id: UUID, file_ids: Sequence[UUID]
    ) -> tuple[StoredReport, ...]:
        by_id = {r.metadata.file_id: r for r in self._reports.get(session_id, [])}
        return tuple(by_id[fid] for fid in file_ids)


def _make_text_report(
    file_id: UUID | None = None,
    filename: str = "notes.txt",
    content: bytes = b"Hello world",
) -> StoredReport:
    fid = file_id or uuid4()
    digest = hashlib.sha256(content).hexdigest()
    return StoredReport(
        metadata=ReportUploadResponse(
            file_id=fid,
            filename=filename,
            media_type=ReportMediaType.TEXT,
            byte_size=len(content),
            extracted_text_length=len(content.decode("utf-8")),
            sha256=digest,
            page_count=None,
        ),
        original_bytes=content,
        extracted_text=content.decode("utf-8"),
        pages=(),
    )


@pytest.mark.anyio
async def test_export_empty_session_returns_json() -> None:
    """An empty session with no files exports as JSON."""
    sid = uuid4()
    session = SessionResponse(session_id=sid, retained_turns=0)
    store = _StubSessionStore({sid: session})
    report_store = _StubReportStore()

    data, content_type, filename = await export_session(sid, store, report_store)

    assert content_type == "application/json"
    assert filename == f"session-{sid}.json"

    payload = json.loads(data)
    assert payload["format_version"] == FORMAT_VERSION
    assert payload["session_id"] == str(sid)
    assert payload["turns"] == []
    assert payload["files"] == []


@pytest.mark.anyio
async def test_export_session_with_messages_returns_json() -> None:
    """A session with turns but no files exports as JSON with turns."""
    sid = uuid4()
    session = SessionResponse(
        session_id=sid,
        messages=(
            ConversationMessage(role=ConversationRole.USER, content="hello"),
            ConversationMessage(role=ConversationRole.ASSISTANT, content="hi there"),
        ),
        retained_turns=1,
    )
    store = _StubSessionStore({sid: session})
    report_store = _StubReportStore()

    data, content_type, _filename = await export_session(sid, store, report_store)

    assert content_type == "application/json"
    payload = json.loads(data)
    assert len(payload["turns"]) == 2
    assert payload["turns"][0] == {"role": "user", "content": "hello"}
    assert payload["turns"][1] == {"role": "assistant", "content": "hi there"}
    assert payload["files"] == []


@pytest.mark.anyio
async def test_export_session_with_files_returns_zip() -> None:
    """A session with files exports as ZIP containing session.json and uploaded files."""
    sid = uuid4()
    session = SessionResponse(session_id=sid, retained_turns=0)
    file_content = b"Report content here"
    report = _make_text_report(content=file_content)
    file_id = report.metadata.file_id

    store = _StubSessionStore({sid: session})
    report_store = _StubReportStore({sid: [report]})

    data, content_type, filename = await export_session(sid, store, report_store)

    assert content_type == "application/zip"
    assert filename == f"session-{sid}.zip"

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = set(zf.namelist())
        assert "session.json" in names
        assert f"uploaded/{file_id}" in names

        session_json = json.loads(zf.read("session.json"))
        assert session_json["format_version"] == FORMAT_VERSION
        assert len(session_json["files"]) == 1
        exported_file = session_json["files"][0]
        assert exported_file["file_id"] == str(file_id)
        assert exported_file["filename"] == "notes.txt"
        assert exported_file["byte_size"] == len(file_content)
        assert exported_file["sha256"] == hashlib.sha256(file_content).hexdigest()
        assert exported_file["extracted_text"] == file_content.decode("utf-8")
        assert exported_file["extracted_text_length"] == len(
            file_content.decode("utf-8")
        )

        raw_bytes = zf.read(f"uploaded/{file_id}")
        assert raw_bytes == file_content


@pytest.mark.anyio
async def test_export_zip_entries_match_declared_files() -> None:
    """ZIP entries under uploaded/ must exactly match the declared files array."""
    sid = uuid4()
    session = SessionResponse(session_id=sid, retained_turns=0)
    reports = [_make_text_report(content=b"A"), _make_text_report(content=b"B")]

    store = _StubSessionStore({sid: session})
    report_store = _StubReportStore({sid: reports})

    data, content_type, _ = await export_session(sid, store, report_store)
    assert content_type == "application/zip"

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        session_json = json.loads(zf.read("session.json"))
        declared_ids = {f["file_id"] for f in session_json["files"]}
        zip_uploaded = {
            name.removeprefix("uploaded/")
            for name in zf.namelist()
            if name.startswith("uploaded/")
        }
        assert declared_ids == zip_uploaded


@pytest.mark.anyio
async def test_export_file_hashes_and_sizes_are_correct() -> None:
    """Exported file metadata must have correct sha256 and byte_size."""
    sid = uuid4()
    session = SessionResponse(session_id=sid, retained_turns=0)
    content = b"Deterministic content for hash verification"
    report = _make_text_report(content=content)

    store = _StubSessionStore({sid: session})
    report_store = _StubReportStore({sid: [report]})

    data, _, _ = await export_session(sid, store, report_store)

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        session_json = json.loads(zf.read("session.json"))
        exported = session_json["files"][0]
        raw = zf.read(f"uploaded/{exported['file_id']}")
        assert exported["byte_size"] == len(raw)
        assert exported["sha256"] == hashlib.sha256(raw).hexdigest()


@pytest.mark.anyio
async def test_export_missing_session_raises() -> None:
    """Exporting a nonexistent session raises SessionNotFoundError."""
    store = _StubSessionStore()
    report_store = _StubReportStore()

    with pytest.raises(SessionNotFoundError):
        await export_session(uuid4(), store, report_store)


@pytest.mark.anyio
async def test_export_does_not_include_read_only_or_traces() -> None:
    """Exported JSON must not contain Langfuse traces or read_only fields."""
    sid = uuid4()
    session = SessionResponse(session_id=sid, retained_turns=0)
    store = _StubSessionStore({sid: session})
    report_store = _StubReportStore()

    data, _, _ = await export_session(sid, store, report_store)
    payload = json.loads(data)

    raw_text = data.decode("utf-8")
    assert "langfuse" not in raw_text.lower()
    assert "read_only" not in raw_text
    assert "trace" not in raw_text.lower()
    # Only expected top-level keys
    assert set(payload.keys()) == {"format_version", "session_id", "turns", "files"}


@pytest.mark.anyio
async def test_export_json_has_empty_files_array() -> None:
    """A JSON export (no files) must have an empty files array, not absent."""
    sid = uuid4()
    session = SessionResponse(session_id=sid, retained_turns=0)
    store = _StubSessionStore({sid: session})
    report_store = _StubReportStore()

    data, content_type, _ = await export_session(sid, store, report_store)
    assert content_type == "application/json"

    payload = json.loads(data)
    assert "files" in payload
    assert payload["files"] == []


@pytest.mark.anyio
async def test_export_filenames_are_display_metadata() -> None:
    """Archive paths use file UUIDs, not the original filenames."""
    sid = uuid4()
    session = SessionResponse(session_id=sid, retained_turns=0)
    report = _make_text_report(filename="my report.txt", content=b"content")

    store = _StubSessionStore({sid: session})
    report_store = _StubReportStore({sid: [report]})

    data, _, _ = await export_session(sid, store, report_store)

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for name in zf.namelist():
            assert "my report" not in name
        session_json = json.loads(zf.read("session.json"))
        assert session_json["files"][0]["filename"] == "my report.txt"
