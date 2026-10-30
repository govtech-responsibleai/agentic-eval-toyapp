"""Unit tests for the bundled demonstration session loader and seeder."""

import hashlib
import io
import json
import zipfile
from collections.abc import Sequence
from pathlib import Path
from uuid import UUID

import pytest

from agentic_eval_toyapp.demo import (
    DemoLoadError,
    load_demo,
    seed_demo,
)
from agentic_eval_toyapp.models import (
    ConversationRole,
    ReportMediaType,
    ReportUploadResponse,
)
from agentic_eval_toyapp.reports import InMemoryReportStore, StoredReport
from agentic_eval_toyapp.session_export import FORMAT_VERSION
from agentic_eval_toyapp.sessions import InMemorySessionStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEMO_SESSION_ID = "59e8b331-9992-4d2c-a059-4967604a6980"


def _minimal_json(
    *,
    session_id: str = DEMO_SESSION_ID,
    format_version: int = FORMAT_VERSION,
    turns: list[dict[str, str]] | None = None,
    files: Sequence[object] | None = None,
) -> bytes:
    if turns is None:
        turns = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
    payload: dict[str, object] = {
        "format_version": format_version,
        "session_id": session_id,
        "turns": turns,
        "files": list(files) if files is not None else [],
    }
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _write_tmp_json(tmp_path: Path, data: bytes) -> Path:
    path = tmp_path / "demo.json"
    path.write_bytes(data)
    return path


def _make_zip(
    session_json: bytes,
    uploaded: dict[str, bytes] | None = None,
) -> bytes:
    """Build a ZIP with session.json and optional uploaded/ entries."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("session.json", session_json)
        for name, content in (uploaded or {}).items():
            zf.writestr(f"uploaded/{name}", content)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# load_demo — valid JSON
# ---------------------------------------------------------------------------


def test_load_valid_json(tmp_path: Path) -> None:
    """A well-formed JSON demo loads successfully."""
    path = _write_tmp_json(tmp_path, _minimal_json())
    demo = load_demo(path)
    assert demo.session_id == UUID(DEMO_SESSION_ID)
    assert len(demo.turns) == 1
    assert demo.turns[0][0].role is ConversationRole.USER
    assert demo.turns[0][1].role is ConversationRole.ASSISTANT
    assert demo.reports == ()


def test_load_multiple_turn_pairs(tmp_path: Path) -> None:
    """Multiple user/assistant pairs are loaded correctly."""
    turns = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "A2"},
    ]
    path = _write_tmp_json(tmp_path, _minimal_json(turns=turns))
    demo = load_demo(path)
    assert len(demo.turns) == 2


def test_load_empty_turns_is_valid(tmp_path: Path) -> None:
    """A demo with zero turns is structurally valid."""
    path = _write_tmp_json(tmp_path, _minimal_json(turns=[]))
    demo = load_demo(path)
    assert demo.turns == ()


# ---------------------------------------------------------------------------
# load_demo — invalid JSON
# ---------------------------------------------------------------------------


def test_load_missing_file() -> None:
    """A missing file path raises DemoLoadError."""
    with pytest.raises(DemoLoadError, match="cannot read"):
        load_demo(Path("/nonexistent/demo.json"))


def test_load_empty_file(tmp_path: Path) -> None:
    """An empty file raises DemoLoadError."""
    path = tmp_path / "empty.json"
    path.write_bytes(b"")
    with pytest.raises(DemoLoadError, match="empty"):
        load_demo(path)


def test_load_malformed_json(tmp_path: Path) -> None:
    """Malformed JSON raises DemoLoadError."""
    path = _write_tmp_json(tmp_path, b"not json at all")
    with pytest.raises(DemoLoadError, match="malformed"):
        load_demo(path)


def test_load_wrong_format_version(tmp_path: Path) -> None:
    """An unsupported format_version raises DemoLoadError."""
    path = _write_tmp_json(tmp_path, _minimal_json(format_version=999))
    with pytest.raises(DemoLoadError, match="format_version"):
        load_demo(path)


def test_load_invalid_session_id(tmp_path: Path) -> None:
    """An invalid UUID for session_id raises DemoLoadError."""
    path = _write_tmp_json(tmp_path, _minimal_json(session_id="not-a-uuid"))
    with pytest.raises(DemoLoadError, match="UUID"):
        load_demo(path)


def test_load_odd_number_of_turns(tmp_path: Path) -> None:
    """An odd number of turns (incomplete pair) raises DemoLoadError."""
    turns = [{"role": "user", "content": "Q1"}]
    path = _write_tmp_json(tmp_path, _minimal_json(turns=turns))
    with pytest.raises(DemoLoadError, match="complete user/assistant pairs"):
        load_demo(path)


def test_load_wrong_turn_order(tmp_path: Path) -> None:
    """Turns that don't alternate user/assistant raise DemoLoadError."""
    turns = [
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q1"},
    ]
    path = _write_tmp_json(tmp_path, _minimal_json(turns=turns))
    with pytest.raises(DemoLoadError, match="must be 'user'"):
        load_demo(path)


def test_load_json_with_files_rejected(tmp_path: Path) -> None:
    """JSON format with non-empty files array is rejected."""
    files = [{"file_id": "00000000-0000-0000-0000-000000000001"}]
    path = _write_tmp_json(tmp_path, _minimal_json(files=files))
    with pytest.raises(DemoLoadError, match="cannot carry file bytes"):
        load_demo(path)


def test_load_empty_turn_content(tmp_path: Path) -> None:
    """A turn with empty content raises DemoLoadError."""
    turns = [
        {"role": "user", "content": "   "},
        {"role": "assistant", "content": "ok"},
    ]
    path = _write_tmp_json(tmp_path, _minimal_json(turns=turns))
    with pytest.raises(DemoLoadError, match="empty content"):
        load_demo(path)


# ---------------------------------------------------------------------------
# load_demo — valid ZIP
# ---------------------------------------------------------------------------


def _make_file_entry(
    file_id: str,
    content: bytes,
    filename: str = "report.txt",
    media_type: str = "text/plain",
) -> tuple[dict[str, object], bytes]:
    """Return a (file-metadata-dict, raw-bytes) pair for a ZIP demo."""
    digest = hashlib.sha256(content).hexdigest()
    entry: dict[str, object] = {
        "file_id": file_id,
        "filename": filename,
        "media_type": media_type,
        "byte_size": len(content),
        "sha256": digest,
        "extracted_text": content.decode("utf-8"),
        "extracted_text_length": len(content),
        "page_count": None,
        "pages": [],
    }
    return entry, content


def test_load_valid_zip(tmp_path: Path) -> None:
    """A well-formed ZIP demo loads with files."""
    fid = "00000000-0000-0000-0000-000000000001"
    raw_content = b"Demo report content"
    entry, content = _make_file_entry(fid, raw_content)

    session_json = _minimal_json(files=[entry])
    zip_bytes = _make_zip(session_json, uploaded={fid: content})
    path = tmp_path / "demo.zip"
    path.write_bytes(zip_bytes)

    demo = load_demo(path)
    assert demo.session_id == UUID(DEMO_SESSION_ID)
    assert len(demo.reports) == 1
    assert demo.reports[0].metadata.file_id == UUID(fid)
    assert demo.reports[0].original_bytes == raw_content


# ---------------------------------------------------------------------------
# load_demo — invalid ZIP
# ---------------------------------------------------------------------------


def test_load_zip_missing_session_json(tmp_path: Path) -> None:
    """A ZIP without session.json raises DemoLoadError."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("other.txt", "data")
    path = tmp_path / "bad.zip"
    path.write_bytes(buf.getvalue())
    with pytest.raises(DemoLoadError, match="session.json"):
        load_demo(path)


def test_load_zip_sha256_mismatch(tmp_path: Path) -> None:
    """A ZIP where file bytes don't match declared sha256 raises DemoLoadError."""
    fid = "00000000-0000-0000-0000-000000000002"
    entry, _content = _make_file_entry(fid, b"original")
    # Put different bytes in the ZIP
    zip_bytes = _make_zip(_minimal_json(files=[entry]), uploaded={fid: b"tampered"})
    path = tmp_path / "bad.zip"
    path.write_bytes(zip_bytes)
    with pytest.raises(DemoLoadError, match="sha256 mismatch"):
        load_demo(path)


def test_load_zip_byte_size_mismatch(tmp_path: Path) -> None:
    """A ZIP where file bytes don't match declared byte_size raises DemoLoadError."""
    fid = "00000000-0000-0000-0000-000000000003"
    content = b"hello"
    entry, _ = _make_file_entry(fid, content)
    # Tamper with byte_size but keep correct sha256 from the tampered content
    tampered = b"hello world"
    entry["sha256"] = hashlib.sha256(tampered).hexdigest()
    zip_bytes = _make_zip(_minimal_json(files=[entry]), uploaded={fid: tampered})
    path = tmp_path / "bad.zip"
    path.write_bytes(zip_bytes)
    with pytest.raises(DemoLoadError, match="byte_size mismatch"):
        load_demo(path)


def test_load_zip_undeclared_uploaded_entry(tmp_path: Path) -> None:
    """A ZIP with an uploaded/ entry not in files raises DemoLoadError."""
    session_json = _minimal_json(files=[])
    zip_bytes = _make_zip(session_json, uploaded={"rogue-id": b"data"})
    path = tmp_path / "bad.zip"
    path.write_bytes(zip_bytes)
    with pytest.raises(DemoLoadError, match="do not match"):
        load_demo(path)


def test_load_zip_missing_uploaded_entry(tmp_path: Path) -> None:
    """A ZIP declaring a file but missing its uploaded/ entry raises DemoLoadError."""
    fid = "00000000-0000-0000-0000-000000000004"
    entry, _ = _make_file_entry(fid, b"content")
    zip_bytes = _make_zip(_minimal_json(files=[entry]), uploaded={})
    path = tmp_path / "bad.zip"
    path.write_bytes(zip_bytes)
    with pytest.raises(DemoLoadError, match="missing ZIP entry"):
        load_demo(path)


# ---------------------------------------------------------------------------
# load_demo — bundled resource
# ---------------------------------------------------------------------------


def test_load_bundled_demo() -> None:
    """The committed demo-session.json resource loads without error."""
    demo = load_demo()
    assert demo.session_id == UUID(DEMO_SESSION_ID)
    assert len(demo.turns) >= 1
    for user_msg, assistant_msg in demo.turns:
        assert user_msg.role is ConversationRole.USER
        assert assistant_msg.role is ConversationRole.ASSISTANT


# ---------------------------------------------------------------------------
# seed_demo — in-memory stores
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_seed_demo_fresh_inmemory() -> None:
    """Seeding into fresh in-memory stores succeeds."""
    demo = load_demo()
    session_store = InMemorySessionStore()
    report_store = InMemoryReportStore(max_reports_per_session=10)

    seeded = await seed_demo(demo, session_store, report_store)
    assert seeded is True

    # Session is accessible
    session = await session_store.get(demo.session_id)
    assert session.session_id == demo.session_id
    assert session.retained_turns == len(demo.turns)


@pytest.mark.anyio
async def test_seed_demo_idempotent_inmemory() -> None:
    """Seeding the same demo twice returns False the second time."""
    demo = load_demo()
    session_store = InMemorySessionStore()
    report_store = InMemoryReportStore(max_reports_per_session=10)

    first = await seed_demo(demo, session_store, report_store)
    second = await seed_demo(demo, session_store, report_store)
    assert first is True
    assert second is False


@pytest.mark.anyio
async def test_seed_demo_preserves_messages() -> None:
    """Seeded session contains the expected conversation messages."""
    demo = load_demo()
    session_store = InMemorySessionStore()
    report_store = InMemoryReportStore(max_reports_per_session=10)

    await seed_demo(demo, session_store, report_store)
    session = await session_store.get(demo.session_id)

    assert len(session.messages) == len(demo.turns) * 2
    for index, (user_msg, assistant_msg) in enumerate(demo.turns):
        assert session.messages[index * 2].content == user_msg.content
        assert session.messages[index * 2 + 1].content == assistant_msg.content


@pytest.mark.anyio
async def test_seed_demo_with_files(tmp_path: Path) -> None:
    """Seeding a demo with files inserts them into the report store."""
    fid = "00000000-0000-0000-0000-000000000001"
    raw_content = b"Demo report content"
    entry, content = _make_file_entry(fid, raw_content)

    session_json = _minimal_json(files=[entry])
    zip_bytes = _make_zip(session_json, uploaded={fid: content})
    path = tmp_path / "demo.zip"
    path.write_bytes(zip_bytes)

    demo = load_demo(path)
    session_store = InMemorySessionStore()
    report_store = InMemoryReportStore(max_reports_per_session=10)

    seeded = await seed_demo(demo, session_store, report_store)
    assert seeded is True

    # Reports are accessible
    reports = await report_store.list_for_session(demo.session_id)
    assert len(reports) == 1
    assert reports[0].file_id == UUID(fid)

    # Resolve the report
    resolved = await report_store.resolve(demo.session_id, [UUID(fid)])
    assert resolved[0].original_bytes == raw_content


@pytest.mark.anyio
async def test_seed_demo_file_id_collision(tmp_path: Path) -> None:
    """Seeding fails with DemoLoadError when a file_id already exists."""
    fid = "00000000-0000-0000-0000-000000000001"
    raw_content = b"Content A"
    entry, content = _make_file_entry(fid, raw_content)

    session_json = _minimal_json(files=[entry])
    zip_bytes = _make_zip(session_json, uploaded={fid: content})
    path = tmp_path / "demo.zip"
    path.write_bytes(zip_bytes)

    demo = load_demo(path)
    session_store = InMemorySessionStore()
    report_store = InMemoryReportStore(max_reports_per_session=10)

    # Pre-populate a report with the same file_id under a different session
    other_session = await session_store.create()
    existing_report = StoredReport(
        metadata=ReportUploadResponse(
            file_id=UUID(fid),
            filename="existing.txt",
            media_type=ReportMediaType.TEXT,
            byte_size=9,
            extracted_text_length=9,
            sha256=hashlib.sha256(b"Content A").hexdigest(),
        ),
        original_bytes=b"Content A",
        extracted_text="Content A",
    )
    report_store.seed(other_session.session_id, [existing_report])

    from agentic_eval_toyapp.reports import ReportCapacityError
    from agentic_eval_toyapp.sessions import SessionNotFoundError

    with pytest.raises(ReportCapacityError, match="collision"):
        await seed_demo(demo, session_store, report_store)

    # Atomicity: the session should have been rolled back
    with pytest.raises(SessionNotFoundError):
        await session_store.get(demo.session_id)
