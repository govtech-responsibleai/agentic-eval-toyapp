"""Integration tests for the session export API endpoint."""

import io
import json
import zipfile
from collections.abc import Sequence
from typing import cast
from uuid import uuid4

from httpx2 import Client, Response
from starlette.testclient import TestClient

from agentic_eval_toyapp.app import create_app
from agentic_eval_toyapp.config import AppSettings
from agentic_eval_toyapp.models import (
    AgentTurnResult,
    AgentTurnStatus,
    ClarifyAction,
    ConversationMessage,
    TerminationReason,
)
from agentic_eval_toyapp.reports import InMemoryReportStore, StoredReport
from agentic_eval_toyapp.session_export import FORMAT_VERSION
from agentic_eval_toyapp.sessions import InMemorySessionStore


class _StubAgent:
    """Minimal agent that returns a clarification for every message."""

    async def run(
        self,
        user_message: str,
        *,
        history: Sequence[ConversationMessage] = (),
        session_id: str | None = None,
        reports: Sequence[StoredReport] = (),
    ) -> AgentTurnResult:
        del history, session_id, reports
        question = f"Clarify: {user_message}"
        return AgentTurnResult(
            status=AgentTurnStatus.CLARIFICATION_REQUIRED,
            message=question,
            termination_reason=TerminationReason.CLARIFICATION_REQUIRED,
            model_calls=1,
            searches=0,
            model_actions=(ClarifyAction(question=question),),
        )


def _build_client() -> Client:
    settings = AppSettings()
    application = create_app(
        settings,
        session_store=InMemorySessionStore(),
        report_store=InMemoryReportStore(max_reports_per_session=10),
        agent=_StubAgent(),
    )
    return cast(Client, TestClient(application))


def test_export_empty_session_returns_json() -> None:
    """Exporting a session with no files returns JSON with safe headers."""
    client = _build_client()
    created: Response = client.post("/api/v1/sessions")
    assert created.status_code == 201
    session_id = created.json()["session_id"]

    export: Response = client.get(f"/api/v1/sessions/{session_id}/export")
    assert export.status_code == 200
    assert export.headers["content-type"] == "application/json"
    assert "attachment" in export.headers["content-disposition"]
    assert f"session-{session_id}.json" in export.headers["content-disposition"]
    assert export.headers["x-content-type-options"] == "nosniff"

    payload = export.json()
    assert payload["format_version"] == FORMAT_VERSION
    assert payload["session_id"] == session_id
    assert payload["turns"] == []
    assert payload["files"] == []


def test_export_session_with_messages_includes_turns() -> None:
    """Exporting a session with conversation turns includes them in the JSON."""
    client = _build_client()
    created: Response = client.post("/api/v1/sessions")
    session_id = created.json()["session_id"]

    client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"message": "What is the leave policy?"},
    )

    export: Response = client.get(f"/api/v1/sessions/{session_id}/export")
    assert export.status_code == 200
    payload = export.json()
    assert len(payload["turns"]) == 2
    assert payload["turns"][0]["role"] == "user"
    assert payload["turns"][1]["role"] == "assistant"


def test_export_session_with_file_returns_zip() -> None:
    """Exporting a session with files returns a ZIP archive."""
    client = _build_client()
    created: Response = client.post("/api/v1/sessions")
    session_id = created.json()["session_id"]

    content = b"# Analysis\nKey finding."
    uploaded: Response = client.post(
        f"/api/v1/sessions/{session_id}/files",
        files={"file": ("report.md", content, "text/markdown")},
    )
    assert uploaded.status_code == 201
    file_id = uploaded.json()["file_id"]

    export: Response = client.get(f"/api/v1/sessions/{session_id}/export")
    assert export.status_code == 200
    assert export.headers["content-type"] == "application/zip"
    assert f"session-{session_id}.zip" in export.headers["content-disposition"]

    with zipfile.ZipFile(io.BytesIO(export.content)) as zf:
        names = set(zf.namelist())
        assert "session.json" in names
        assert f"uploaded/{file_id}" in names

        session_json = json.loads(zf.read("session.json"))
        assert session_json["format_version"] == FORMAT_VERSION
        assert len(session_json["files"]) == 1
        assert session_json["files"][0]["file_id"] == file_id
        assert session_json["files"][0]["filename"] == "report.md"

        raw_bytes = zf.read(f"uploaded/{file_id}")
        assert raw_bytes == content


def test_export_missing_session_returns_404() -> None:
    """Exporting a nonexistent session returns 404."""
    client = _build_client()
    missing = str(uuid4())
    export: Response = client.get(f"/api/v1/sessions/{missing}/export")
    assert export.status_code == 404
    assert export.json()["code"] == "session_not_found"


def test_export_zip_entries_match_declared_files() -> None:
    """ZIP uploaded/ entries must match the declared files exactly."""
    client = _build_client()
    created: Response = client.post("/api/v1/sessions")
    session_id = created.json()["session_id"]

    first = client.post(
        f"/api/v1/sessions/{session_id}/files",
        files={"file": ("a.txt", b"AAA", "text/plain")},
    ).json()
    second = client.post(
        f"/api/v1/sessions/{session_id}/files",
        files={"file": ("b.txt", b"BBB", "text/plain")},
    ).json()

    export: Response = client.get(f"/api/v1/sessions/{session_id}/export")
    assert export.status_code == 200

    with zipfile.ZipFile(io.BytesIO(export.content)) as zf:
        session_json = json.loads(zf.read("session.json"))
        declared_ids = {f["file_id"] for f in session_json["files"]}
        zip_uploaded = {
            name.removeprefix("uploaded/")
            for name in zf.namelist()
            if name.startswith("uploaded/")
        }
        assert declared_ids == zip_uploaded
        assert declared_ids == {first["file_id"], second["file_id"]}


def test_export_does_not_include_langfuse_or_read_only() -> None:
    """Exported data must not contain Langfuse traces or read_only fields."""
    client = _build_client()
    created: Response = client.post("/api/v1/sessions")
    session_id = created.json()["session_id"]

    export: Response = client.get(f"/api/v1/sessions/{session_id}/export")
    raw_text = export.text
    assert "langfuse" not in raw_text.lower()
    assert "read_only" not in raw_text
    assert "trace" not in raw_text.lower()
