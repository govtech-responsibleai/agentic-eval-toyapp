"""Integration tests for read-only session enforcement."""

from collections.abc import Generator, Sequence
from io import BytesIO
from typing import cast

import pytest
from httpx2 import Client, Response
from starlette.testclient import TestClient

from agentic_eval_toyapp.app import create_app
from agentic_eval_toyapp.config import AppSettings
from agentic_eval_toyapp.demo import load_demos
from agentic_eval_toyapp.models import (
    AgentTurnResult,
    AgentTurnStatus,
    ClarifyAction,
    ConversationMessage,
    TerminationReason,
)
from agentic_eval_toyapp.reports import InMemoryReportStore, StoredReport
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


@pytest.fixture
def demo_client() -> Generator[Client, None, None]:
    """Yield a test client whose lifespan has run (demo is seeded)."""
    settings = AppSettings()
    application = create_app(
        settings,
        session_store=InMemorySessionStore(),
        report_store=InMemoryReportStore(max_reports_per_session=10),
        agent=_StubAgent(),
    )
    with TestClient(application) as raw_client:
        yield cast(Client, raw_client)


def test_demo_session_is_read_only(demo_client: Client) -> None:
    """The seeded demo session reports read_only=True in the API response."""
    demo = load_demos()[0]
    response: Response = demo_client.get(f"/api/v1/sessions/{demo.session_id}")
    assert response.status_code == 200
    assert response.json()["read_only"] is True


def test_new_session_is_mutable(demo_client: Client) -> None:
    """A newly created session is mutable (read_only=False)."""
    response: Response = demo_client.post("/api/v1/sessions")
    assert response.status_code == 201
    assert response.json()["read_only"] is False


def test_message_rejected_on_read_only_session(demo_client: Client) -> None:
    """Sending a message to a read-only session returns 409."""
    demo = load_demos()[0]
    response: Response = demo_client.post(
        f"/api/v1/sessions/{demo.session_id}/messages",
        json={"message": "test message"},
    )
    assert response.status_code == 409
    assert response.json()["code"] == "session_read_only"


def test_upload_rejected_on_read_only_session(demo_client: Client) -> None:
    """Uploading a file to a read-only session returns 409."""
    demo = load_demos()[0]
    response: Response = demo_client.post(
        f"/api/v1/sessions/{demo.session_id}/files",
        files={"file": ("test.txt", BytesIO(b"hello"), "text/plain")},
    )
    assert response.status_code == 409
    assert response.json()["code"] == "session_read_only"


def test_delete_rejected_on_read_only_session(demo_client: Client) -> None:
    """Deleting a file from a read-only session returns 409."""
    demo = load_demos()[0]
    fake_file_id = "00000000-0000-0000-0000-000000000099"
    response: Response = demo_client.delete(
        f"/api/v1/sessions/{demo.session_id}/files/{fake_file_id}",
    )
    assert response.status_code == 409
    assert response.json()["code"] == "session_read_only"


def test_get_still_works_on_read_only_session(demo_client: Client) -> None:
    """GET on a read-only session still succeeds."""
    demo = load_demos()[0]
    response: Response = demo_client.get(f"/api/v1/sessions/{demo.session_id}")
    assert response.status_code == 200


def test_export_still_works_on_read_only_session(demo_client: Client) -> None:
    """Export on a read-only session still succeeds and excludes read_only."""
    demo = load_demos()[0]
    response: Response = demo_client.get(f"/api/v1/sessions/{demo.session_id}/export")
    assert response.status_code == 200
    payload = response.json()
    assert "read_only" not in payload


def test_mutable_session_accepts_message(demo_client: Client) -> None:
    """A mutable session can receive messages normally."""
    session_response: Response = demo_client.post("/api/v1/sessions")
    session_id = session_response.json()["session_id"]
    response: Response = demo_client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"message": "test message"},
    )
    assert response.status_code == 200
