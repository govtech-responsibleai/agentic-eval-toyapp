"""Integration tests for the seeded demonstration session."""

from collections.abc import Generator, Sequence
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


def test_demo_session_is_accessible_via_api(demo_client: Client) -> None:
    """The seeded demo session can be retrieved through the REST API."""
    demo = load_demos()[0]

    response: Response = demo_client.get(f"/api/v1/sessions/{demo.session_id}")
    assert response.status_code == 200

    payload = response.json()
    assert payload["session_id"] == str(demo.session_id)
    assert payload["retained_turns"] == len(demo.turns)
    assert len(payload["messages"]) == len(demo.turns) * 2


def test_demo_session_url_renders_html(demo_client: Client) -> None:
    """The stable demo URL /session/<id> renders the HTML shell."""
    demo = load_demos()[0]

    response: Response = demo_client.get(f"/session/{demo.session_id}")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert str(demo.session_id) in response.text


def test_demo_button_data_attribute_present(demo_client: Client) -> None:
    """The index page includes the demo session ID as a data attribute."""
    demo = load_demos()[0]

    response: Response = demo_client.get("/")
    assert response.status_code == 200
    assert str(demo.session_id) in response.text
    assert "data-demos=" in response.text


def test_demo_session_messages_match_resource(demo_client: Client) -> None:
    """The seeded session messages match the bundled resource exactly."""
    demo = load_demos()[0]

    response: Response = demo_client.get(f"/api/v1/sessions/{demo.session_id}")
    payload = response.json()

    for index, (user_msg, assistant_msg) in enumerate(demo.turns):
        assert payload["messages"][index * 2]["content"] == user_msg.content
        assert payload["messages"][index * 2 + 1]["content"] == assistant_msg.content


def test_demo_session_export_works(demo_client: Client) -> None:
    """The demo session can be exported through the export endpoint."""
    demo = load_demos()[0]

    response: Response = demo_client.get(f"/api/v1/sessions/{demo.session_id}/export")
    assert response.status_code == 200
    # Demo has no files, so it should be JSON
    assert response.headers["content-type"] == "application/json"
    payload = response.json()
    assert payload["session_id"] == str(demo.session_id)
