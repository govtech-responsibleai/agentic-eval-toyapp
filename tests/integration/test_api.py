"""End-to-end tests for the synchronous versioned chat API."""

from collections.abc import Callable
from datetime import date
from typing import cast
from uuid import uuid4

import pytest
from httpx2 import Client, Response
from starlette.testclient import TestClient

from agentic_eval_toyapp.agent import RagAgent
from agentic_eval_toyapp.app import AgentInitialisationError, create_app
from agentic_eval_toyapp.config import AppSettings
from agentic_eval_toyapp.models import DocumentMetadata, RetrievedChunk
from agentic_eval_toyapp.providers import ExtractiveGenerationProvider
from agentic_eval_toyapp.sessions import AgentRunner, InMemorySessionStore


class _PolicyRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(
        self, query: str, *, top_k: int | None = None
    ) -> tuple[RetrievedChunk, ...]:
        self.queries.append(query)
        return (
            RetrievedChunk(
                chunk_id="LEAVE-001#1",
                document_id="LEAVE-001",
                title="Annual Leave Policy",
                text="Employees receive 18 days of annual leave each year.",
                metadata=DocumentMetadata(
                    topic="leave",
                    effective_date=date(2026, 1, 1),
                ),
                similarity_score=0.95,
            ),
        )


def _build_agent(retriever: _PolicyRetriever | None = None) -> RagAgent:
    return RagAgent(
        provider=ExtractiveGenerationProvider(),
        retriever=retriever or _PolicyRetriever(),
        system_prompt="Search before answering and cite retrieved evidence.",
    )


def _build_client(
    *,
    agent: AgentRunner | None = None,
    agent_factory: Callable[[], AgentRunner] | None = None,
    max_turns: int = 10,
) -> Client:
    settings = AppSettings(session_max_turns=max_turns)
    application = create_app(
        settings,
        session_store=InMemorySessionStore(max_turns=max_turns),
        agent=agent,
        agent_factory=agent_factory,
    )
    return cast(Client, TestClient(application))


def test_session_message_and_snapshot_flow_uses_the_rag_agent() -> None:
    """METIS should be able to complete the full synchronous API workflow."""
    retriever = _PolicyRetriever()
    client = _build_client(agent=_build_agent(retriever))

    created: Response = client.post("/api/v1/sessions")
    assert created.status_code == 201
    session_id = created.json()["session_id"]
    assert created.json() == {
        "session_id": session_id,
        "messages": [],
        "retained_turns": 0,
    }

    answered: Response = client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"message": "  How much annual leave do employees receive?  "},
    )

    assert answered.status_code == 200
    payload = answered.json()
    assert payload["session_id"] == session_id
    assert payload["result"]["status"] == "answered"
    assert payload["result"]["termination_reason"] == "answered"
    assert payload["result"]["citations"] == ["LEAVE-001#1"]
    assert payload["result"]["model_calls"] == 2
    assert payload["result"]["searches"] == 1
    assert "18 days" in payload["result"]["message"]
    assert retriever.queries == ["How much annual leave do employees receive?"]

    snapshot: Response = client.get(f"/api/v1/sessions/{session_id}")
    assert snapshot.status_code == 200
    assert snapshot.json()["retained_turns"] == 1
    assert snapshot.json()["messages"] == [
        {
            "role": "user",
            "content": "How much annual leave do employees receive?",
        },
        {
            "role": "assistant",
            "content": payload["result"]["message"],
        },
    ]


def test_api_session_history_is_bounded_by_complete_turns() -> None:
    """API snapshots should retain whole recent turns at the configured bound."""
    client = _build_client(agent=_build_agent(), max_turns=1)
    session_id = client.post("/api/v1/sessions").json()["session_id"]

    for message in ("First question", "Second question"):
        response: Response = client.post(
            f"/api/v1/sessions/{session_id}/messages",
            json={"message": message},
        )
        assert response.status_code == 200

    snapshot = client.get(f"/api/v1/sessions/{session_id}").json()
    assert snapshot["retained_turns"] == 1
    assert [message["content"] for message in snapshot["messages"]][0] == (
        "Second question"
    )
    assert [message["role"] for message in snapshot["messages"]] == [
        "user",
        "assistant",
    ]


@pytest.mark.parametrize(
    ("method", "path", "json_body", "status_code", "code"),
    [
        ("get", "/api/v1/sessions/not-a-uuid", None, 422, "invalid_request"),
        (
            "get",
            f"/api/v1/sessions/{uuid4()}",
            None,
            404,
            "session_not_found",
        ),
        (
            "post",
            f"/api/v1/sessions/{uuid4()}/messages",
            {"message": "Question"},
            404,
            "session_not_found",
        ),
    ],
)
def test_api_returns_stable_errors_for_invalid_or_unknown_sessions(
    method: str,
    path: str,
    json_body: dict[str, str] | None,
    status_code: int,
    code: str,
) -> None:
    """Session failures should use the public error contract without internals."""
    client = _build_client(agent=_build_agent())

    response: Response = client.request(method, path, json=json_body)

    assert response.status_code == status_code
    assert response.json()["code"] == code
    assert set(response.json()) == {"code", "message"}


@pytest.mark.parametrize("message", ["", "   ", "x" * 8_001])
def test_message_endpoint_rejects_blank_and_oversized_text(message: str) -> None:
    """Invalid text must fail before the agent or session history is changed."""
    retriever = _PolicyRetriever()
    client = _build_client(agent=_build_agent(retriever))
    session_id = client.post("/api/v1/sessions").json()["session_id"]

    response: Response = client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"message": message},
    )

    assert response.status_code == 422
    assert response.json()["code"] == "invalid_request"
    assert retriever.queries == []
    assert client.get(f"/api/v1/sessions/{session_id}").json()["messages"] == []


def test_application_startup_fails_when_agent_initialisation_fails() -> None:
    """The server must not accept requests with unavailable RAG dependencies."""

    def unavailable_factory() -> AgentRunner:
        raise RuntimeError("private model-cache failure")

    application = create_app(
        AppSettings(),
        agent_factory=unavailable_factory,
    )

    with (
        pytest.raises(
            AgentInitialisationError,
            match="agent dependencies could not be initialised",
        ),
        TestClient(application),
    ):
        pass
