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
from agentic_eval_toyapp.config import AppSettings, GenerationProviderKind
from agentic_eval_toyapp.models import (
    AgentAction,
    AgentActionKind,
    AnswerAction,
    ClarifyAction,
    DocumentMetadata,
    ModelRequest,
    RetrievedChunk,
    SearchAction,
)
from agentic_eval_toyapp.providers import (
    ExtractiveGenerationProvider,
    GenerationProviderError,
    ScriptedGenerationProvider,
)
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


class _EmptyRetriever:
    def search(
        self, query: str, *, top_k: int | None = None
    ) -> tuple[RetrievedChunk, ...]:
        return ()


class _FailingGenerationProvider:
    """Model a transport failure without making an external request."""

    model_id = "failing-provider-fixture"
    revision = "v1"

    def generate_action(self, request: ModelRequest) -> AgentAction:
        raise GenerationProviderError("private upstream transport detail")


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
    assert payload["result"]["model_actions"] == [
        {
            "kind": "search",
            "query": "How much annual leave do employees receive?",
            "top_k": 3,
        },
        {
            "kind": "answer",
            "markdown": payload["result"]["message"],
            "citations": ["LEAVE-001#1"],
        },
    ]
    assert payload["result"]["validation_findings"] == []
    assert payload["result"]["controller_events"] == []
    assert set(payload["result"]) == {
        "status",
        "message",
        "citations",
        "retrieved_chunks",
        "action_summaries",
        "controller_events",
        "validation_findings",
        "model_actions",
        "termination_reason",
        "model_calls",
        "searches",
    }
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


def test_api_default_history_retains_the_latest_ten_complete_turns() -> None:
    """The documented default bound should evict one whole turn after eleven."""
    client = _build_client(agent=_build_agent())
    session_id = client.post("/api/v1/sessions").json()["session_id"]

    for turn_number in range(1, 12):
        response: Response = client.post(
            f"/api/v1/sessions/{session_id}/messages",
            json={"message": f"Question {turn_number}"},
        )
        assert response.status_code == 200

    snapshot = client.get(f"/api/v1/sessions/{session_id}").json()
    assert snapshot["retained_turns"] == 10
    assert len(snapshot["messages"]) == 20
    assert snapshot["messages"][0] == {
        "role": "user",
        "content": "Question 2",
    }
    assert snapshot["messages"][-2]["content"] == "Question 11"
    assert [message["role"] for message in snapshot["messages"]] == [
        role for _ in range(10) for role in ("user", "assistant")
    ]


def test_api_keeps_two_session_histories_isolated() -> None:
    """Session isolation must hold at the public HTTP boundary."""
    client = _build_client(agent=_build_agent())
    first_id = client.post("/api/v1/sessions").json()["session_id"]
    second_id = client.post("/api/v1/sessions").json()["session_id"]

    first_response: Response = client.post(
        f"/api/v1/sessions/{first_id}/messages",
        json={"message": "First session BLUE-ORCHID"},
    )
    second_response: Response = client.post(
        f"/api/v1/sessions/{second_id}/messages",
        json={"message": "Second session AMBER-COMET"},
    )

    assert first_response.status_code == second_response.status_code == 200
    first_messages = client.get(f"/api/v1/sessions/{first_id}").json()["messages"]
    second_messages = client.get(f"/api/v1/sessions/{second_id}").json()["messages"]
    assert first_messages[0]["content"] == "First session BLUE-ORCHID"
    assert second_messages[0]["content"] == "Second session AMBER-COMET"
    assert all("AMBER-COMET" not in message["content"] for message in first_messages)
    assert all("BLUE-ORCHID" not in message["content"] for message in second_messages)


def test_api_preserves_provider_response_after_empty_retrieval() -> None:
    """METIS should observe the response together with an empty evidence list."""
    agent = RagAgent(
        provider=ExtractiveGenerationProvider(),
        retriever=_EmptyRetriever(),
        system_prompt="Search before answering and expose empty evidence.",
    )
    client = _build_client(agent=agent)
    session_id = client.post("/api/v1/sessions").json()["session_id"]

    response: Response = client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"message": "What is the unknown policy value?"},
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["status"] == "answered"
    assert result["message"] == (
        "No relevant evidence was found in the knowledge base."
    )
    assert result["retrieved_chunks"] == []
    assert result["citations"] == []
    assert result["model_calls"] == 2
    assert result["searches"] == 1


def test_api_repeated_turns_allow_no_citations_for_irrelevant_candidates() -> None:
    """Repeated session turns must not turn irrelevant candidates into failures."""
    agent = RagAgent(
        provider=ScriptedGenerationProvider(
            (
                SearchAction(query="sick leave"),
                AnswerAction(
                    markdown="No relevant sick-leave evidence was found.",
                    citations=(),
                ),
                SearchAction(query="sick leave"),
                AnswerAction(
                    markdown="No relevant sick-leave evidence was found.",
                    citations=(),
                ),
            )
        ),
        retriever=_PolicyRetriever(),
        system_prompt="Treat retrieved chunks as candidates.",
    )
    client = _build_client(agent=agent)
    session_id = client.post("/api/v1/sessions").json()["session_id"]

    for _ in range(2):
        response: Response = client.post(
            f"/api/v1/sessions/{session_id}/messages",
            json={"message": "I need help with sick leave"},
        )

        assert response.status_code == 200
        result = response.json()["result"]
        assert result["status"] == "answered"
        assert result["termination_reason"] == "answered"
        assert result["citations"] == []
        assert len(result["retrieved_chunks"]) == 1
        assert result["retrieved_chunks"][0]["chunk_id"] == "LEAVE-001#1"
        assert result["model_calls"] == 2
        assert result["searches"] == 1


def test_api_recovers_after_clarification_and_repeated_search() -> None:
    """The reported clarification flow must not end in repeated-search failure."""
    provider = ScriptedGenerationProvider(
        (
            ClarifyAction(question="Which type of leave would you like to know about?"),
            SearchAction(query="sick leave"),
            SearchAction(query="SICK   LEAVE"),
            AnswerAction(
                markdown="No relevant sick-leave evidence was found.",
                citations=(),
            ),
        )
    )
    agent = RagAgent(
        provider=provider,
        retriever=_PolicyRetriever(),
        system_prompt="Treat retrieved chunks as candidates.",
    )
    client = _build_client(agent=agent)
    session_id = client.post("/api/v1/sessions").json()["session_id"]

    clarification: Response = client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"message": "tell me more about leaves"},
    )
    answer: Response = client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"message": "sick leaves"},
    )

    assert clarification.status_code == 200
    assert clarification.json()["result"]["status"] == "clarification_required"
    assert answer.status_code == 200
    result = answer.json()["result"]
    assert result["status"] == "answered"
    assert result["termination_reason"] == "answered"
    assert result["model_calls"] == 3
    assert result["searches"] == 1
    assert result["citations"] == []
    assert provider.requests[-1].available_actions == (
        AgentActionKind.CLARIFY,
        AgentActionKind.ANSWER,
    )
    assert provider.requests[-1].history[-1].content == (
        "Which type of leave would you like to know about?"
    )
    second_turn_first_request = provider.requests[1]
    assert second_turn_first_request.model_call_number == 1
    assert second_turn_first_request.remaining_searches == 2
    assert second_turn_first_request.retrieval_attempted is False
    assert second_turn_first_request.available_actions == tuple(AgentActionKind)


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


@pytest.mark.parametrize(
    "json_body",
    [
        {},
        {"message": 123},
        {"message": "Question", "unexpected": True},
        [],
        None,
    ],
)
def test_message_endpoint_rejects_invalid_json_shapes_without_partial_turn(
    json_body: object,
) -> None:
    """Missing, mistyped, extra, and non-object JSON must not mutate a session."""
    retriever = _PolicyRetriever()
    client = _build_client(agent=_build_agent(retriever))
    session_id = client.post("/api/v1/sessions").json()["session_id"]

    response: Response = client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json=json_body,
    )

    assert response.status_code == 422
    assert response.json() == {
        "code": "invalid_request",
        "message": "The request did not match the API schema.",
    }
    assert retriever.queries == []
    assert client.get(f"/api/v1/sessions/{session_id}").json()["messages"] == []


def test_message_endpoint_rejects_malformed_json_without_partial_turn() -> None:
    """Unreadable JSON must use the stable error contract and preserve history."""
    retriever = _PolicyRetriever()
    client = _build_client(agent=_build_agent(retriever))
    session_id = client.post("/api/v1/sessions").json()["session_id"]

    response: Response = client.post(
        f"/api/v1/sessions/{session_id}/messages",
        content=b'{"message":',
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422
    assert response.json() == {
        "code": "invalid_request",
        "message": "The request did not match the API schema.",
    }
    assert retriever.queries == []
    assert client.get(f"/api/v1/sessions/{session_id}").json()["messages"] == []


def test_api_returns_safe_typed_result_for_generation_provider_failure() -> None:
    """Provider transport details must not cross the public API boundary."""
    agent = RagAgent(
        provider=_FailingGenerationProvider(),
        retriever=_PolicyRetriever(),
        system_prompt="Choose one action.",
    )
    client = _build_client(agent=agent)
    session_id = client.post("/api/v1/sessions").json()["session_id"]

    response: Response = client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"message": "Question"},
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["status"] == "failed"
    assert result["termination_reason"] == "provider_error"
    assert result["message"] == "The generation provider could not produce an action."
    assert result["model_calls"] == 1
    assert result["model_actions"] == []
    assert "private upstream" not in response.text


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


def test_default_application_fails_without_live_provider_configuration() -> None:
    """Normal runtime must not silently fall back to deterministic generation."""
    application = create_app(
        AppSettings(
            generation_provider=GenerationProviderKind.OPENAI,
            openai_api_key=None,
            openai_base_url=None,
        )
    )

    with pytest.raises(AgentInitialisationError) as exc_info, TestClient(application):
        pass

    cause = exc_info.value.__cause__
    assert isinstance(cause, ValueError)
    assert "OPENAI_API_KEY and OPENAI_BASE_URL" in str(cause)
