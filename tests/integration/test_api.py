"""End-to-end tests for the request/response versioned chat API."""

import hashlib
import logging
from collections.abc import Callable, Sequence
from datetime import date
from typing import cast
from uuid import uuid4

import pytest
from anyio import CapacityLimiter
from httpx2 import Client, Response
from starlette.testclient import TestClient

from agentic_eval_toyapp.agent import RagAgent, RagSpecialistAdapter
from agentic_eval_toyapp.analysis import (
    DeterministicAnalysisGenerationProvider,
    ReportAnalysisAgent,
    load_analysis_prompt,
)
from agentic_eval_toyapp.app import AgentInitialisationError, create_app
from agentic_eval_toyapp.config import AgentMode, AppSettings, GenerationProviderKind
from agentic_eval_toyapp.coordinator import (
    CoordinatorAgent,
    DeterministicCoordinatorGenerationProvider,
    ScriptedCoordinatorGenerationProvider,
    load_coordinator_prompt,
)
from agentic_eval_toyapp.models import (
    AgentAction,
    AgentActionKind,
    AgentTurnResult,
    AgentTurnStatus,
    AnswerAction,
    ClarifyAction,
    ConversationMessage,
    DocumentMetadata,
    ModelRequest,
    RetrievedChunk,
    SearchAction,
    TerminationReason,
)
from agentic_eval_toyapp.orchestration_models import CoordinatorRouteRagAction
from agentic_eval_toyapp.providers import (
    ExtractiveGenerationProvider,
    GenerationProviderError,
    ScriptedGenerationProvider,
)
from agentic_eval_toyapp.reports import StoredReport
from agentic_eval_toyapp.retrieval import AsyncRetriever
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

    async def generate_action(self, request: ModelRequest) -> AgentAction:
        raise GenerationProviderError("private upstream transport detail")


class _CountingAgent:
    """Record turn invocations without depending on retrieval behavior."""

    def __init__(self) -> None:
        self.calls = 0
        self.report_ids: list[tuple[str, ...]] = []

    async def run(
        self,
        user_message: str,
        *,
        history: Sequence[ConversationMessage] = (),
        session_id: str | None = None,
        reports: Sequence[StoredReport] = (),
    ) -> AgentTurnResult:
        del history, session_id
        self.calls += 1
        self.report_ids.append(
            tuple(str(report.metadata.file_id) for report in reports)
        )
        question = f"Clarify: {user_message}"
        return AgentTurnResult(
            status=AgentTurnStatus.CLARIFICATION_REQUIRED,
            message=question,
            termination_reason=TerminationReason.CLARIFICATION_REQUIRED,
            model_calls=1,
            searches=0,
            model_actions=(ClarifyAction(question=question),),
        )


def _build_agent(retriever: _PolicyRetriever | None = None) -> RagAgent:
    return RagAgent(
        provider=ExtractiveGenerationProvider(),
        retriever=AsyncRetriever((retriever or _PolicyRetriever()), CapacityLimiter(1)),
        system_prompt="Search before answering and cite retrieved evidence.",
    )


def _build_coordinator(retriever: _PolicyRetriever | None = None) -> CoordinatorAgent:
    rag_agent = _build_agent(retriever)
    return CoordinatorAgent(
        provider=DeterministicCoordinatorGenerationProvider(),
        system_prompt=load_coordinator_prompt(),
        rag_specialist=RagSpecialistAdapter(rag_agent),
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
    """METIS should be able to complete the full request/response API workflow."""
    retriever = _PolicyRetriever()
    client = _build_client(agent=_build_agent(retriever))

    created: Response = client.post("/api/v1/sessions")
    assert created.status_code == 201
    session_id = created.json()["session_id"]
    assert created.json() == {
        "session_id": session_id,
        "messages": [],
        "retained_turns": 0,
        "read_only": False,
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
        "execution_steps",
        "report_references",
        "termination_reason",
        "model_calls",
        "searches",
    }
    assert "18 days" in payload["result"]["message"]
    assert retriever.queries == ["How much annual leave do employees receive?"]
    assert payload["result"]["execution_steps"] == []
    assert payload["result"]["report_references"] == []

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


def test_report_upload_is_separate_from_message_submission() -> None:
    """Uploading should retain metadata without running an agent or adding a turn."""
    agent = _CountingAgent()
    client = _build_client(agent=agent)
    session_id = client.post("/api/v1/sessions").json()["session_id"]
    content = b"# Quarterly report\n\nRevenue increased by 12%."

    uploaded: Response = client.post(
        f"/api/v1/sessions/{session_id}/files",
        files={"file": ("report.md", content, "text/markdown")},
    )

    assert uploaded.status_code == 201
    payload = uploaded.json()
    assert payload == {
        "file_id": payload["file_id"],
        "filename": "report.md",
        "media_type": "text/markdown",
        "byte_size": len(content),
        "extracted_text_length": len(content.decode()),
        "sha256": hashlib.sha256(content).hexdigest(),
        "page_count": None,
    }
    assert agent.calls == 0
    assert client.get(f"/api/v1/sessions/{session_id}").json() == {
        "session_id": session_id,
        "messages": [],
        "retained_turns": 0,
        "read_only": False,
    }

    answered = client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={
            "message": "How much annual leave do employees receive?",
            "file_ids": [payload["file_id"]],
        },
    )

    assert answered.status_code == 200
    assert answered.json()["result"]["status"] == "clarification_required"
    assert agent.calls == 1


def test_report_list_and_delete_are_session_scoped_and_do_not_run_agent() -> None:
    """A stored report is reusable until its owner explicitly deletes it."""
    agent = _CountingAgent()
    client = _build_client(agent=agent)
    owner = client.post("/api/v1/sessions").json()["session_id"]
    other = client.post("/api/v1/sessions").json()["session_id"]
    first = client.post(
        f"/api/v1/sessions/{owner}/files",
        files={"file": ("first.md", b"# First\nFinding A", "text/markdown")},
    ).json()
    second = client.post(
        f"/api/v1/sessions/{owner}/files",
        files={"file": ("second.txt", b"Finding B", "text/plain")},
    ).json()

    assert client.get(f"/api/v1/sessions/{owner}/files").json() == {
        "files": [first, second]
    }
    assert "Finding A" not in client.get(f"/api/v1/sessions/{owner}/files").text
    assert client.get(f"/api/v1/sessions/{other}/files").json() == {"files": []}
    assert (
        client.delete(f"/api/v1/sessions/{other}/files/{first['file_id']}").status_code
        == 404
    )
    assert (
        client.delete(f"/api/v1/sessions/{owner}/files/{uuid4()}").json()["code"]
        == "report_not_found"
    )
    deleted = client.delete(f"/api/v1/sessions/{owner}/files/{first['file_id']}")
    assert deleted.status_code == 204
    assert deleted.content == b""
    assert client.get(f"/api/v1/sessions/{owner}/files").json() == {"files": [second]}
    assert (
        client.post(
            f"/api/v1/sessions/{owner}/messages",
            json={"message": "Use the deleted report", "file_ids": [first["file_id"]]},
        ).status_code
        == 404
    )
    assert client.get(f"/api/v1/sessions/{owner}").json()["messages"] == []
    assert agent.calls == 0


def test_report_list_and_delete_reject_unknown_sessions() -> None:
    """File routes must not manufacture storage for an unknown session ID."""
    client = _build_client(agent=_CountingAgent())
    missing_session_id = uuid4()
    missing_file_id = uuid4()

    listed = client.get(f"/api/v1/sessions/{missing_session_id}/files")
    deleted = client.delete(
        f"/api/v1/sessions/{missing_session_id}/files/{missing_file_id}"
    )

    assert listed.status_code == 404
    assert listed.json()["code"] == "session_not_found"
    assert deleted.status_code == 404
    assert deleted.json()["code"] == "session_not_found"


def test_only_explicit_file_ids_reach_agent_on_each_message() -> None:
    """Upload and previous turns must not implicitly attach stored files."""
    agent = _CountingAgent()
    client = _build_client(agent=agent)
    session_id = client.post("/api/v1/sessions").json()["session_id"]
    first = client.post(
        f"/api/v1/sessions/{session_id}/files",
        files={"file": ("first.txt", b"First finding", "text/plain")},
    ).json()["file_id"]
    second = client.post(
        f"/api/v1/sessions/{session_id}/files",
        files={"file": ("second.txt", b"Second finding", "text/plain")},
    ).json()["file_id"]

    for file_ids in ([], [first], [first], [], [second]):
        response = client.post(
            f"/api/v1/sessions/{session_id}/messages",
            json={"message": "Analyse the selected report", "file_ids": file_ids},
        )
        assert response.status_code == 200

    assert agent.report_ids == [(), (first,), (first,), (), (second,)]


def test_deleting_used_report_preserves_prior_public_turn() -> None:
    """Deleting source storage cannot retroactively remove a completed answer."""
    agent = _CountingAgent()
    client = _build_client(agent=agent)
    session_id = client.post("/api/v1/sessions").json()["session_id"]
    file_id = client.post(
        f"/api/v1/sessions/{session_id}/files",
        files={"file": ("report.txt", b"Past finding", "text/plain")},
    ).json()["file_id"]
    client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"message": "Use this report", "file_ids": [file_id]},
    )
    assert (
        client.delete(f"/api/v1/sessions/{session_id}/files/{file_id}").status_code
        == 204
    )

    assert client.get(f"/api/v1/sessions/{session_id}").json()["retained_turns"] == 1
    assert client.get(f"/api/v1/sessions/{session_id}/files").json() == {"files": []}
    assert agent.report_ids == [(file_id,)]


def test_report_references_are_isolated_by_session(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Missing and cross-session IDs must fail before agent execution or history."""
    retriever = _PolicyRetriever()
    client = _build_client(agent=_build_agent(retriever))
    owner_id = client.post("/api/v1/sessions").json()["session_id"]
    other_id = client.post("/api/v1/sessions").json()["session_id"]
    uploaded = client.post(
        f"/api/v1/sessions/{owner_id}/files",
        files={"file": ("report.txt", b"Private finding", "text/plain")},
    ).json()

    with caplog.at_level(logging.WARNING, logger="uvicorn.error"):
        for session_id, file_id in (
            (other_id, uploaded["file_id"]),
            (owner_id, str(uuid4())),
        ):
            response: Response = client.post(
                f"/api/v1/sessions/{session_id}/messages",
                json={"message": "Use the report", "file_ids": [file_id]},
            )

            assert response.status_code == 404
            assert response.json()["code"] == "report_not_found"

    assert retriever.queries == []
    assert client.get(f"/api/v1/sessions/{owner_id}").json()["messages"] == []
    assert client.get(f"/api/v1/sessions/{other_id}").json()["messages"] == []
    assert caplog.text.count("code=report_not_found") == 2
    assert "Private finding" not in caplog.text


def test_report_endpoint_returns_stable_public_validation_errors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Unsupported, malformed, and oversized reports should expose safe errors."""
    settings = AppSettings(upload_max_bytes=8)
    application = create_app(
        settings,
        session_store=InMemorySessionStore(),
        agent=_build_agent(),
    )
    client = cast(Client, TestClient(application))
    session_id = client.post("/api/v1/sessions").json()["session_id"]

    cases = (
        (
            ("report.csv", b"a,b", "text/csv"),
            415,
            "unsupported_file_type",
        ),
        (
            ("report.pdf", b"%PDF-bad", "application/pdf"),
            422,
            "report_processing_failed",
        ),
        (
            ("report.txt", b"123456789", "text/plain"),
            413,
            "report_too_large",
        ),
    )

    with caplog.at_level(logging.WARNING, logger="uvicorn.error"):
        for file_value, status_code, code in cases:
            response = client.post(
                f"/api/v1/sessions/{session_id}/files",
                files={"file": file_value},
            )
            assert response.status_code == status_code
            assert response.json()["code"] == code
            assert set(response.json()) == {"code", "message"}

    failures = [
        record.message for record in caplog.records if "event=upload" in record.message
    ]
    assert len(failures) == len(cases)
    assert all(f"session_id={session_id}" in message for message in failures)
    assert all(
        f"code={code}" in message
        for message, (_, _, code) in zip(failures, cases, strict=True)
    )
    assert "123456789" not in caplog.text


def test_report_limits_and_invalid_session_fail_before_agent_execution(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Attachment-count and session validation should precede agent execution."""
    retriever = _PolicyRetriever()
    client = _build_client(agent=_build_agent(retriever))
    session_id = client.post("/api/v1/sessions").json()["session_id"]

    with caplog.at_level(logging.WARNING, logger="uvicorn.error"):
        too_many = client.post(
            f"/api/v1/sessions/{session_id}/messages",
            json={
                "message": "Use these reports",
                "file_ids": [str(uuid4()), str(uuid4())],
            },
        )
    duplicate = client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={
            "message": "Use this report",
            "file_ids": ["00000000-0000-0000-0000-000000000001"] * 2,
        },
    )
    unknown_session = client.post(
        f"/api/v1/sessions/{uuid4()}/files",
        files={"file": ("report.txt", b"Finding", "text/plain")},
    )

    assert too_many.status_code == 422
    assert too_many.json()["code"] == "too_many_report_files"
    assert duplicate.status_code == 422
    assert duplicate.json()["code"] == "invalid_request"
    assert unknown_session.status_code == 404
    assert unknown_session.json()["code"] == "session_not_found"
    assert retriever.queries == []
    assert client.get(f"/api/v1/sessions/{session_id}").json()["messages"] == []
    assert "code=too_many_report_files" in caplog.text


def test_opt_in_coordinator_mode_delegates_policy_turns_to_rag() -> None:
    """Coordinator mode should return the existing grounded RAG result."""
    retriever = _PolicyRetriever()
    application = create_app(
        AppSettings(
            agent_mode=AgentMode.COORDINATOR,
            generation_provider=GenerationProviderKind.DETERMINISTIC,
            langfuse_enabled=False,
        ),
        agent=_build_coordinator(retriever),
    )
    with TestClient(application) as raw_client:
        client = cast(Client, raw_client)
        session_id = client.post("/api/v1/sessions").json()["session_id"]

        direct = client.post(
            f"/api/v1/sessions/{session_id}/messages",
            json={"message": "Hello"},
        )
        policy = client.post(
            f"/api/v1/sessions/{session_id}/messages",
            json={"message": "How many annual leave days do employees receive?"},
        )
        uploaded = client.post(
            f"/api/v1/sessions/{session_id}/files",
            files={
                "file": (
                    "quarterly.md",
                    b"Revenue increased while supplier risk remained high.",
                    "text/markdown",
                )
            },
        ).json()
        analysis = client.post(
            f"/api/v1/sessions/{session_id}/messages",
            json={
                "message": "Summarise the main findings.",
                "file_ids": [uploaded["file_id"]],
            },
        )

    assert direct.status_code == 200
    assert direct.json()["result"]["status"] == "answered"
    assert [step["action"] for step in direct.json()["result"]["execution_steps"]] == [
        "inspect_input",
        "answer",
    ]
    policy_result = policy.json()["result"]
    assert policy_result["status"] == "answered"
    assert policy_result["termination_reason"] == "answered"
    assert policy_result["citations"] == ["LEAVE-001#1"]
    assert policy_result["model_calls"] == 3
    assert policy_result["searches"] == 1
    assert [
        (step["agent"], step["action"]) for step in policy_result["execution_steps"]
    ] == [
        ("coordinator", "inspect_input"),
        ("coordinator", "route_rag"),
        ("rag", "search"),
        ("rag", "answer"),
        ("coordinator", "return_response"),
    ]
    assert retriever.queries == ["How many annual leave days do employees receive?"]
    assert analysis.json()["result"]["termination_reason"] == ("specialist_unavailable")
    assert analysis.json()["result"]["execution_steps"][-1]["action"] == (
        "route_analysis"
    )


def test_coordinator_analysis_uses_uploaded_report_only_after_message_send() -> None:
    """The normal API should route a submitted report and return typed evidence."""
    coordinator = CoordinatorAgent(
        provider=DeterministicCoordinatorGenerationProvider(),
        system_prompt=load_coordinator_prompt(),
        analysis_specialist=ReportAnalysisAgent(
            provider=DeterministicAnalysisGenerationProvider(),
            system_prompt=load_analysis_prompt(),
        ),
    )
    client = _build_client(agent=coordinator)
    session_id = client.post("/api/v1/sessions").json()["session_id"]
    uploaded = client.post(
        f"/api/v1/sessions/{session_id}/files",
        files={
            "file": (
                "quarterly.md",
                b"# Findings\nRevenue rose 18%.\n# Limitation\nSingapore only.",
                "text/markdown",
            )
        },
    )
    assert uploaded.status_code == 201
    assert client.get(f"/api/v1/sessions/{session_id}").json()["messages"] == []

    response = client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={
            "message": "Summarise the main findings and limitation.",
            "file_ids": [uploaded.json()["file_id"]],
        },
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["status"] == "answered"
    assert result["model_calls"] == 2
    assert result["searches"] == 0
    assert result["citations"] == []
    assert result["retrieved_chunks"] == []
    assert {item["locator"]["section"] for item in result["report_references"]} == {
        "Findings",
        "Limitation",
    }
    assert [(step["agent"], step["action"]) for step in result["execution_steps"]] == [
        ("coordinator", "inspect_input"),
        ("coordinator", "route_analysis"),
        ("analysis", "inspect_content"),
        ("analysis", "answer"),
        ("coordinator", "return_response"),
    ]


def test_file_derived_policy_task_uses_only_corpus_evidence() -> None:
    """File text may define the task but only corpus chunks may become evidence."""
    retriever = _PolicyRetriever()
    application = create_app(
        AppSettings(
            agent_mode=AgentMode.COORDINATOR,
            generation_provider=GenerationProviderKind.DETERMINISTIC,
            langfuse_enabled=False,
        ),
        agent=_build_coordinator(retriever),
    )
    with TestClient(application) as raw_client:
        client = cast(Client, raw_client)
        session_id = client.post("/api/v1/sessions").json()["session_id"]
        uploaded = client.post(
            f"/api/v1/sessions/{session_id}/files",
            files={
                "file": (
                    "policy-question.md",
                    b"How many annual leave days do employees receive?",
                    "text/markdown",
                )
            },
        ).json()
        response = client.post(
            f"/api/v1/sessions/{session_id}/messages",
            json={
                "message": "Please answer the question in this file.",
                "file_ids": [uploaded["file_id"]],
            },
        )

    result = response.json()["result"]
    assert response.status_code == 200
    assert result["status"] == "answered"
    assert retriever.queries == ["How many annual leave days do employees receive?"]
    assert result["citations"] == ["LEAVE-001#1"]
    assert [chunk["chunk_id"] for chunk in result["retrieved_chunks"]] == [
        "LEAVE-001#1"
    ]
    assert all(
        chunk["text"] != "How many annual leave days do employees receive?"
        for chunk in result["retrieved_chunks"]
    )


def test_coordinator_rag_delegation_preserves_multi_turn_history() -> None:
    """A delegated follow-up should retain complete prior public turns."""
    chunk = _PolicyRetriever().search("fixture")[0]
    provider = ScriptedGenerationProvider(
        (
            SearchAction(query="annual leave"),
            AnswerAction(
                markdown="Annual leave policy. [LEAVE-001#1]",
                citations=("LEAVE-001#1",),
            ),
            SearchAction(query="annual leave eligibility"),
            AnswerAction(
                markdown="Provide supporting evidence. [LEAVE-001#1]",
                citations=("LEAVE-001#1",),
            ),
        )
    )
    rag_agent = RagAgent(
        provider=provider,
        retriever=AsyncRetriever(_PolicyRetriever(), CapacityLimiter(1)),
        system_prompt="Search before answering.",
    )
    coordinator = CoordinatorAgent(
        provider=DeterministicCoordinatorGenerationProvider(),
        system_prompt=load_coordinator_prompt(),
        rag_specialist=RagSpecialistAdapter(rag_agent),
    )
    client = _build_client(agent=coordinator)
    session_id = client.post("/api/v1/sessions").json()["session_id"]

    first = client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"message": "What is the annual leave policy?"},
    )
    second = client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"message": "What evidence do I need?"},
    )

    assert first.json()["result"]["retrieved_chunks"] == [chunk.model_dump(mode="json")]
    assert second.json()["result"]["status"] == "answered"
    assert [message.content for message in provider.requests[2].history] == [
        "What is the annual leave policy?",
        "Annual leave policy. [LEAVE-001#1]",
    ]


def test_coordinator_preserves_zero_evidence_rag_answer() -> None:
    """An empty search should return the RAG agent's observation-first answer."""
    rag_agent = RagAgent(
        provider=ExtractiveGenerationProvider(),
        retriever=AsyncRetriever(_EmptyRetriever(), CapacityLimiter(1)),
        system_prompt="Search before answering.",
    )
    coordinator = CoordinatorAgent(
        provider=ScriptedCoordinatorGenerationProvider(
            (CoordinatorRouteRagAction(task="Unsupported policy question"),)
        ),
        system_prompt=load_coordinator_prompt(),
        rag_specialist=RagSpecialistAdapter(rag_agent),
    )
    client = _build_client(agent=coordinator)
    session_id = client.post("/api/v1/sessions").json()["session_id"]

    response = client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"message": "Tell me about an unsupported policy."},
    )

    result = response.json()["result"]
    assert result["status"] == "answered"
    assert result["retrieved_chunks"] == []
    assert result["citations"] == []
    assert result["model_calls"] == 3
    assert result["searches"] == 1
    assert [step["action"] for step in result["execution_steps"]] == [
        "inspect_input",
        "route_rag",
        "search",
        "answer",
        "return_response",
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
        retriever=AsyncRetriever(_EmptyRetriever(), CapacityLimiter(1)),
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
        retriever=AsyncRetriever(_PolicyRetriever(), CapacityLimiter(1)),
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
        retriever=AsyncRetriever(_PolicyRetriever(), CapacityLimiter(1)),
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
        retriever=AsyncRetriever(_PolicyRetriever(), CapacityLimiter(1)),
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
    assert str(cause) == "invalid agent configuration"
