"""Offline checks for one coordinator-owned, multi-agent Langfuse tree."""

import logging
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date
from typing import Any, cast
from uuid import uuid4

import pytest
from anyio import CapacityLimiter
from langfuse import Langfuse
from starlette.testclient import TestClient

import agentic_eval_toyapp.multi_agent_tracing as tracing_module
from agentic_eval_toyapp.agent import RagAgent, RagSpecialistAdapter
from agentic_eval_toyapp.analysis import (
    DeterministicAnalysisGenerationProvider,
    ReportAnalysisAgent,
)
from agentic_eval_toyapp.app import create_app
from agentic_eval_toyapp.config import AppSettings, GenerationProviderKind
from agentic_eval_toyapp.coordinator import (
    CoordinatorAgent,
    ScriptedCoordinatorGenerationProvider,
)
from agentic_eval_toyapp.diagnostics import CompletionDiagnostics
from agentic_eval_toyapp.models import (
    AgentTurnStatus,
    AnswerAction,
    DocumentMetadata,
    ReportMediaType,
    ReportUploadResponse,
    RetrievedChunk,
    SearchAction,
)
from agentic_eval_toyapp.multi_agent_tracing import (
    LangfuseCoordinatorTraceRecorder,
    record_completion_usage,
)
from agentic_eval_toyapp.orchestration_models import (
    CoordinatorAnswerAction,
    CoordinatorRequest,
    CoordinatorRouteAnalysisAction,
    CoordinatorRouteRagAction,
)
from agentic_eval_toyapp.providers import (
    InvalidGenerationActionError,
    ScriptedGenerationProvider,
)
from agentic_eval_toyapp.reports import StoredReport
from agentic_eval_toyapp.retrieval import AsyncRetriever


def _updates() -> list[dict[str, object]]:
    return []


@dataclass
class _Observation:
    name: str
    kind: str
    input: object
    depth: int
    metadata: object | None
    model: str | None
    id: str = ""
    updates: list[dict[str, object]] = field(default_factory=_updates)

    def update(self, **values: object) -> None:
        self.updates.append(values)


class _Client:
    def __init__(self) -> None:
        self.observations: list[_Observation] = []
        self.depth = 0
        self.active: list[_Observation] = []
        self.flushes = 0
        self.shutdowns = 0

    @contextmanager
    def start_as_current_observation(
        self,
        *,
        name: str,
        as_type: str,
        input: object,
        metadata: object | None = None,
        model: str | None = None,
    ) -> Generator[_Observation]:
        observation = _Observation(name, as_type, input, self.depth, metadata, model)
        observation.id = f"{len(self.observations) + 1:016x}"
        self.observations.append(observation)
        self.active.append(observation)
        self.depth += 1
        try:
            yield observation
        finally:
            self.depth -= 1
            self.active.pop()

    def get_current_trace_id(self) -> str | None:
        return "a" * 32 if self.active else None

    def get_current_observation_id(self) -> str | None:
        return self.active[-1].id if self.active else None

    def flush(self) -> None:
        self.flushes += 1

    def shutdown(self) -> None:
        self.shutdowns += 1


class _Retriever:
    def search(
        self, query: str, *, top_k: int | None = None
    ) -> tuple[RetrievedChunk, ...]:
        del query, top_k
        return (
            RetrievedChunk(
                chunk_id="LEAVE-001#1",
                document_id="LEAVE-001",
                title="Leave Policy",
                text="Employees receive 18 days of annual leave.",
                metadata=DocumentMetadata(
                    topic="leave", effective_date=date(2026, 1, 1)
                ),
                similarity_score=0.91,
            ),
        )


def _report() -> StoredReport:
    text = "# Findings\nRevenue increased by 12%."
    return StoredReport(
        metadata=ReportUploadResponse(
            file_id=uuid4(),
            filename="findings.md",
            media_type=ReportMediaType.MARKDOWN,
            byte_size=len(text.encode()),
            extracted_text_length=len(text),
            sha256="0" * 64,
        ),
        original_bytes=text.encode(),
        extracted_text=text,
    )


def _recorder(
    client: _Client, *, include_report_text: bool = False
) -> LangfuseCoordinatorTraceRecorder:
    return LangfuseCoordinatorTraceRecorder(
        public_key="public",
        secret_key="secret",
        base_url="https://langfuse.example",
        environment="test",
        release="0.1.0",
        base_metadata={"corpus_sha256": "a" * 64},
        rag_model_id="scripted-rag-fixture",
        include_report_text=include_report_text,
        client=cast(Langfuse, client),
    )


@pytest.fixture(autouse=True)
def _fake_propagation(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[list[dict[str, object]]]:
    calls: list[dict[str, object]] = []

    @contextmanager
    def record(**values: object) -> Generator[None]:
        calls.append(values)
        yield

    monkeypatch.setattr(tracing_module, "propagate_attributes", record)
    yield calls


def _shape(client: _Client) -> list[tuple[str, str, int]]:
    return [(item.name, item.kind, item.depth) for item in client.observations]


def _updated(observation: _Observation, key: str) -> list[object]:
    return [update[key] for update in observation.updates if key in update]


@pytest.mark.anyio
async def test_direct_answer_has_one_output_bearing_root_and_no_delegation(
    _fake_propagation: list[dict[str, object]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = _Client()
    agent = CoordinatorAgent(
        provider=ScriptedCoordinatorGenerationProvider(
            (CoordinatorAnswerAction(markdown="Hello."),)
        ),
        system_prompt="Coordinate.",
        trace_recorder=_recorder(client),
    )

    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        result = await agent.run("Hello", session_id="session-123")

    assert result.status is AgentTurnStatus.ANSWERED
    assert _shape(client) == [
        ("agent-turn", "agent", 0),
        ("inspect-input", "span", 1),
        ("coordinator-action", "generation", 1),
    ]
    assert _updated(client.observations[0], "output") == ["Hello."]
    assert _fake_propagation[0]["session_id"] == "session-123"
    assert client.flushes == 1
    assert "event=started" in caplog.text
    assert "name=inspect-input" in caplog.text
    assert "event=flush_attempted" in caplog.text
    assert "trace_id=" + "a" * 32 in caplog.text


def test_upload_only_does_not_create_a_root_and_message_does() -> None:
    client = _Client()
    agent = CoordinatorAgent(
        provider=ScriptedCoordinatorGenerationProvider(
            (CoordinatorAnswerAction(markdown="Received."),)
        ),
        system_prompt="Coordinate.",
        trace_recorder=_recorder(client),
    )
    application = create_app(
        AppSettings(
            generation_provider=GenerationProviderKind.DETERMINISTIC,
            langfuse_enabled=False,
        ),
        agent=agent,
    )

    with TestClient(application) as api:
        session = api.post("/api/v1/sessions")
        session_id = session.json()["session_id"]
        upload = api.post(
            f"/api/v1/sessions/{session_id}/files",
            files={
                "file": ("findings.md", b"# Findings\nRevenue rose.", "text/markdown")
            },
        )
        assert upload.status_code == 201
        assert client.observations == []

        response = api.post(
            f"/api/v1/sessions/{session_id}/messages",
            json={
                "message": "Acknowledge this report",
                "file_ids": [upload.json()["file_id"]],
            },
        )
        assert response.status_code == 200

    assert [item.name for item in client.observations].count("agent-turn") == 1
    assert client.flushes == 1


@pytest.mark.anyio
async def test_failed_provider_attempt_is_a_classified_generation_not_a_second_root() -> (
    None
):
    class _RepairingProvider:
        model_id = "scripted-model"
        revision = "v1"

        def __init__(self) -> None:
            self.calls = 0

        async def generate_action(
            self, request: CoordinatorRequest
        ) -> CoordinatorAnswerAction:
            del request
            self.calls += 1
            if self.calls == 1:
                raise InvalidGenerationActionError("private malformed output")
            return CoordinatorAnswerAction(markdown="Recovered.")

    client = _Client()
    agent = CoordinatorAgent(
        provider=_RepairingProvider(),
        system_prompt="Coordinate.",
        trace_recorder=_recorder(client),
    )

    result = await agent.run("Hello", session_id="session-repair")

    assert result.status is AgentTurnStatus.ANSWERED
    assert [item.name for item in client.observations].count("agent-turn") == 1
    actions = [
        item for item in client.observations if item.name == "coordinator-action"
    ]
    assert len(actions) == 2
    assert _updated(actions[0], "output") == [{"error_code": "provider_error"}]
    assert (
        cast(dict[str, object], _updated(actions[1], "output")[0])["kind"] == "answer"
    )
    assert "private malformed output" not in str(client.observations)


@pytest.mark.anyio
async def test_rag_route_has_one_nested_specialist_and_search_tool() -> None:
    client = _Client()
    rag = RagAgent(
        provider=ScriptedGenerationProvider(
            (
                SearchAction(query="annual leave", top_k=3),
                AnswerAction(
                    markdown="Employees receive 18 days. [LEAVE-001#1]",
                    citations=("LEAVE-001#1",),
                ),
            )
        ),
        retriever=AsyncRetriever(_Retriever(), CapacityLimiter(1)),
        system_prompt="Answer policy questions.",
    )
    agent = CoordinatorAgent(
        provider=ScriptedCoordinatorGenerationProvider(
            (CoordinatorRouteRagAction(task="Explain annual leave"),)
        ),
        system_prompt="Coordinate.",
        rag_specialist=RagSpecialistAdapter(rag),
        trace_recorder=_recorder(client),
    )

    result = await agent.run("How much annual leave?", session_id="session-rag")

    assert result.status is AgentTurnStatus.ANSWERED
    assert _shape(client) == [
        ("agent-turn", "agent", 0),
        ("inspect-input", "span", 1),
        ("coordinator-action", "generation", 1),
        ("delegate-rag", "agent", 1),
        ("model-action", "generation", 2),
        ("search_knowledge_base", "tool", 2),
        ("model-action", "generation", 2),
        ("validate_answer", "evaluator", 2),
        ("return-response", "span", 1),
    ]
    assert _updated(client.observations[0], "output") == [result.message]
    assert _updated(client.observations[3], "output") == [result.message]
    assert client.observations[4].model == "scripted-rag-fixture"


@pytest.mark.anyio
@pytest.mark.parametrize("include_report_text", [False, True])
async def test_analysis_route_exports_only_selected_report_context_by_policy(
    include_report_text: bool,
) -> None:
    client = _Client()
    report = _report()
    agent = CoordinatorAgent(
        provider=ScriptedCoordinatorGenerationProvider(
            (
                CoordinatorRouteAnalysisAction(
                    task="Summarise findings",
                    file_ids=(report.metadata.file_id,),
                ),
            )
        ),
        system_prompt="Coordinate.",
        analysis_specialist=ReportAnalysisAgent(
            provider=DeterministicAnalysisGenerationProvider(),
            system_prompt="Analyse reports.",
        ),
        trace_recorder=_recorder(client, include_report_text=include_report_text),
    )

    result = await agent.run(
        "Summarise findings", session_id="session-analysis", reports=(report,)
    )

    assert result.status is AgentTurnStatus.ANSWERED
    assert _shape(client) == [
        ("agent-turn", "agent", 0),
        ("inspect-input", "span", 1),
        ("coordinator-action", "generation", 1),
        ("delegate-analysis", "agent", 1),
        ("inspect-content", "span", 2),
        ("analyse-report", "generation", 2),
        ("validate-report-references", "evaluator", 2),
        ("return-response", "span", 1),
    ]
    coordinator_input = cast(dict[str, Any], client.observations[2].input)
    analysis_input = cast(dict[str, Any], client.observations[5].input)
    if include_report_text:
        assert (
            coordinator_input["attachments"][0]["extracted_text"]
            == report.extracted_text
        )
        assert analysis_input["sources"][0]["text"] == report.extracted_text
    else:
        assert report.extracted_text not in str(client.observations)
    assert analysis_input["report_text_exported"] is include_report_text


@pytest.mark.anyio
async def test_provider_usage_is_attached_only_to_active_generation() -> None:
    client = _Client()
    recorder = _recorder(client)
    record_completion_usage(
        CompletionDiagnostics(input_tokens=1, output_tokens=2, total_tokens=3)
    )
    async with recorder.start_turn(
        session_id="session-usage", user_message="Hello", history_message_count=0
    ) as turn:
        with turn.model_action(
            CoordinatorRequest(
                system_prompt="Hidden", user_message="Hello", model_call_number=1
            ),
            model="live-model",
        ) as action:
            record_completion_usage(
                CompletionDiagnostics(
                    input_tokens=11,
                    output_tokens=7,
                    total_tokens=18,
                    reasoning_tokens=3,
                    finish_reason="stop",
                )
            )
            action.succeed(CoordinatorAnswerAction(markdown="Hi"))
    generation = client.observations[1]
    assert generation.model == "live-model"
    assert _updated(generation, "usage") == [{"input": 11, "output": 7, "total": 18}]
    assert _updated(generation, "metadata") == [
        {"finish_reason": "stop", "reasoning_tokens": 3}
    ]
    assert "Hidden" not in str(client.observations)
