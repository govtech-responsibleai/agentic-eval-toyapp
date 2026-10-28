"""Opt-in end-to-end Langfuse v4 export and retrieval compatibility test."""

import json
import logging
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace
from typing import cast

import pytest
from anyio import CapacityLimiter
from httpx2 import Client
from langfuse import Langfuse
from langfuse.api.commons.types.observation_v2 import ObservationV2
from starlette.testclient import TestClient

from agentic_eval_toyapp import __version__
from agentic_eval_toyapp.agent import RagAgent, RagSpecialistAdapter
from agentic_eval_toyapp.app import create_app
from agentic_eval_toyapp.config import AgentMode, AppSettings
from agentic_eval_toyapp.coordinator import (
    CoordinatorAgent,
    ScriptedCoordinatorGenerationProvider,
)
from agentic_eval_toyapp.models import (
    AnswerAction,
    DocumentMetadata,
    RetrievedChunk,
    SearchAction,
)
from agentic_eval_toyapp.multi_agent_tracing import LangfuseCoordinatorTraceRecorder
from agentic_eval_toyapp.orchestration_models import (
    CoordinatorAnswerAction,
    CoordinatorRouteRagAction,
)
from agentic_eval_toyapp.providers import ScriptedGenerationProvider
from agentic_eval_toyapp.retrieval import AsyncRetriever
from agentic_eval_toyapp.tracing import LangfuseTraceRecorder


class _LiveTestRetriever:
    def search(
        self, query: str, *, top_k: int | None = None
    ) -> tuple[RetrievedChunk, ...]:
        del query, top_k
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


def _decode_observation_io(value: object) -> object:
    """Decode v4 observation I/O while tolerating already-decoded values."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


@dataclass(frozen=True)
class _ExpectedTurn:
    message: str
    output: str
    observations: Counter[tuple[str, str]]
    parents: dict[str, str]


def _get_session_observations(
    client: Langfuse, *, session_id: str, started_at: datetime
) -> list[ObservationV2]:
    """Read every v4 observations page; a partial page is not a full trace."""
    observations: list[ObservationV2] = []
    cursor: str | None = None
    upper_bound = datetime.now(UTC) + timedelta(minutes=1)
    seen_cursors: set[str] = set()
    while True:
        page = client.api.observations.get_many(
            session_id=session_id,
            fields="core,basic,io,trace_context",
            from_start_time=started_at,
            to_start_time=upper_bound,
            limit=100,
            cursor=cursor,
        )
        observations.extend(page.data)
        cursor = page.meta.cursor
        if cursor is None:
            return observations
        if cursor in seen_cursors:
            raise AssertionError("Langfuse observations pagination repeated a cursor")
        seen_cursors.add(cursor)


def _trace_issues(
    observations: Sequence[ObservationV2],
    *,
    session_id: str,
    expected: Sequence[_ExpectedTurn],
) -> list[str]:
    """Compare actual trace trees with scripted execution, not just root counts."""
    issues: list[str] = []
    roots = [item for item in observations if item.is_root_observation]
    if len(roots) != len(expected):
        issues.append(f"expected {len(expected)} roots, found {len(roots)}")

    ids = [item.id for item in observations]
    if len(ids) != len(set(ids)):
        issues.append("duplicate observation IDs")

    by_trace: dict[str, list[ObservationV2]] = {}
    for item in observations:
        if item.trace_id is None:
            issues.append(f"observation {item.id} has no trace ID")
            continue
        by_trace.setdefault(item.trace_id, []).append(item)
        if item.session_id != session_id:
            issues.append(f"observation {item.id} has wrong session ID")

    expected_by_message = {turn.message: turn for turn in expected}
    seen_messages: set[str] = set()
    for root in roots:
        message = _decode_observation_io(root.input)
        if not isinstance(message, str) or message not in expected_by_message:
            issues.append(f"unexpected root {root.id}")
            continue
        if message in seen_messages:
            issues.append(f"duplicate root for {message!r}")
            continue
        seen_messages.add(message)
        turn = expected_by_message[message]
        if root.name != "agent-turn" or root.type != "AGENT":
            issues.append(f"unexpected root type for {message!r}")
        if _decode_observation_io(root.output) != turn.output:
            issues.append(f"wrong or missing root output for {message!r}")
        if root.trace_id is None:
            continue
        tree = by_trace[root.trace_id]
        if sum(item.is_root_observation is True for item in tree) != 1:
            issues.append(f"trace {root.trace_id} has multiple roots")
        actual = Counter((item.name or "", item.type) for item in tree)
        if actual != turn.observations:
            issues.append(
                f"trace {root.trace_id} observation mismatch: "
                f"missing={turn.observations - actual}, extra={actual - turn.observations}"
            )
        tree_ids = {item.id for item in tree}
        names_by_id = {item.id: item.name for item in tree}
        for item in tree:
            if item.id == root.id:
                continue
            if item.parent_observation_id not in tree_ids:
                issues.append(
                    f"trace {root.trace_id} has orphan {item.name}:{item.id} "
                    f"parent={item.parent_observation_id}"
                )
            elif (
                item.name in turn.parents
                and names_by_id[item.parent_observation_id] != turn.parents[item.name]
            ):
                issues.append(
                    f"trace {root.trace_id} has wrong parent for {item.name}:{item.id}"
                )

    for trace_id, tree in by_trace.items():
        if not any(item.is_root_observation for item in tree):
            issues.append(f"trace {trace_id} has {len(tree)} observations but no root")
    return issues


def _wait_for_complete_traces(
    client: Langfuse,
    *,
    session_id: str,
    started_at: datetime,
    expected: Sequence[_ExpectedTurn],
    app_events: Sequence[str] = (),
) -> list[ObservationV2]:
    """Allow for v4 ingestion lag, but fail if any expected span stays absent."""
    deadline = time.monotonic() + 180
    while True:
        observations = _get_session_observations(
            client, session_id=session_id, started_at=started_at
        )
        issues = _trace_issues(observations, session_id=session_id, expected=expected)
        if not issues:
            return observations
        if time.monotonic() >= deadline:
            observed = ", ".join(
                f"{item.name}:{item.id}@{item.trace_id}" for item in observations
            )
            pytest.fail(
                f"Langfuse trace incomplete for session {session_id}: "
                + "; ".join(issues)
                + f"; app_events={list(app_events)}; observed=[{observed}]"
            )
        time.sleep(2)


def _assert_no_otel_export_errors(
    records: Sequence[logging.LogRecord],
) -> None:
    """Fail before polling when the SDK reports an OTLP transport rejection."""
    errors = [
        record.getMessage()
        for record in records
        if record.name.startswith("opentelemetry.exporter.otlp")
        and record.levelno >= logging.ERROR
    ]
    assert not errors, f"Langfuse OTLP export failed: {errors}"


def _test_observation(
    name: str,
    *,
    observation_id: str,
    parent_id: str | None = None,
    trace_id: str = "trace-1",
    root: bool = False,
) -> ObservationV2:
    return ObservationV2(
        id=observation_id,
        trace_id=trace_id,
        start_time=datetime.now(UTC),
        project_id="test-project",
        parent_observation_id=parent_id,
        type="AGENT" if root else "SPAN",
        is_root_observation=root,
        name=name,
        session_id="test-session",
        input="Hello" if root else None,
        output="Hello." if root else None,
    )


def test_otel_export_error_is_reported_before_ingestion_poll(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.ERROR):
        logging.getLogger(
            "opentelemetry.exporter.otlp.proto.http.trace_exporter"
        ).error("Failed to export spans batch code: 500")
    with pytest.raises(AssertionError, match="Langfuse OTLP export failed"):
        _assert_no_otel_export_errors(caplog.records)


def test_trace_completeness_detects_missing_and_orphaned_observations() -> None:
    """An answer-bearing root alone is not sufficient for evaluation."""
    expected = (
        _ExpectedTurn(
            message="Hello",
            output="Hello.",
            observations=Counter(
                {("agent-turn", "AGENT"): 1, ("inspect-input", "SPAN"): 1}
            ),
            parents={"inspect-input": "agent-turn"},
        ),
    )
    root = _test_observation("agent-turn", observation_id="root", root=True)
    inspect = _test_observation(
        "inspect-input", observation_id="inspect", parent_id="root"
    )
    assert (
        _trace_issues([root, inspect], session_id="test-session", expected=expected)
        == []
    )

    missing = _trace_issues([root], session_id="test-session", expected=expected)
    assert any("inspect-input" in issue for issue in missing)

    orphan = _test_observation(
        "inspect-input",
        observation_id="orphan",
        parent_id="missing-root",
        trace_id="orphan-trace",
    )
    issues = _trace_issues(
        [root, inspect, orphan], session_id="test-session", expected=expected
    )
    assert any("no root" in issue for issue in issues)

    wrong_parent = _test_observation(
        "inspect-input", observation_id="inspect", parent_id="inspect"
    )
    issues = _trace_issues(
        [root, wrong_parent], session_id="test-session", expected=expected
    )
    assert any("wrong parent" in issue for issue in issues)


def test_trace_completeness_requires_every_submitted_turn_root() -> None:
    root = _test_observation("agent-turn", observation_id="root", root=True)
    expected = (
        _ExpectedTurn(
            message="Hello",
            output="Hello.",
            observations=Counter({("agent-turn", "AGENT"): 1}),
            parents={},
        ),
        _ExpectedTurn(
            message="Second message",
            output="Second reply",
            observations=Counter({("agent-turn", "AGENT"): 1}),
            parents={},
        ),
    )
    issues = _trace_issues([root], session_id="test-session", expected=expected)
    assert "expected 2 roots, found 1" in issues


def test_session_observation_fetch_reads_every_v4_page() -> None:
    root = _test_observation("agent-turn", observation_id="root", root=True)
    inspect = _test_observation(
        "inspect-input", observation_id="inspect", parent_id="root"
    )

    class _PagedApi:
        def __init__(self) -> None:
            self.cursors: list[object] = []

        def get_many(self, **values: object) -> SimpleNamespace:
            cursor = values["cursor"]
            self.cursors.append(cursor)
            return SimpleNamespace(
                data=[root] if cursor is None else [inspect],
                meta=SimpleNamespace(cursor="second-page" if cursor is None else None),
            )

    api = _PagedApi()
    client = cast(Langfuse, SimpleNamespace(api=SimpleNamespace(observations=api)))
    observations = _get_session_observations(
        client, session_id="test-session", started_at=datetime.now(UTC)
    )
    assert observations == [root, inspect]
    assert api.cursors == [None, "second-page"]


@pytest.mark.langfuse_live
def test_api_turn_is_retrievable_by_app_session_id(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Use the same session query contract consumed by METIS's extractor."""
    settings = AppSettings(agent_mode=AgentMode.RAG)
    if not settings.langfuse_enabled:
        pytest.fail("TOYAPP_LANGFUSE_ENABLED must be true for the live trace test")
    public_key, secret_key, base_url = settings.require_langfuse_configuration()
    recorder = LangfuseTraceRecorder(
        public_key=public_key,
        secret_key=secret_key,
        base_url=base_url,
        environment=settings.langfuse_environment,
        release=__version__,
        base_metadata={"test_kind": "langfuse_compatibility"},
    )
    search = SearchAction(query="annual leave entitlement", top_k=3)
    answer = AnswerAction(
        markdown="Employees receive 18 days of annual leave. [LEAVE-001#1]",
        citations=("LEAVE-001#1",),
    )
    agent = RagAgent(
        provider=ScriptedGenerationProvider((search, answer)),
        retriever=AsyncRetriever(_LiveTestRetriever(), CapacityLimiter(1)),
        system_prompt="Search the policy corpus before answering.",
        trace_recorder=recorder,
    )
    application = create_app(settings, agent=agent)
    query_started_at = datetime.now(UTC) - timedelta(minutes=1)

    with TestClient(application) as raw_client:
        client = cast(Client, raw_client)
        session_response = client.post("/api/v1/sessions")
        session_id = cast(str, session_response.json()["session_id"])
        turn_response = client.post(
            f"/api/v1/sessions/{session_id}/messages",
            json={"message": "How much annual leave do employees receive?"},
        )

        assert session_response.status_code == 201
        assert turn_response.status_code == 200
        assert turn_response.json()["result"]["status"] == "answered"

        _assert_no_otel_export_errors(caplog.records)
        observations = _wait_for_complete_traces(
            recorder.client,
            session_id=session_id,
            started_at=query_started_at,
            expected=(
                _ExpectedTurn(
                    message="How much annual leave do employees receive?",
                    output=answer.markdown,
                    observations=Counter(
                        {
                            ("agent-turn", "AGENT"): 1,
                            ("select-action", "SPAN"): 2,
                            ("search_knowledge_base", "TOOL"): 1,
                            ("validate_answer", "EVALUATOR"): 1,
                        }
                    ),
                    parents={
                        "select-action": "agent-turn",
                        "search_knowledge_base": "agent-turn",
                        "validate_answer": "agent-turn",
                    },
                ),
            ),
        )

        roots = [
            observation
            for observation in observations
            if observation.is_root_observation
        ]
        assert len(roots) == 1
        root = roots[0]
        assert root.name == "agent-turn"
        assert _decode_observation_io(root.input) == (
            "How much annual leave do employees receive?"
        )
        assert _decode_observation_io(root.output) == answer.markdown
        assert root.session_id == session_id
        observation_contract = {(item.name, item.type) for item in observations}
        assert ("search_knowledge_base", "TOOL") in observation_contract
        assert ("validate_answer", "EVALUATOR") in observation_contract


@pytest.mark.langfuse_live
def test_coordinator_turn_is_one_retrievable_root_with_nested_rag(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Langfuse v4 must retain the coordinator-to-specialist parentage."""
    settings = AppSettings(agent_mode=AgentMode.COORDINATOR)
    if not settings.langfuse_enabled:
        pytest.fail("TOYAPP_LANGFUSE_ENABLED must be true for the live trace test")
    public_key, secret_key, base_url = settings.require_langfuse_configuration()
    recorder = LangfuseCoordinatorTraceRecorder(
        public_key=public_key,
        secret_key=secret_key,
        base_url=base_url,
        environment=settings.langfuse_environment,
        release=__version__,
        base_metadata={"test_kind": "coordinator_langfuse_compatibility"},
        rag_model_id="scripted-generation-fixture",
    )
    rag = RagAgent(
        provider=ScriptedGenerationProvider(
            (
                SearchAction(query="annual leave entitlement", top_k=3),
                AnswerAction(
                    markdown="Employees receive 18 days. [LEAVE-001#1]",
                    citations=("LEAVE-001#1",),
                ),
            )
        ),
        retriever=AsyncRetriever(_LiveTestRetriever(), CapacityLimiter(1)),
        system_prompt="Search the policy corpus before answering.",
    )
    agent = CoordinatorAgent(
        provider=ScriptedCoordinatorGenerationProvider(
            (
                CoordinatorAnswerAction(markdown="Hello."),
                CoordinatorRouteRagAction(task="Explain annual leave entitlement"),
            )
        ),
        system_prompt="Route policy questions.",
        rag_specialist=RagSpecialistAdapter(rag),
        trace_recorder=recorder,
    )
    query_started_at = datetime.now(UTC) - timedelta(minutes=1)

    with (
        caplog.at_level(logging.INFO, logger="uvicorn.error"),
        TestClient(create_app(settings, agent=agent)) as raw_client,
    ):
        client = cast(Client, raw_client)
        session_response = client.post("/api/v1/sessions")
        session_id = cast(str, session_response.json()["session_id"])
        direct_response = client.post(
            f"/api/v1/sessions/{session_id}/messages",
            json={"message": "Hello"},
        )
        turn_response = client.post(
            f"/api/v1/sessions/{session_id}/messages",
            json={"message": "How much annual leave do employees receive?"},
        )
        assert direct_response.status_code == 200
        assert direct_response.json()["result"]["message"] == "Hello."
        assert turn_response.status_code == 200
        assert turn_response.json()["result"]["status"] == "answered"
        app_events = [
            record.message
            for record in caplog.records
            if record.message.startswith("toyapp_trace ")
        ]
        assert (
            sum(
                "event=started" in line and "name=agent-turn" in line
                for line in app_events
            )
            == 2
        )

        _assert_no_otel_export_errors(caplog.records)
        observations = _wait_for_complete_traces(
            recorder.client,
            session_id=session_id,
            started_at=query_started_at,
            app_events=app_events,
            expected=(
                _ExpectedTurn(
                    message="Hello",
                    output="Hello.",
                    observations=Counter(
                        {
                            ("agent-turn", "AGENT"): 1,
                            ("inspect-input", "SPAN"): 1,
                            ("coordinator-action", "GENERATION"): 1,
                        }
                    ),
                    parents={
                        "inspect-input": "agent-turn",
                        "coordinator-action": "agent-turn",
                    },
                ),
                _ExpectedTurn(
                    message="How much annual leave do employees receive?",
                    output=turn_response.json()["result"]["message"],
                    observations=Counter(
                        {
                            ("agent-turn", "AGENT"): 1,
                            ("inspect-input", "SPAN"): 1,
                            ("coordinator-action", "GENERATION"): 1,
                            ("delegate-rag", "AGENT"): 1,
                            ("model-action", "GENERATION"): 2,
                            ("search_knowledge_base", "TOOL"): 1,
                            ("validate_answer", "EVALUATOR"): 1,
                            ("return-response", "SPAN"): 1,
                        }
                    ),
                    parents={
                        "inspect-input": "agent-turn",
                        "coordinator-action": "agent-turn",
                        "delegate-rag": "agent-turn",
                        "model-action": "delegate-rag",
                        "search_knowledge_base": "delegate-rag",
                        "validate_answer": "delegate-rag",
                        "return-response": "agent-turn",
                    },
                ),
            ),
        )

    roots = [item for item in observations if item.is_root_observation]
    assert len(roots) == 2
    assert {root.session_id for root in roots} == {session_id}
    assert ("delegate-rag", "AGENT") in {
        (item.name, item.type) for item in observations
    }
    assert ("search_knowledge_base", "TOOL") in {
        (item.name, item.type) for item in observations
    }
