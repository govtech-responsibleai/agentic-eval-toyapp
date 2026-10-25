"""Tests for the Langfuse observation contract without network access."""

import logging
import traceback
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date
from typing import Any, cast

import pytest
from langfuse import Langfuse

import agentic_eval_toyapp.tracing as tracing_module
from agentic_eval_toyapp.models import (
    AgentTurnResult,
    AgentTurnStatus,
    AnswerAction,
    DocumentMetadata,
    ModelRequest,
    RetrievedChunk,
    SearchAction,
    TerminationReason,
    ValidationFinding,
    ValidationFindingCode,
)
from agentic_eval_toyapp.tracing import LangfuseTraceRecorder, NoOpTraceRecorder


def _empty_updates() -> list[dict[str, object]]:
    return []


@dataclass
class _RecordedObservation:
    name: str
    observation_type: str
    input: object
    depth: int
    updates: list[dict[str, object]] = field(default_factory=_empty_updates)

    def update(self, **values: object) -> None:
        self.updates.append(values)


class _RecordingLangfuseClient:
    def __init__(self) -> None:
        self.observations: list[_RecordedObservation] = []
        self._depth = 0
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
    ) -> Generator[_RecordedObservation]:
        del metadata
        observation = _RecordedObservation(name, as_type, input, self._depth)
        self.observations.append(observation)
        self._depth += 1
        try:
            yield observation
        finally:
            self._depth -= 1

    def flush(self) -> None:
        self.flushes += 1

    def shutdown(self) -> None:
        self.shutdowns += 1


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="LEAVE-001#1",
        document_id="LEAVE-001",
        title="Leave policy",
        text="Employees receive 18 days of annual leave.",
        metadata=DocumentMetadata(topic="leave", effective_date=date(2026, 1, 1)),
        similarity_score=0.91,
    )


def test_noop_recorder_preserves_the_agent_contract() -> None:
    """Offline execution should require no credentials and emit no side effects."""
    recorder = NoOpTraceRecorder()

    with (
        recorder.start_turn(
            session_id=None,
            user_message="Question",
            history_message_count=0,
        ) as turn,
        turn.model_action(
            ModelRequest(
                system_prompt="Choose an action.",
                user_message="Question",
                model_call_number=1,
                remaining_searches=1,
            )
        ) as observation,
    ):
        observation.fail("fixture_error")

    assert recorder.enabled is False
    recorder.flush()
    recorder.shutdown()


def test_langfuse_flush_failure_logs_only_safe_category(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Export failures must be distinguishable without leaking SDK text."""

    class _FailingClient(_RecordingLangfuseClient):
        def flush(self) -> None:
            raise RuntimeError("private Langfuse connection detail")

    recorder = LangfuseTraceRecorder(
        public_key="public",
        secret_key="secret",
        base_url="https://langfuse.example",
        environment="test",
        release="0.1.0",
        base_metadata={},
        client=cast(Langfuse, _FailingClient()),
    )

    with (
        caplog.at_level(logging.ERROR, logger="uvicorn.error"),
        pytest.raises(RuntimeError, match="Langfuse trace flush failed") as failure,
    ):
        recorder.flush()

    assert len(caplog.records) == 1
    assert "code=langfuse_flush_error" in caplog.text
    assert "private Langfuse" not in caplog.text
    assert "private Langfuse" not in "".join(traceback.format_exception(failure.value))


def test_langfuse_shutdown_failure_logs_only_safe_category(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Shutdown failures retain a safe category without exposing SDK text."""

    class _FailingClient(_RecordingLangfuseClient):
        def shutdown(self) -> None:
            raise RuntimeError("private Langfuse shutdown detail")

    recorder = LangfuseTraceRecorder(
        public_key="public",
        secret_key="secret",
        base_url="https://langfuse.example",
        environment="test",
        release="0.1.0",
        base_metadata={},
        client=cast(Langfuse, _FailingClient()),
    )

    with (
        caplog.at_level(logging.ERROR, logger="uvicorn.error"),
        pytest.raises(RuntimeError, match="Langfuse trace shutdown failed") as failure,
    ):
        recorder.shutdown()

    assert len(caplog.records) == 1
    assert "code=langfuse_shutdown_error" in caplog.text
    assert "private Langfuse" not in caplog.text
    assert "private Langfuse" not in "".join(traceback.format_exception(failure.value))


def test_langfuse_recorder_emits_metis_compatible_turn_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One app turn must become one output-bearing, session-correlated trace."""
    propagation_calls: list[dict[str, object]] = []

    @contextmanager
    def record_propagation(**values: object) -> Generator[None]:
        propagation_calls.append(values)
        yield

    monkeypatch.setattr(tracing_module, "propagate_attributes", record_propagation)
    fake_client = _RecordingLangfuseClient()
    recorder = LangfuseTraceRecorder(
        public_key="public",
        secret_key="secret",
        base_url="https://langfuse.example",
        environment="test",
        release="0.1.0",
        base_metadata={"corpus_version": "2026-01"},
        client=cast(Langfuse, fake_client),
    )
    chunk = _chunk()
    search = SearchAction(query="annual leave", top_k=3)
    answer = AnswerAction(
        markdown="Employees receive 18 days. [LEAVE-001#1]",
        citations=("LEAVE-001#1",),
    )
    finding = ValidationFinding(
        code=ValidationFindingCode.DUPLICATE_CITATION,
        detail="Fixture finding.",
        citation_ids=("LEAVE-001#1",),
    )
    request = ModelRequest(
        system_prompt="Choose an action.",
        user_message="How much annual leave is provided?",
        model_call_number=1,
        remaining_searches=2,
    )
    result = AgentTurnResult(
        status=AgentTurnStatus.ANSWERED,
        message=answer.markdown,
        citations=answer.citations,
        retrieved_chunks=(chunk,),
        validation_findings=(finding,),
        model_actions=(search, answer),
        termination_reason=TerminationReason.ANSWERED,
        model_calls=2,
        searches=1,
    )

    with recorder.start_turn(
        session_id="session-123",
        user_message=request.user_message,
        history_message_count=2,
    ) as turn:
        with turn.model_action(request) as model_observation:
            model_observation.succeed(search)
        turn.record_search(query=search.query, top_k=search.top_k, results=(chunk,))
        turn.record_validation(
            citations=answer.citations,
            candidate_ids=(chunk.chunk_id,),
            findings=(finding,),
        )
        turn.complete(result)

    assert propagation_calls == [
        {
            "session_id": "session-123",
            "trace_name": "agent-turn",
            "metadata": {"corpus_version": "2026-01"},
            "version": "0.1.0",
            "tags": ["agentic-eval-toyapp", "rag"],
        }
    ]
    assert [
        (observation.name, observation.observation_type, observation.depth)
        for observation in fake_client.observations
    ] == [
        ("agent-turn", "agent", 0),
        ("select-action", "span", 1),
        ("search_knowledge_base", "tool", 1),
        ("validate_answer", "evaluator", 1),
    ]
    root_update = fake_client.observations[0].updates[-1]
    assert root_update["output"] == result.message
    assert (
        cast(dict[str, object], root_update["metadata"])["termination_reason"]
        == "answered"
    )
    search_update = fake_client.observations[2].updates[-1]
    assert cast(list[dict[str, Any]], search_update["output"])[0]["chunk_id"] == (
        "LEAVE-001#1"
    )
    assert fake_client.flushes == 1

    recorder.shutdown()
    assert fake_client.shutdowns == 1


def test_clarification_uses_the_metis_recognised_tool_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """METIS detects clarification from a request_user_input TOOL observation."""

    @contextmanager
    def ignore_propagation(**values: object) -> Generator[None]:
        del values
        yield

    monkeypatch.setattr(tracing_module, "propagate_attributes", ignore_propagation)
    fake_client = _RecordingLangfuseClient()
    recorder = LangfuseTraceRecorder(
        public_key="public",
        secret_key="secret",
        base_url="https://langfuse.example",
        environment="test",
        release="0.1.0",
        base_metadata={},
        client=cast(Langfuse, fake_client),
        flush_on_turn=False,
    )

    with recorder.start_turn(
        session_id="session-clarify",
        user_message="Tell me about leave.",
        history_message_count=0,
    ) as turn:
        turn.record_clarification("Which type of leave?")

    clarification = fake_client.observations[1]
    assert clarification.name == "request_user_input"
    assert clarification.observation_type == "tool"
    assert clarification.input == {"question": "Which type of leave?"}
    assert fake_client.flushes == 0
