"""Offline checks for the per-turn and shutdown Langfuse flush contract."""

from collections.abc import Generator
from contextlib import contextmanager
from datetime import date
from typing import Any, cast

import pytest
from anyio import CapacityLimiter
from langfuse import Langfuse
from starlette.testclient import TestClient

import agentic_eval_toyapp.multi_agent_tracing as multi_agent_tracing_module
import agentic_eval_toyapp.tracing as tracing_module
from agentic_eval_toyapp.agent import RagAgent
from agentic_eval_toyapp.app import create_app
from agentic_eval_toyapp.config import AppSettings, GenerationProviderKind
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
from agentic_eval_toyapp.orchestration_models import CoordinatorAnswerAction
from agentic_eval_toyapp.providers import ScriptedGenerationProvider
from agentic_eval_toyapp.retrieval import AsyncRetriever
from agentic_eval_toyapp.sessions import AgentRunner
from agentic_eval_toyapp.tracing import LangfuseTraceRecorder

TURNS = 2


class _Observation:
    def update(self, **values: object) -> None:
        del values


class _FakeLangfuseClient:
    """Counts flush and shutdown calls; never touches the network."""

    def __init__(self) -> None:
        self.flushes = 0
        self.shutdowns = 0

    @contextmanager
    def start_as_current_observation(self, **values: object) -> Generator[_Observation]:
        del values
        yield _Observation()

    def get_current_trace_id(self) -> str:
        return "a" * 32

    def get_current_observation_id(self) -> str:
        return "b" * 16

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


@pytest.fixture(autouse=True)
def _fake_propagation(monkeypatch: pytest.MonkeyPatch) -> None:
    @contextmanager
    def ignore(**values: object) -> Generator[None]:
        del values
        yield

    monkeypatch.setattr(tracing_module, "propagate_attributes", ignore)
    monkeypatch.setattr(multi_agent_tracing_module, "propagate_attributes", ignore)


def _rag_agent(client: _FakeLangfuseClient, *, flush_on_turn: bool) -> RagAgent:
    actions: list[SearchAction | AnswerAction] = []
    for _ in range(TURNS):
        actions.append(SearchAction(query="annual leave", top_k=3))
        actions.append(
            AnswerAction(
                markdown="Employees receive 18 days. [LEAVE-001#1]",
                citations=("LEAVE-001#1",),
            )
        )
    return RagAgent(
        provider=ScriptedGenerationProvider(tuple(actions)),
        retriever=AsyncRetriever(_Retriever(), CapacityLimiter(1)),
        system_prompt="Answer policy questions.",
        trace_recorder=LangfuseTraceRecorder(
            public_key="public",
            secret_key="secret",
            base_url="https://langfuse.example",
            environment="test",
            release="0.1.0",
            base_metadata={},
            flush_on_turn=flush_on_turn,
            client=cast(Langfuse, client),
        ),
    )


def _coordinator_agent(
    client: _FakeLangfuseClient, *, flush_on_turn: bool
) -> CoordinatorAgent:
    return CoordinatorAgent(
        provider=ScriptedCoordinatorGenerationProvider(
            tuple(CoordinatorAnswerAction(markdown="Hello.") for _ in range(TURNS))
        ),
        system_prompt="Coordinate.",
        trace_recorder=LangfuseCoordinatorTraceRecorder(
            public_key="public",
            secret_key="secret",
            base_url="https://langfuse.example",
            environment="test",
            release="0.1.0",
            base_metadata={},
            rag_model_id="scripted-rag-fixture",
            flush_on_turn=flush_on_turn,
            client=cast(Langfuse, client),
        ),
    )


def _run_turns_through_lifespan(
    agent: AgentRunner, client: _FakeLangfuseClient
) -> tuple[int, int]:
    """Send TURNS messages; return (flushes, shutdowns) seen before and after exit."""
    application = create_app(
        AppSettings(
            generation_provider=GenerationProviderKind.DETERMINISTIC,
            langfuse_enabled=False,
        ),
        agent=agent,
    )
    with TestClient(application) as api:
        session_id = api.post("/api/v1/sessions").json()["session_id"]
        for _ in range(TURNS):
            response = api.post(
                f"/api/v1/sessions/{session_id}/messages",
                json={"message": "How much annual leave do I get?"},
            )
            assert response.status_code == 200
        # The lifespan has not exited, so no shutdown flush has happened yet.
        assert client.shutdowns == 0
        flushes_during_turns = client.flushes
    return flushes_during_turns, client.shutdowns


@pytest.mark.parametrize("mode", ["rag", "coordinator"])
@pytest.mark.parametrize(
    ("flush_on_turn", "expected_turn_flushes"), [(False, 0), (True, TURNS)]
)
def test_per_turn_flush_follows_the_flag_and_shutdown_runs_once(
    mode: str, flush_on_turn: bool, expected_turn_flushes: int
) -> None:
    """Flag off: zero per-turn flushes. Flag on: one per turn. Shutdown: once."""
    client = _FakeLangfuseClient()
    agent: AgentRunner = (
        _rag_agent(client, flush_on_turn=flush_on_turn)
        if mode == "rag"
        else _coordinator_agent(client, flush_on_turn=flush_on_turn)
    )

    flushes_during_turns, shutdowns = _run_turns_through_lifespan(agent, client)

    assert flushes_during_turns == expected_turn_flushes
    # The recorder's shutdown() calls the SDK's shutdown(), which flushes
    # internally, so the final flush is one shutdown call and not an extra flush().
    assert shutdowns == 1
    assert client.flushes == expected_turn_flushes


def test_shutdown_without_any_turn_still_shuts_the_client_down_once() -> None:
    """An idle deferred-mode process must still drain the SDK queue on exit."""
    client = _FakeLangfuseClient()
    agent = _coordinator_agent(client, flush_on_turn=False)

    application = create_app(
        AppSettings(
            generation_provider=GenerationProviderKind.DETERMINISTIC,
            langfuse_enabled=False,
        ),
        agent=agent,
    )
    with TestClient(application):
        pass

    assert client.flushes == 0
    assert client.shutdowns == 1


def test_injected_agent_without_shutdown_is_left_alone() -> None:
    """Custom runners that are not RagAgent or CoordinatorAgent are not touched."""
    client = _FakeLangfuseClient()
    real_agent = _coordinator_agent(client, flush_on_turn=False)

    class _Wrapper:
        """Delegates run() but is not a CoordinatorAgent, so the resolver skips it."""

        async def run(self, *args: Any, **kwargs: Any) -> Any:
            return await real_agent.run(*args, **kwargs)

    application = create_app(
        AppSettings(
            generation_provider=GenerationProviderKind.DETERMINISTIC,
            langfuse_enabled=False,
        ),
        agent=cast(AgentRunner, _Wrapper()),
    )
    with TestClient(application):
        pass

    assert client.shutdowns == 0
