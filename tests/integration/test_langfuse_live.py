"""Opt-in end-to-end Langfuse v4 export and retrieval compatibility test."""

import json
import time
from datetime import UTC, date, datetime, timedelta
from typing import cast

import pytest
from httpx2 import Client
from starlette.testclient import TestClient

from agentic_eval_toyapp import __version__
from agentic_eval_toyapp.agent import RagAgent
from agentic_eval_toyapp.app import create_app
from agentic_eval_toyapp.config import AppSettings
from agentic_eval_toyapp.models import (
    AnswerAction,
    DocumentMetadata,
    RetrievedChunk,
    SearchAction,
)
from agentic_eval_toyapp.providers import ScriptedGenerationProvider
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


@pytest.mark.langfuse_live
def test_api_turn_is_retrievable_by_app_session_id() -> None:
    """Use the same session query contract consumed by METIS's extractor."""
    settings = AppSettings()
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
        retriever=_LiveTestRetriever(),
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

        deadline = time.monotonic() + 45
        observations = []
        while time.monotonic() < deadline:
            observations = recorder.client.api.observations.get_many(
                session_id=session_id,
                fields="core,basic,io,trace_context",
                from_start_time=query_started_at,
                to_start_time=datetime.now(UTC) + timedelta(minutes=1),
                limit=100,
            ).data
            names = {observation.name for observation in observations}
            roots = [
                observation
                for observation in observations
                if observation.is_root_observation
            ]
            if (
                len(roots) == 1
                and roots[0].output is not None
                and {"search_knowledge_base", "validate_answer"} <= names
            ):
                break
            time.sleep(2)

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
