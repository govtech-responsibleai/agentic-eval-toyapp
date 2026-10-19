"""Opt-in smoke test for the default semantic chat API wiring."""

from typing import cast

import pytest
from httpx2 import Client, Response
from starlette.testclient import TestClient

from agentic_eval_toyapp.app import create_app
from agentic_eval_toyapp.config import (
    AgentMode,
    AppSettings,
    GenerationProviderKind,
)


@pytest.mark.live
def test_default_api_completes_a_semantic_retrieval_turn() -> None:
    """The pinned embedder, FAISS, agent, sessions, and API should compose."""
    settings = AppSettings(generation_provider=GenerationProviderKind.DETERMINISTIC)
    with TestClient(create_app(settings)) as raw_client:
        client = cast(Client, raw_client)
        readiness: Response = client.get("/ready")
        created: Response = client.post("/api/v1/sessions")
        session_id = created.json()["session_id"]
        answered: Response = client.post(
            f"/api/v1/sessions/{session_id}/messages",
            json={"message": "How many annual leave days do employees receive?"},
        )

        assert readiness.status_code == 200
        assert readiness.json()["status"] == "ready"
        assert created.status_code == 201
        assert answered.status_code == 200
        assert answered.json()["result"]["status"] == "answered"
        assert "LEAVE-001#1" in answered.json()["result"]["citations"]


@pytest.mark.live
def test_coordinator_api_delegates_to_the_semantic_rag_stack() -> None:
    """Coordinator mode should compose with the pinned FAISS RAG specialist."""
    settings = AppSettings(
        agent_mode=AgentMode.COORDINATOR,
        generation_provider=GenerationProviderKind.DETERMINISTIC,
        langfuse_enabled=False,
    )
    with TestClient(create_app(settings)) as raw_client:
        client = cast(Client, raw_client)
        readiness: Response = client.get("/ready")
        created: Response = client.post("/api/v1/sessions")
        session_id = created.json()["session_id"]
        answered: Response = client.post(
            f"/api/v1/sessions/{session_id}/messages",
            json={"message": "How many annual leave days do employees receive?"},
        )

        result = answered.json()["result"]
        assert readiness.json()["status"] == "ready"
        assert answered.status_code == 200
        assert result["status"] == "answered"
        assert "LEAVE-001#1" in result["citations"]
        assert [step["action"] for step in result["execution_steps"]] == [
            "inspect_input",
            "route_rag",
            "search",
            "answer",
            "return_response",
        ]
