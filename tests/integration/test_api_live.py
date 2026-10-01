"""Opt-in smoke test for the default semantic chat API wiring."""

from typing import cast

import pytest
from httpx2 import Client, Response
from starlette.testclient import TestClient

from agentic_eval_toyapp.app import create_app
from agentic_eval_toyapp.config import AppSettings


@pytest.mark.live
def test_default_api_completes_a_semantic_retrieval_turn() -> None:
    """The pinned embedder, FAISS, agent, sessions, and API should compose."""
    with TestClient(create_app(AppSettings())) as raw_client:
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
