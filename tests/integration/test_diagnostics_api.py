"""API-to-provider diagnostic correlation without external services."""

import logging
import re
from typing import Any, cast

import pytest
from httpx2 import Client
from starlette.testclient import TestClient

from agentic_eval_toyapp.agent import RagAgent
from agentic_eval_toyapp.app import create_app
from agentic_eval_toyapp.config import AgentMode, AppSettings, GenerationProviderKind
from agentic_eval_toyapp.models import RetrievedChunk
from agentic_eval_toyapp.providers import OpenAIChatGenerationProvider


class _FailingCompletionClient:
    async def create(
        self,
        *,
        model: str,
        messages: tuple[dict[str, str], ...],
        response_schema: dict[str, Any],
        temperature: float,
        max_tokens: int,
    ) -> str:
        del model, messages, response_schema, temperature, max_tokens
        raise TimeoutError("private prompt and API key")


class _UnusedRetriever:
    def search(
        self, query: str, *, top_k: int | None = None
    ) -> tuple[RetrievedChunk, ...]:
        del query, top_k
        raise AssertionError("provider failed before retrieval")


def test_two_failed_turns_share_session_but_not_turn_correlation_id(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """API requests should bind IDs that reach the nested live-model adapter."""
    provider = OpenAIChatGenerationProvider(
        api_key="unused-test-key",
        base_url="https://provider.example/v1",
        model="platform-model",
        revision="provider-managed",
        temperature=0,
        max_tokens=512,
        timeout_seconds=1,
        client=_FailingCompletionClient(),
    )
    agent = RagAgent(
        provider=provider,
        retriever=_UnusedRetriever(),
        system_prompt="Do not expose this prompt.",
    )
    app = create_app(
        AppSettings(
            agent_mode=AgentMode.RAG,
            generation_provider=GenerationProviderKind.DETERMINISTIC,
            langfuse_enabled=False,
        ),
        agent=agent,
    )

    with (
        caplog.at_level(logging.ERROR, logger="uvicorn.error"),
        TestClient(app) as raw_client,
    ):
        client = cast(Client, raw_client)
        session_id = client.post("/api/v1/sessions").json()["session_id"]
        for _ in range(2):
            response = client.post(
                f"/api/v1/sessions/{session_id}/messages",
                json={"message": "Private question"},
            )
            assert response.status_code == 200
            assert response.json()["result"]["termination_reason"] == "provider_error"

    failures = [
        record.message
        for record in caplog.records
        if "event=provider_call" in record.message
    ]
    assert len(failures) == 2
    assert all(f"session_id={session_id}" in message for message in failures)
    turn_ids = [re.search(r"turn_id=([0-9a-f-]+)", message) for message in failures]
    assert all(match is not None for match in turn_ids)
    assert len({match.group(1) for match in turn_ids if match is not None}) == 2
    assert all("code=timeout" in message for message in failures)
    assert "Private question" not in caplog.text
    assert "private prompt and API key" not in caplog.text
    assert "Do not expose this prompt" not in caplog.text
