"""Integration tests for service-introspection endpoints."""

from collections.abc import Sequence
from typing import cast
from unittest.mock import patch

import pytest
from httpx2 import Client, Response
from starlette.testclient import TestClient

from agentic_eval_toyapp.app import create_app
from agentic_eval_toyapp.config import (
    AgentMode,
    AppSettings,
    GenerationProviderKind,
    LogLevel,
)
from agentic_eval_toyapp.models import AgentTurnResult, ConversationMessage
from agentic_eval_toyapp.reports import StoredReport
from agentic_eval_toyapp.sessions import AgentRunner


class _IntrospectionAgent:
    def run(
        self,
        user_message: str,
        *,
        history: Sequence[ConversationMessage] = (),
        session_id: str | None = None,
        reports: Sequence[StoredReport] = (),
    ) -> AgentTurnResult:
        del session_id, reports
        raise AssertionError("introspection endpoints must not execute the agent")


def build_client() -> Client:
    """Create an isolated API client with no `.env` file dependency."""
    settings = AppSettings(
        app_name="Test Evaluation Target",
        host="127.0.0.1",
        port=8000,
        log_level=LogLevel.INFO,
        langfuse_enabled=False,
    )
    return cast(Client, TestClient(create_app(settings, agent=_IntrospectionAgent())))


def test_health_reports_liveness_and_version() -> None:
    """The health route should be stable and independent of future providers."""
    response: Response = build_client().get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "Test Evaluation Target",
        "version": "0.1.0",
    }


def test_ready_reports_session_and_agent_dependencies() -> None:
    """Readiness should cover every dependency needed by the chat API."""
    response: Response = build_client().get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "checks": [
            {
                "name": "configuration",
                "status": "pass",
                "detail": "service settings validated",
            },
            {
                "name": "session_store",
                "status": "pass",
                "detail": "retains up to 10 complete turns",
            },
            {
                "name": "report_store",
                "status": "pass",
                "detail": "retains up to 10 processed reports per session",
            },
            {
                "name": "langfuse_tracing",
                "status": "pass",
                "detail": "disabled",
            },
            {
                "name": "rag_agent",
                "status": "pass",
                "detail": "rag agent dependencies initialised",
            },
        ],
    }


def test_startup_initialises_the_agent_factory_exactly_once() -> None:
    """The server should finish RAG initialisation before serving readiness."""
    factory_calls = 0

    def agent_factory() -> AgentRunner:
        nonlocal factory_calls
        factory_calls += 1
        return _IntrospectionAgent()

    application = create_app(
        AppSettings(app_name="Startup Test Target"),
        agent_factory=agent_factory,
    )

    with TestClient(application) as raw_client:
        client = cast(Client, raw_client)
        assert factory_calls == 1
        assert client.get("/ready").json()["status"] == "ready"
        assert client.get("/ready").json()["status"] == "ready"
        assert factory_calls == 1


def test_startup_logs_safe_runtime_modes() -> None:
    """Operators should see provider and tracing modes without credentials."""
    settings = AppSettings(
        generation_provider=GenerationProviderKind.DETERMINISTIC,
        langfuse_enabled=False,
    )
    application = create_app(settings, agent=_IntrospectionAgent())

    with (
        patch("agentic_eval_toyapp.app.LOGGER.info") as log_info,
        TestClient(application),
    ):
        pass

    log_info.assert_called_once_with(
        "application_started agent_mode=%s generation_provider=%s langfuse_enabled=%s",
        "rag",
        "deterministic",
        "false",
    )


def test_coordinator_capabilities_exclude_deferred_specialists_and_tracing() -> None:
    """Stage 12 must not advertise Stage 13, 14, or 16 behavior as operational."""
    settings = AppSettings(
        agent_mode=AgentMode.COORDINATOR,
        generation_provider=GenerationProviderKind.DETERMINISTIC,
        langfuse_enabled=True,
    )
    client = cast(
        Client,
        TestClient(create_app(settings, agent=_IntrospectionAgent())),
    )

    readiness = client.get("/ready").json()
    capabilities = client.get("/api/v1/capabilities").json()

    assert readiness["checks"][3]["detail"] == (
        "coordinator trace export is deferred to Stage 16"
    )
    assert readiness["checks"][4] == {
        "name": "coordinator_agent",
        "status": "pass",
        "detail": "coordinator agent dependencies initialised",
    }
    assert capabilities["features"]["semantic_retrieval"] is False
    assert capabilities["features"]["langfuse_tracing"] is False
    assert capabilities["actions"] == [
        "answer",
        "clarify",
        "route_rag",
        "route_analysis",
    ]


def test_capabilities_advertise_only_operational_chat_features() -> None:
    """METIS should see chat and retrieval but not tracing or scenarios."""
    response: Response = build_client().get("/api/v1/capabilities")

    assert response.status_code == 200
    assert response.json() == {
        "api_version": "v1",
        "service_version": "0.1.0",
        "lifecycle": "synchronous",
        "features": {
            "session_chat": True,
            "semantic_retrieval": True,
            "langfuse_tracing": False,
            "controlled_scenarios": False,
            "file_uploads": True,
        },
        "actions": ["search", "clarify", "answer"],
        "output_formats": ["application/json"],
    }


def test_introspection_contract_ignores_local_langfuse_enablement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A developer's `.env` must not change deterministic endpoint tests."""
    monkeypatch.setenv("TOYAPP_LANGFUSE_ENABLED", "true")

    response: Response = build_client().get("/api/v1/capabilities")

    assert response.json()["features"]["langfuse_tracing"] is False


def test_openapi_contains_only_the_intended_application_routes() -> None:
    """The initial API schema should remain deliberately small."""
    response: Response = build_client().get("/openapi.json")
    payload = cast(dict[str, object], response.json())
    paths = payload["paths"]

    assert response.status_code == 200
    assert isinstance(paths, dict)
    typed_paths = cast(dict[str, object], paths)
    assert set(typed_paths) == {
        "/health",
        "/ready",
        "/api/v1/capabilities",
        "/api/v1/sessions",
        "/api/v1/sessions/{session_id}",
        "/api/v1/sessions/{session_id}/files",
        "/api/v1/sessions/{session_id}/messages",
    }
