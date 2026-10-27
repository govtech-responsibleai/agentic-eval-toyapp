"""Integration tests for service-introspection endpoints."""

import logging
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import cast
from unittest.mock import patch
from uuid import UUID

import pytest
from httpx2 import Client, Response
from starlette.testclient import TestClient

from agentic_eval_toyapp.app import AgentInitialisationError, create_app
from agentic_eval_toyapp.config import (
    AgentMode,
    AppSettings,
    GenerationProviderKind,
    LogLevel,
    SessionStoreBackend,
)
from agentic_eval_toyapp.models import (
    AgentTurnResult,
    ConversationMessage,
    ReportUploadResponse,
    SessionResponse,
)
from agentic_eval_toyapp.reports import (
    InMemoryReportStore,
    ReportStoreUnavailableError,
    StoredReport,
)
from agentic_eval_toyapp.sessions import (
    AgentRunner,
    InMemorySessionStore,
    SessionStoreUnavailableError,
)


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


class _UnavailableSessionStore(InMemorySessionStore):
    def check(self) -> None:
        raise SessionStoreUnavailableError("private Redis connection details")

    def create(self) -> SessionResponse:
        raise SessionStoreUnavailableError("private Redis connection details")


class _UnavailableReportStore(InMemoryReportStore):
    def check(self) -> None:
        raise ReportStoreUnavailableError("private Redis connection details")

    def list_for_session(self, session_id: UUID) -> tuple[ReportUploadResponse, ...]:
        del session_id
        raise ReportStoreUnavailableError("private Redis connection details")


def build_client() -> Client:
    """Create an isolated API client with no `.env` file dependency."""
    settings = AppSettings(
        app_name="Test Evaluation Target",
        agent_mode=AgentMode.COORDINATOR,
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
                "detail": "retains up to 10 complete reports per session",
            },
            {
                "name": "langfuse_tracing",
                "status": "pass",
                "detail": "disabled",
            },
            {
                "name": "coordinator_agent",
                "status": "pass",
                "detail": "coordinator agent dependencies initialised",
            },
        ],
    }


def test_unavailable_persistent_store_fails_readiness_and_session_requests() -> None:
    """Redis outages should be public-safe and make the service not ready."""
    application = create_app(
        AppSettings(app_name="Persistent Test Target"),
        session_store=_UnavailableSessionStore(),
        agent=_IntrospectionAgent(),
    )
    client = cast(Client, TestClient(application))

    readiness = client.get("/ready")
    assert readiness.status_code == 200
    assert readiness.json()["status"] == "not_ready"
    assert readiness.json()["checks"][1] == {
        "name": "session_store",
        "status": "fail",
        "detail": "configured session store is unavailable",
    }

    response = client.post("/api/v1/sessions")
    assert response.status_code == 503
    assert response.json() == {
        "code": "session_store_unavailable",
        "message": "The conversation session store is temporarily unavailable.",
    }
    assert "private" not in response.text


def test_unavailable_report_store_fails_readiness_and_report_requests() -> None:
    """Report-store outages should be public-safe and make the service not ready."""
    session_store = InMemorySessionStore()
    session = session_store.create()
    application = create_app(
        AppSettings(app_name="Persistent Test Target"),
        session_store=session_store,
        report_store=_UnavailableReportStore(max_reports_per_session=10),
        agent=_IntrospectionAgent(),
    )
    client = cast(Client, TestClient(application))

    readiness = client.get("/ready")
    assert readiness.status_code == 200
    assert readiness.json()["status"] == "not_ready"
    assert readiness.json()["checks"][2] == {
        "name": "report_store",
        "status": "fail",
        "detail": "configured report store is unavailable",
    }

    response = client.get(f"/api/v1/sessions/{session.session_id}/files")
    assert response.status_code == 503
    assert response.json() == {
        "code": "report_store_unavailable",
        "message": "The report store is temporarily unavailable.",
    }
    assert "private" not in response.text


def test_redis_backend_configuration_builds_the_persistent_stores() -> None:
    """Application wiring should pass validated Redis settings to both stores."""
    settings = AppSettings.model_validate(
        {
            "session_store_backend": SessionStoreBackend.REDIS,
            "redis_url": "rediss://user:private@redis.example:6380/0",
            "redis_key_prefix": "test-prefix",
            "session_max_turns": 12,
        }
    )
    session_store = InMemorySessionStore(max_turns=12)
    report_store = InMemoryReportStore(max_reports_per_session=10)

    with (
        patch(
            "agentic_eval_toyapp.app.RedisSessionStore.from_url",
            return_value=session_store,
        ) as session_from_url,
        patch(
            "agentic_eval_toyapp.app.RedisReportStore.from_url",
            return_value=report_store,
        ) as report_from_url,
    ):
        client = cast(
            Client,
            TestClient(create_app(settings, agent=_IntrospectionAgent())),
        )
        assert client.get("/ready").json()["status"] == "ready"

    session_from_url.assert_called_once_with(
        "rediss://user:private@redis.example:6380/0",
        max_turns=12,
        key_prefix="test-prefix",
        socket_timeout_seconds=5.0,
        lock_timeout_seconds=1_800.0,
        lock_blocking_timeout_seconds=10.0,
    )
    report_from_url.assert_called_once_with(
        "rediss://user:private@redis.example:6380/0",
        max_reports_per_session=10,
        key_prefix="test-prefix",
        socket_timeout_seconds=5.0,
        lock_timeout_seconds=1_800.0,
        lock_blocking_timeout_seconds=10.0,
    )


def test_startup_initialises_the_agent_factory_exactly_once() -> None:
    """The server should finish agent initialisation before serving readiness."""
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


def test_default_startup_builds_the_coordinator(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An unset agent mode should select the coordinator at startup."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TOYAPP_AGENT_MODE", raising=False)
    settings = AppSettings(
        generation_provider=GenerationProviderKind.DETERMINISTIC,
        langfuse_enabled=False,
    )
    with (
        patch(
            "agentic_eval_toyapp.app._build_coordinator_agent",
            return_value=_IntrospectionAgent(),
        ) as build_coordinator,
        patch("agentic_eval_toyapp.app._build_rag_agent") as build_rag,
        TestClient(create_app(settings)) as client,
    ):
        assert client.get("/ready").json()["status"] == "ready"

    build_coordinator.assert_called_once_with(settings)
    build_rag.assert_not_called()


def test_startup_failure_logs_category_without_credential_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A dependency failure should be visible even when readiness never starts."""

    def failing_factory() -> AgentRunner:
        raise ValueError("private OPENAI_API_KEY=do-not-log")

    application = create_app(
        AppSettings(langfuse_enabled=False),
        agent_factory=failing_factory,
    )

    with (
        caplog.at_level(logging.ERROR, logger="uvicorn.error"),
        pytest.raises(
            AgentInitialisationError,
            match="agent dependencies could not be initialised",
        ) as failure,
        TestClient(application),
    ):
        pass

    failures = [
        record.message
        for record in caplog.records
        if "toyapp_failure" in record.message
    ]
    assert len(failures) == 1
    assert "event=startup" in failures[0]
    assert "code=configuration_error" in failures[0]
    assert "do-not-log" not in "".join(traceback.format_exception(failure.value))
    assert "do-not-log" not in caplog.text


def test_startup_logs_safe_runtime_modes() -> None:
    """Operators should see provider and tracing modes without credentials."""
    settings = AppSettings(
        agent_mode=AgentMode.COORDINATOR,
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
        "coordinator",
        "deterministic",
        "false",
    )


def test_coordinator_capabilities_advertise_multi_agent_tracing() -> None:
    """Coordinator and its selected specialist share operational tracing."""
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

    assert readiness["checks"][3]["detail"] == "enabled and configured"
    assert readiness["checks"][4] == {
        "name": "coordinator_agent",
        "status": "pass",
        "detail": "coordinator agent dependencies initialised",
    }
    assert capabilities["features"]["semantic_retrieval"] is True
    assert capabilities["features"]["langfuse_tracing"] is True
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
        "actions": ["answer", "clarify", "route_rag", "route_analysis"],
        "output_formats": ["application/json"],
    }


def test_explicit_rag_mode_advertises_standalone_actions_and_tracing() -> None:
    """The optional RAG runtime must retain its operational trace contract."""
    settings = AppSettings(agent_mode=AgentMode.RAG, langfuse_enabled=True)
    client = cast(Client, TestClient(create_app(settings, agent=_IntrospectionAgent())))

    capabilities = client.get("/api/v1/capabilities").json()

    assert capabilities["features"]["langfuse_tracing"] is True
    assert capabilities["actions"] == ["search", "clarify", "answer"]


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
        "/api/v1/sessions/{session_id}/files/{file_id}",
        "/api/v1/sessions/{session_id}/messages",
    }
