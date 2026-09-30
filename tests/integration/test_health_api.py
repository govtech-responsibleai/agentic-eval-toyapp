"""Integration tests for service-introspection endpoints."""

from typing import cast

from httpx2 import Client, Response
from starlette.testclient import TestClient

from agentic_eval_toyapp.app import create_app
from agentic_eval_toyapp.config import AppSettings, LogLevel


def build_client() -> Client:
    """Create an isolated API client with no `.env` file dependency."""
    settings = AppSettings(
        app_name="Test Evaluation Target",
        host="127.0.0.1",
        port=8000,
        log_level=LogLevel.INFO,
    )
    return cast(Client, TestClient(create_app(settings)))


def test_health_reports_liveness_and_version() -> None:
    """The health route should be stable and independent of future providers."""
    response: Response = build_client().get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "Test Evaluation Target",
        "version": "0.1.0",
    }


def test_ready_reports_configuration_check() -> None:
    """Stage 2 readiness covers validated configuration only."""
    response: Response = build_client().get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "checks": [
            {
                "name": "configuration",
                "status": "pass",
                "detail": "service settings validated",
            }
        ],
    }


def test_capabilities_do_not_claim_unimplemented_features() -> None:
    """METIS should see only features that are operational now."""
    response: Response = build_client().get("/api/v1/capabilities")

    assert response.status_code == 200
    assert response.json() == {
        "api_version": "v1",
        "service_version": "0.1.0",
        "lifecycle": "synchronous",
        "features": {
            "session_chat": False,
            "semantic_retrieval": False,
            "langfuse_tracing": False,
            "controlled_scenarios": False,
        },
        "actions": [],
        "output_formats": ["application/json"],
    }


def test_openapi_contains_only_the_intended_application_routes() -> None:
    """The initial API schema should remain deliberately small."""
    response: Response = build_client().get("/openapi.json")
    payload = cast(dict[str, object], response.json())
    paths = payload["paths"]

    assert response.status_code == 200
    assert isinstance(paths, dict)
    typed_paths = cast(dict[str, object], paths)
    assert set(typed_paths) == {"/health", "/ready", "/api/v1/capabilities"}
