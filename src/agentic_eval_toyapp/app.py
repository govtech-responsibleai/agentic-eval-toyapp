"""FastAPI application and initial service-introspection endpoints."""

from fastapi import FastAPI

from agentic_eval_toyapp import __version__
from agentic_eval_toyapp.config import AppSettings, get_settings
from agentic_eval_toyapp.models import (
    CapabilityResponse,
    CheckStatus,
    HealthResponse,
    ReadinessCheck,
    ReadinessResponse,
    ReadinessStatus,
)

API_PREFIX = "/api/v1"


def create_app(settings: AppSettings | None = None) -> FastAPI:
    """Construct an application with explicit, validated settings."""
    resolved_settings = settings or get_settings()
    application = FastAPI(
        title=resolved_settings.app_name,
        version=__version__,
        description="Controlled RAG target application for METIS evaluations.",
    )

    @application.get("/health", response_model=HealthResponse, tags=["service"])
    def health() -> HealthResponse:
        """Report process liveness without checking external dependencies."""
        return HealthResponse(
            service=resolved_settings.app_name,
            version=__version__,
        )

    @application.get("/ready", response_model=ReadinessResponse, tags=["service"])
    def ready() -> ReadinessResponse:
        """Report readiness for the functionality implemented so far."""
        return ReadinessResponse(
            status=ReadinessStatus.READY,
            checks=(
                ReadinessCheck(
                    name="configuration",
                    status=CheckStatus.PASS,
                    detail="service settings validated",
                ),
            ),
        )

    @application.get(
        f"{API_PREFIX}/capabilities",
        response_model=CapabilityResponse,
        tags=["target"],
    )
    def capabilities() -> CapabilityResponse:
        """Advertise only target features that are currently operational."""
        return CapabilityResponse(service_version=__version__)

    return application


app = create_app()
