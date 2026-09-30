"""Pydantic contracts for the initial service API."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ApiModel(BaseModel):
    """Base contract that rejects undeclared fields and is immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class HealthStatus(StrEnum):
    """Liveness status values."""

    OK = "ok"


class ReadinessStatus(StrEnum):
    """Aggregate readiness status values."""

    READY = "ready"
    NOT_READY = "not_ready"


class CheckStatus(StrEnum):
    """Individual readiness-check status values."""

    PASS = "pass"
    FAIL = "fail"


class LifecycleMode(StrEnum):
    """Supported request lifecycle modes."""

    SYNCHRONOUS = "synchronous"


class HealthResponse(ApiModel):
    """Response returned by the liveness endpoint."""

    status: HealthStatus = HealthStatus.OK
    service: str = Field(min_length=1)
    version: str = Field(min_length=1)


class ReadinessCheck(ApiModel):
    """One named readiness check."""

    name: str = Field(min_length=1)
    status: CheckStatus
    detail: str | None = None


class ReadinessResponse(ApiModel):
    """Aggregate service readiness response."""

    status: ReadinessStatus
    checks: tuple[ReadinessCheck, ...]


class FeatureAvailability(ApiModel):
    """Feature flags advertised to target clients."""

    session_chat: bool = False
    semantic_retrieval: bool = False
    langfuse_tracing: bool = False
    controlled_scenarios: bool = False


class CapabilityResponse(ApiModel):
    """Machine-readable description of the currently available target API."""

    api_version: str = "v1"
    service_version: str = Field(min_length=1)
    lifecycle: LifecycleMode = LifecycleMode.SYNCHRONOUS
    features: FeatureAvailability = Field(default_factory=FeatureAvailability)
    actions: tuple[str, ...] = ()
    output_formats: tuple[str, ...] = ("application/json",)


class ErrorResponse(ApiModel):
    """Stable public error shape for future API routes."""

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
