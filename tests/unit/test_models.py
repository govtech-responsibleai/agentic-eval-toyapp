"""Tests for public API models."""

import pytest
from pydantic import ValidationError

from agentic_eval_toyapp.models import (
    CapabilityResponse,
    ErrorResponse,
    FeatureAvailability,
    HealthResponse,
    LifecycleMode,
)


def test_capability_defaults_do_not_advertise_future_features() -> None:
    """Unimplemented target features must remain disabled."""
    response = CapabilityResponse(service_version="0.1.0")

    assert response.lifecycle is LifecycleMode.SYNCHRONOUS
    assert response.features == FeatureAvailability()
    assert response.actions == ()
    assert response.output_formats == ("application/json",)


def test_api_models_reject_undeclared_fields() -> None:
    """Unexpected response fields indicate contract drift."""
    with pytest.raises(ValidationError):
        HealthResponse.model_validate(
            {
                "status": "ok",
                "service": "toyapp",
                "version": "0.1.0",
                "unexpected": True,
            }
        )


@pytest.mark.parametrize(
    ("payload", "field"),
    [
        ({"code": "", "message": "failure"}, "code"),
        ({"code": "invalid_request", "message": ""}, "message"),
    ],
)
def test_error_response_rejects_empty_fields(
    payload: dict[str, str], field: str
) -> None:
    """Public errors require stable non-empty codes and messages."""
    with pytest.raises(ValidationError) as exc_info:
        ErrorResponse.model_validate(payload)

    assert field in str(exc_info.value)
