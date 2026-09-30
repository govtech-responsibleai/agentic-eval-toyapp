"""Tests for typed application configuration."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from agentic_eval_toyapp.config import AppSettings, LogLevel

_SETTING_ENV_VARS = (
    "TOYAPP_APP_NAME",
    "TOYAPP_HOST",
    "TOYAPP_PORT",
    "TOYAPP_LOG_LEVEL",
)


@pytest.fixture(autouse=True)
def isolate_settings_sources(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Prevent a developer's environment or `.env` file affecting tests."""
    monkeypatch.chdir(tmp_path)
    for variable in _SETTING_ENV_VARS:
        monkeypatch.delenv(variable, raising=False)


def test_settings_defaults_are_safe_for_local_development() -> None:
    """Defaults should bind locally and use a stable log level."""
    settings = AppSettings()

    assert settings.app_name == "Agentic Eval Toy App"
    assert settings.host == "127.0.0.1"
    assert settings.port == 8000
    assert settings.log_level is LogLevel.INFO


def test_settings_load_prefixed_environment_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only TOYAPP-prefixed values should override defaults."""
    monkeypatch.setenv("TOYAPP_APP_NAME", "  Evaluation Target  ")
    monkeypatch.setenv("TOYAPP_HOST", "0.0.0.0")
    monkeypatch.setenv("TOYAPP_PORT", "9000")
    monkeypatch.setenv("TOYAPP_LOG_LEVEL", "DEBUG")

    settings = AppSettings()

    assert settings.app_name == "Evaluation Target"
    assert settings.host == "0.0.0.0"
    assert settings.port == 9000
    assert settings.log_level is LogLevel.DEBUG


@pytest.mark.parametrize("port", [0, 65536])
def test_settings_reject_out_of_range_ports(port: int) -> None:
    """TCP ports must stay within the valid range."""
    with pytest.raises(ValidationError):
        AppSettings.model_validate({"port": port})


@pytest.mark.parametrize("field", ["app_name", "host"])
def test_settings_reject_blank_required_text(field: str) -> None:
    """Whitespace-only service identifiers and hosts are invalid."""
    with pytest.raises(ValidationError):
        AppSettings.model_validate({field: "   "})
