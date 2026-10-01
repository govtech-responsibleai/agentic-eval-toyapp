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
    "TOYAPP_EMBEDDING_MODEL_ID",
    "TOYAPP_EMBEDDING_MODEL_REVISION",
    "TOYAPP_EMBEDDING_CACHE_DIR",
    "TOYAPP_EMBEDDING_DEVICE",
    "TOYAPP_RETRIEVAL_TOP_K",
    "TOYAPP_RETRIEVAL_MIN_SIMILARITY",
    "TOYAPP_AGENT_MAX_MODEL_CALLS",
    "TOYAPP_AGENT_MAX_SEARCHES",
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
    assert settings.embedding_model_id == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.embedding_model_revision == (
        "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    )
    assert settings.embedding_device == "cpu"
    assert settings.retrieval_top_k == 3
    assert settings.retrieval_min_similarity == 0.25
    assert settings.agent_max_model_calls == 3
    assert settings.agent_max_searches == 2


def test_settings_load_prefixed_environment_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only TOYAPP-prefixed values should override defaults."""
    monkeypatch.setenv("TOYAPP_APP_NAME", "  Evaluation Target  ")
    monkeypatch.setenv("TOYAPP_HOST", "0.0.0.0")
    monkeypatch.setenv("TOYAPP_PORT", "9000")
    monkeypatch.setenv("TOYAPP_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("TOYAPP_EMBEDDING_DEVICE", "mps")
    monkeypatch.setenv("TOYAPP_RETRIEVAL_TOP_K", "5")
    monkeypatch.setenv("TOYAPP_RETRIEVAL_MIN_SIMILARITY", "0.4")
    monkeypatch.setenv("TOYAPP_AGENT_MAX_MODEL_CALLS", "4")
    monkeypatch.setenv("TOYAPP_AGENT_MAX_SEARCHES", "1")

    settings = AppSettings()

    assert settings.app_name == "Evaluation Target"
    assert settings.host == "0.0.0.0"
    assert settings.port == 9000
    assert settings.log_level is LogLevel.DEBUG
    assert settings.embedding_device == "mps"
    assert settings.retrieval_top_k == 5
    assert settings.retrieval_min_similarity == 0.4
    assert settings.agent_max_model_calls == 4
    assert settings.agent_max_searches == 1


@pytest.mark.parametrize("port", [0, 65536])
def test_settings_reject_out_of_range_ports(port: int) -> None:
    """TCP ports must stay within the valid range."""
    with pytest.raises(ValidationError):
        AppSettings.model_validate({"port": port})


@pytest.mark.parametrize(
    "field",
    [
        "app_name",
        "host",
        "embedding_model_id",
        "embedding_model_revision",
        "embedding_device",
    ],
)
def test_settings_reject_blank_required_text(field: str) -> None:
    """Whitespace-only service identifiers and hosts are invalid."""
    with pytest.raises(ValidationError):
        AppSettings.model_validate({field: "   "})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("retrieval_top_k", 0),
        ("retrieval_min_similarity", -1.01),
        ("retrieval_min_similarity", 1.01),
        ("retrieval_min_similarity", float("nan")),
        ("agent_max_model_calls", 0),
        ("agent_max_searches", 0),
    ],
)
def test_settings_reject_invalid_retrieval_values(field: str, value: float) -> None:
    """Retrieval limits must stay within their mathematical domains."""
    with pytest.raises(ValidationError):
        AppSettings.model_validate({field: value})
