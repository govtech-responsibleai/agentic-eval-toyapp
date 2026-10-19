"""Tests for typed application configuration."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from agentic_eval_toyapp.config import (
    AgentMode,
    AppSettings,
    GenerationProviderKind,
    LogLevel,
)

_SETTING_ENV_VARS = (
    "TOYAPP_APP_NAME",
    "TOYAPP_HOST",
    "TOYAPP_PORT",
    "TOYAPP_LOG_LEVEL",
    "TOYAPP_EMBEDDING_MODEL_ID",
    "TOYAPP_EMBEDDING_MODEL_REVISION",
    "TOYAPP_EMBEDDING_CACHE_DIR",
    "TOYAPP_EMBEDDING_LOCAL_FILES_ONLY",
    "TOYAPP_EMBEDDING_DEVICE",
    "TOYAPP_RETRIEVAL_TOP_K",
    "TOYAPP_RETRIEVAL_MIN_SIMILARITY",
    "TOYAPP_AGENT_MAX_MODEL_CALLS",
    "TOYAPP_AGENT_MAX_SEARCHES",
    "TOYAPP_AGENT_MODE",
    "TOYAPP_COORDINATOR_MAX_MODEL_CALLS",
    "TOYAPP_GENERATION_PROVIDER",
    "TOYAPP_GENERATION_MODEL",
    "TOYAPP_GENERATION_MODEL_REVISION",
    "TOYAPP_GENERATION_TEMPERATURE",
    "TOYAPP_GENERATION_MAX_TOKENS",
    "TOYAPP_GENERATION_TIMEOUT_SECONDS",
    "TOYAPP_OPENAI_API_KEY",
    "TOYAPP_OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "TOYAPP_LANGFUSE_ENABLED",
    "TOYAPP_LANGFUSE_PUBLIC_KEY",
    "TOYAPP_LANGFUSE_SECRET_KEY",
    "TOYAPP_LANGFUSE_BASE_URL",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_BASE_URL",
    "TOYAPP_LANGFUSE_ENVIRONMENT",
    "TOYAPP_LANGFUSE_FLUSH_ON_TURN",
    "TOYAPP_SESSION_MAX_TURNS",
    "TOYAPP_UPLOAD_MAX_BYTES",
    "TOYAPP_UPLOAD_MAX_PDF_PAGES",
    "TOYAPP_UPLOAD_MAX_EXTRACTED_CHARACTERS",
    "TOYAPP_UPLOAD_MAX_FILES_PER_MESSAGE",
    "TOYAPP_UPLOAD_MAX_REPORTS_PER_SESSION",
)


@pytest.fixture(autouse=True)
def isolate_settings_sources(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Prevent a developer's environment or `.env` file affecting tests."""
    monkeypatch.chdir(tmp_path)
    for variable in _SETTING_ENV_VARS:
        monkeypatch.delenv(variable, raising=False)


def test_settings_defaults_select_the_live_runtime_without_storing_credentials() -> (
    None
):
    """Normal app traffic should default to live generation with no fake key."""
    settings = AppSettings()

    assert settings.app_name == "Agentic Eval Toy App"
    assert settings.host == "127.0.0.1"
    assert settings.port == 8000
    assert settings.log_level is LogLevel.INFO
    assert settings.embedding_model_id == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.embedding_model_revision == (
        "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    )
    assert settings.embedding_local_files_only is False
    assert settings.embedding_device == "cpu"
    assert settings.retrieval_top_k == 3
    assert settings.retrieval_min_similarity == 0.25
    assert settings.agent_max_model_calls == 4
    assert settings.agent_max_searches == 2
    assert settings.agent_mode is AgentMode.RAG
    assert settings.coordinator_max_model_calls == 2
    assert settings.generation_provider is GenerationProviderKind.OPENAI
    assert settings.generation_model == "gpt-5.6-luna"
    assert settings.generation_model_revision == "provider-managed"
    assert settings.generation_temperature == 0.0
    assert settings.generation_max_tokens == 1_024
    assert settings.generation_timeout_seconds == 30.0
    assert settings.openai_api_key is None
    assert settings.openai_base_url is None
    assert settings.langfuse_enabled is False
    assert settings.langfuse_public_key is None
    assert settings.langfuse_secret_key is None
    assert settings.langfuse_base_url is None
    assert settings.langfuse_environment == "development"
    assert settings.langfuse_flush_on_turn is True
    assert settings.session_max_turns == 10
    assert settings.upload_max_bytes == 5 * 1024 * 1024
    assert settings.upload_max_pdf_pages == 50
    assert settings.upload_max_extracted_characters == 100_000
    assert settings.upload_max_files_per_message == 1
    assert settings.upload_max_reports_per_session == 10


def test_settings_load_prefixed_environment_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only TOYAPP-prefixed values should override defaults."""
    monkeypatch.setenv("TOYAPP_APP_NAME", "  Evaluation Target  ")
    monkeypatch.setenv("TOYAPP_HOST", "0.0.0.0")
    monkeypatch.setenv("TOYAPP_PORT", "9000")
    monkeypatch.setenv("TOYAPP_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("TOYAPP_EMBEDDING_DEVICE", "mps")
    monkeypatch.setenv("TOYAPP_EMBEDDING_LOCAL_FILES_ONLY", "true")
    monkeypatch.setenv("TOYAPP_RETRIEVAL_TOP_K", "5")
    monkeypatch.setenv("TOYAPP_RETRIEVAL_MIN_SIMILARITY", "0.4")
    monkeypatch.setenv("TOYAPP_AGENT_MAX_MODEL_CALLS", "4")
    monkeypatch.setenv("TOYAPP_AGENT_MAX_SEARCHES", "1")
    monkeypatch.setenv("TOYAPP_AGENT_MODE", "coordinator")
    monkeypatch.setenv("TOYAPP_COORDINATOR_MAX_MODEL_CALLS", "3")
    monkeypatch.setenv("TOYAPP_GENERATION_PROVIDER", "deterministic")
    monkeypatch.setenv("TOYAPP_GENERATION_MODEL", "platform-model")
    monkeypatch.setenv("TOYAPP_GENERATION_MODEL_REVISION", "revision-7")
    monkeypatch.setenv("TOYAPP_GENERATION_TEMPERATURE", "0.2")
    monkeypatch.setenv("TOYAPP_GENERATION_MAX_TOKENS", "512")
    monkeypatch.setenv("TOYAPP_GENERATION_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("OPENAI_API_KEY", "secret-value")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://platform.example/v1/")
    monkeypatch.setenv("TOYAPP_LANGFUSE_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "public-value")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "secret-value")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://langfuse.example/")
    monkeypatch.setenv("TOYAPP_LANGFUSE_ENVIRONMENT", "test")
    monkeypatch.setenv("TOYAPP_LANGFUSE_FLUSH_ON_TURN", "false")
    monkeypatch.setenv("TOYAPP_SESSION_MAX_TURNS", "20")
    monkeypatch.setenv("TOYAPP_UPLOAD_MAX_BYTES", "4096")
    monkeypatch.setenv("TOYAPP_UPLOAD_MAX_PDF_PAGES", "25")
    monkeypatch.setenv("TOYAPP_UPLOAD_MAX_EXTRACTED_CHARACTERS", "50000")
    monkeypatch.setenv("TOYAPP_UPLOAD_MAX_FILES_PER_MESSAGE", "2")
    monkeypatch.setenv("TOYAPP_UPLOAD_MAX_REPORTS_PER_SESSION", "15")

    settings = AppSettings()

    assert settings.app_name == "Evaluation Target"
    assert settings.host == "0.0.0.0"
    assert settings.port == 9000
    assert settings.log_level is LogLevel.DEBUG
    assert settings.embedding_device == "mps"
    assert settings.embedding_local_files_only is True
    assert settings.retrieval_top_k == 5
    assert settings.retrieval_min_similarity == 0.4
    assert settings.agent_max_model_calls == 4
    assert settings.agent_max_searches == 1
    assert settings.agent_mode is AgentMode.COORDINATOR
    assert settings.coordinator_max_model_calls == 3
    assert settings.generation_provider is GenerationProviderKind.DETERMINISTIC
    assert settings.generation_model == "platform-model"
    assert settings.generation_model_revision == "revision-7"
    assert settings.generation_temperature == 0.2
    assert settings.generation_max_tokens == 512
    assert settings.generation_timeout_seconds == 45.0
    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == "secret-value"
    assert settings.openai_base_url == "https://platform.example/v1"
    assert settings.langfuse_enabled is True
    assert settings.langfuse_public_key is not None
    assert settings.langfuse_public_key.get_secret_value() == "public-value"
    assert settings.langfuse_secret_key is not None
    assert settings.langfuse_secret_key.get_secret_value() == "secret-value"
    assert settings.langfuse_base_url == "https://langfuse.example"
    assert settings.langfuse_environment == "test"
    assert settings.langfuse_flush_on_turn is False
    assert settings.session_max_turns == 20
    assert settings.upload_max_bytes == 4096
    assert settings.upload_max_pdf_pages == 25
    assert settings.upload_max_extracted_characters == 50_000
    assert settings.upload_max_files_per_message == 2
    assert settings.upload_max_reports_per_session == 15


def test_live_configuration_requires_both_shared_openai_values() -> None:
    """Live startup must fail clearly instead of silently selecting a fake."""
    settings = AppSettings()

    with pytest.raises(ValueError, match="OPENAI_API_KEY and OPENAI_BASE_URL"):
        settings.require_openai_configuration()


def test_live_configuration_returns_secrets_without_exposing_them_in_repr() -> None:
    """Provider wiring can consume credentials while settings mask the key."""
    settings = AppSettings.model_validate(
        {
            "openai_api_key": "private-key",
            "openai_base_url": "https://platform.example/v1/",
        }
    )

    assert settings.require_openai_configuration() == (
        "private-key",
        "https://platform.example/v1",
    )
    assert "private-key" not in repr(settings)


def test_langfuse_configuration_requires_all_values_when_enabled() -> None:
    """Tracing must not silently start with a partially configured exporter."""
    settings = AppSettings(langfuse_enabled=True)

    with pytest.raises(
        ValueError,
        match=("LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY and LANGFUSE_BASE_URL"),
    ):
        settings.require_langfuse_configuration()


def test_langfuse_configuration_masks_secrets_and_normalises_url() -> None:
    """Runtime wiring may consume credentials without exposing them in repr."""
    settings = AppSettings.model_validate(
        {
            "langfuse_enabled": True,
            "langfuse_public_key": "public-key",
            "langfuse_secret_key": "private-secret",
            "langfuse_base_url": "https://langfuse.example/",
        }
    )

    assert settings.require_langfuse_configuration() == (
        "public-key",
        "private-secret",
        "https://langfuse.example",
    )
    assert "private-secret" not in repr(settings)


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
        "generation_model",
        "generation_model_revision",
        "langfuse_environment",
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
        ("retrieval_top_k", 9),
        ("retrieval_min_similarity", -1.01),
        ("retrieval_min_similarity", 1.01),
        ("retrieval_min_similarity", float("nan")),
        ("agent_max_model_calls", 0),
        ("agent_max_searches", 0),
        ("coordinator_max_model_calls", 0),
        ("coordinator_max_model_calls", 5),
        ("generation_temperature", -0.01),
        ("generation_temperature", 2.01),
        ("generation_max_tokens", 0),
        ("generation_timeout_seconds", 0),
        ("session_max_turns", 0),
        ("session_max_turns", 101),
        ("upload_max_bytes", 0),
        ("upload_max_pdf_pages", 0),
        ("upload_max_extracted_characters", 0),
        ("upload_max_files_per_message", 0),
        ("upload_max_files_per_message", 11),
        ("upload_max_reports_per_session", 0),
        ("upload_max_reports_per_session", 101),
    ],
)
def test_settings_reject_invalid_retrieval_values(field: str, value: float) -> None:
    """Retrieval limits must stay within their mathematical domains."""
    with pytest.raises(ValidationError):
        AppSettings.model_validate({field: value})
