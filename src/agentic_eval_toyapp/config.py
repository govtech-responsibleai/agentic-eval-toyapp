"""Typed application configuration loaded from environment variables."""

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LogLevel(StrEnum):
    """Supported application log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class GenerationProviderKind(StrEnum):
    """Supported runtime generation-provider implementations."""

    OPENAI = "openai"
    DETERMINISTIC = "deterministic"


class AppSettings(BaseSettings):
    """Validated service settings.

    Only this module reads environment variables. Other components receive a
    settings instance explicitly.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="TOYAPP_",
        case_sensitive=False,
        extra="ignore",
        frozen=True,
        populate_by_name=True,
    )

    app_name: str = "Agentic Eval Toy App"
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: LogLevel = LogLevel.INFO
    embedding_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model_revision: str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    embedding_cache_dir: Path | None = None
    embedding_local_files_only: bool = False
    embedding_device: str = "cpu"
    retrieval_top_k: int = Field(default=3, ge=1, le=8)
    retrieval_min_similarity: float = Field(
        default=0.25,
        ge=-1.0,
        le=1.0,
        allow_inf_nan=False,
    )
    agent_max_model_calls: int = Field(default=4, ge=1)
    agent_max_searches: int = Field(default=2, ge=1)
    generation_provider: GenerationProviderKind = GenerationProviderKind.OPENAI
    generation_model: str = "gpt-5.6-luna"
    generation_model_revision: str = "provider-managed"
    generation_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    generation_max_tokens: int = Field(default=1_024, ge=1, le=16_384)
    generation_timeout_seconds: float = Field(default=30.0, gt=0.0, le=300.0)
    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_KEY", "TOYAPP_OPENAI_API_KEY"),
    )
    openai_base_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_BASE_URL", "TOYAPP_OPENAI_BASE_URL"),
    )
    langfuse_enabled: bool = False
    langfuse_public_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "LANGFUSE_PUBLIC_KEY", "TOYAPP_LANGFUSE_PUBLIC_KEY"
        ),
    )
    langfuse_secret_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "LANGFUSE_SECRET_KEY", "TOYAPP_LANGFUSE_SECRET_KEY"
        ),
    )
    langfuse_base_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("LANGFUSE_BASE_URL", "TOYAPP_LANGFUSE_BASE_URL"),
    )
    langfuse_environment: str = "development"
    langfuse_flush_on_turn: bool = True
    session_max_turns: int = Field(default=10, ge=1, le=100)
    upload_max_bytes: int = Field(default=5 * 1024 * 1024, ge=1, le=50 * 1024 * 1024)
    upload_max_pdf_pages: int = Field(default=50, ge=1, le=1_000)
    upload_max_extracted_characters: int = Field(
        default=100_000,
        ge=1,
        le=1_000_000,
    )
    upload_max_files_per_message: int = Field(default=1, ge=1, le=10)
    upload_max_reports_per_session: int = Field(default=10, ge=1, le=100)

    @field_validator(
        "app_name",
        "host",
        "embedding_model_id",
        "embedding_model_revision",
        "embedding_device",
        "generation_model",
        "generation_model_revision",
        "langfuse_environment",
    )
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        """Reject values that contain only whitespace."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped

    @field_validator("openai_base_url", "langfuse_base_url")
    @classmethod
    def normalise_optional_base_url(cls, value: str | None) -> str | None:
        """Normalise a configured OpenAI-compatible endpoint."""
        if value is None:
            return None
        stripped = value.strip().rstrip("/")
        return stripped or None

    def require_openai_configuration(self) -> tuple[str, str]:
        """Return live credentials or raise a non-secret configuration error."""
        api_key = (
            self.openai_api_key.get_secret_value().strip()
            if self.openai_api_key is not None
            else ""
        )
        missing: list[str] = []
        base_url = self.openai_base_url
        if not api_key:
            missing.append("OPENAI_API_KEY")
        if base_url is None:
            missing.append("OPENAI_BASE_URL")
        if missing:
            raise ValueError(
                f"live generation requires {' and '.join(missing)}; "
                "set them or explicitly select the deterministic provider"
            )
        if base_url is None:  # Defensive; the missing-value branch raises above.
            raise AssertionError("validated base URL is unavailable")
        return api_key, base_url

    def require_langfuse_configuration(self) -> tuple[str, str, str]:
        """Return tracing credentials or raise without exposing their values."""
        public_key = (
            self.langfuse_public_key.get_secret_value().strip()
            if self.langfuse_public_key is not None
            else ""
        )
        secret_key = (
            self.langfuse_secret_key.get_secret_value().strip()
            if self.langfuse_secret_key is not None
            else ""
        )
        base_url = self.langfuse_base_url
        missing: list[str] = []
        if not public_key:
            missing.append("LANGFUSE_PUBLIC_KEY")
        if not secret_key:
            missing.append("LANGFUSE_SECRET_KEY")
        if base_url is None:
            missing.append("LANGFUSE_BASE_URL")
        if missing:
            raise ValueError(
                "Langfuse tracing requires "
                f"{' and '.join(missing)} when TOYAPP_LANGFUSE_ENABLED=true"
            )
        if base_url is None:  # Defensive; the missing-value branch raises above.
            raise AssertionError("validated Langfuse base URL is unavailable")
        return public_key, secret_key, base_url


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Load and cache process-level settings."""
    return AppSettings()
