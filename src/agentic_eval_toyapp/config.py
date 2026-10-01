"""Typed application configuration loaded from environment variables."""

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LogLevel(StrEnum):
    """Supported application log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


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
    )

    app_name: str = "Agentic Eval Toy App"
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: LogLevel = LogLevel.INFO
    embedding_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model_revision: str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    embedding_cache_dir: Path | None = None
    embedding_device: str = "cpu"
    retrieval_top_k: int = Field(default=3, ge=1)
    retrieval_min_similarity: float = Field(
        default=0.25,
        ge=-1.0,
        le=1.0,
        allow_inf_nan=False,
    )
    agent_max_model_calls: int = Field(default=3, ge=1)
    agent_max_searches: int = Field(default=2, ge=1)

    @field_validator(
        "app_name",
        "host",
        "embedding_model_id",
        "embedding_model_revision",
        "embedding_device",
    )
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        """Reject values that contain only whitespace."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Load and cache process-level settings."""
    return AppSettings()
