"""Tests for deterministic image-build and container-smoke utilities."""

from pathlib import Path

import pytest
from scripts.cache_embedding_model import cache_model
from scripts.docker_smoke import SmokeTestError, validate_smoke_turn


def valid_turn_payload() -> dict[str, object]:
    """Return the smallest valid deterministic container response."""
    return {
        "session_id": "62a0dcb7-fcd9-4bea-94fb-1b97bd15057d",
        "result": {
            "status": "answered",
            "citations": ["LEAVE-001#1"],
            "retrieved_chunks": [{"chunk_id": "LEAVE-001#1"}],
            "model_calls": 2,
            "searches": 1,
        },
    }


def test_validate_smoke_turn_accepts_supported_retrieval_path() -> None:
    """The image smoke path should accept the expected real-RAG contract."""
    validate_smoke_turn(valid_turn_payload())


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"result": "not-an-object"},
        {"result": {"status": "failed"}},
        {
            "result": {
                "status": "answered",
                "citations": [],
                "retrieved_chunks": [{"chunk_id": "LEAVE-001#1"}],
                "model_calls": 2,
                "searches": 1,
            }
        },
        {
            "result": {
                "status": "answered",
                "citations": ["LEAVE-001#1"],
                "retrieved_chunks": [],
                "model_calls": 2,
                "searches": 1,
            }
        },
        {
            "result": {
                "status": "answered",
                "citations": ["LEAVE-001#1"],
                "retrieved_chunks": [{"chunk_id": "LEAVE-001#1"}],
                "model_calls": 1,
                "searches": 1,
            }
        },
    ],
)
def test_validate_smoke_turn_rejects_incomplete_paths(payload: object) -> None:
    """Malformed, failed, ungrounded, or shortened turns must fail the smoke test."""
    with pytest.raises(SmokeTestError):
        validate_smoke_turn(payload)


@pytest.mark.parametrize(
    ("model_id", "revision"),
    [("", "revision"), ("model", ""), ("   ", "revision")],
)
def test_cache_model_rejects_blank_identity_before_importing_model(
    model_id: str,
    revision: str,
    tmp_path: Path,
) -> None:
    """Bad build arguments should fail before a network-capable model import."""
    with pytest.raises(ValueError, match="must not be blank"):
        cache_model(model_id=model_id, revision=revision, cache_dir=tmp_path)
