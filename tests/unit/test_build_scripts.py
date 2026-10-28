"""Tests for deterministic image-build and container-smoke utilities."""

from pathlib import Path
from typing import cast

import pytest
from scripts.cache_embedding_model import cache_model
from scripts.docker_smoke import (
    SmokeTestError,
    validate_coordinator_analysis_turn,
    validate_coordinator_policy_turn,
    validate_smoke_turn,
)


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


def _coordinator_payload(*, action: str) -> dict[str, object]:
    return {
        "result": {
            "status": "answered",
            "searches": 1 if action == "route_rag" else 0,
            "citations": ["LEAVE-001#1"] if action == "route_rag" else [],
            "retrieved_chunks": [{"chunk_id": "LEAVE-001#1"}]
            if action == "route_rag"
            else [],
            "report_references": [{"file_id": "report-id"}]
            if action == "route_analysis"
            else [],
            "execution_steps": [
                {"action": name}
                for name in (
                    [
                        "inspect_input",
                        "route_rag",
                        "search",
                        "answer",
                        "return_response",
                    ]
                    if action == "route_rag"
                    else [
                        "inspect_input",
                        "route_analysis",
                        "inspect_content",
                        "answer",
                        "return_response",
                    ]
                )
            ],
        }
    }


def test_container_smoke_accepts_coordinator_rag_and_analysis_paths() -> None:
    validate_coordinator_policy_turn(_coordinator_payload(action="route_rag"))
    validate_coordinator_analysis_turn(
        _coordinator_payload(action="route_analysis"), file_id="report-id"
    )


@pytest.mark.parametrize(
    ("action", "bad_field", "bad_value"),
    [
        ("route_rag", "citations", []),
        ("route_rag", "retrieved_chunks", []),
        ("route_rag", "execution_steps", []),
        ("route_analysis", "report_references", []),
        ("route_analysis", "searches", 1),
        ("route_analysis", "execution_steps", []),
    ],
)
def test_container_smoke_rejects_incomplete_coordinator_paths(
    action: str, bad_field: str, bad_value: object
) -> None:
    payload = _coordinator_payload(action=action)
    result = cast(dict[str, object], payload["result"])
    result[bad_field] = bad_value
    with pytest.raises(SmokeTestError):
        if action == "route_rag":
            validate_coordinator_policy_turn(payload)
        else:
            validate_coordinator_analysis_turn(payload, file_id="report-id")


def test_analysis_smoke_rejects_reference_to_another_report() -> None:
    with pytest.raises(SmokeTestError, match="different report"):
        validate_coordinator_analysis_turn(
            _coordinator_payload(action="route_analysis"), file_id="other-report"
        )


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
