"""Unit tests for deterministic load-test aggregation and validation."""

import pytest
from scripts.load_test import (
    USER_AGENT,
    UserResult,
    build_request,
    check_turn_result,
    normalise_base_url,
    percentile,
    summarise,
)


def test_percentile_uses_nearest_rank() -> None:
    assert percentile([4.0, 1.0, 3.0, 2.0], 0.50) == 2.0
    assert percentile([4.0, 1.0, 3.0, 2.0], 0.95) == 4.0


def test_summary_reports_failures_and_latency_distribution() -> None:
    summary = summarise(
        [
            UserResult(user=1, succeeded=True, latency_seconds=0.2),
            UserResult(
                user=2,
                succeeded=False,
                latency_seconds=0.4,
                error="timeout",
            ),
        ],
        elapsed_seconds=0.5,
    )

    assert summary["users"] == 2
    assert summary["succeeded"] == 1
    assert summary["failed"] == 1
    assert summary["throughput_users_per_second"] == 4.0
    assert summary["latency_seconds"] == {
        "min": 0.2,
        "p50": 0.2,
        "p95": 0.4,
        "p99": 0.4,
        "max": 0.4,
    }
    assert summary["failures"] == [
        {
            "user": 2,
            "succeeded": False,
            "latency_seconds": 0.4,
            "error": "timeout",
            "status": None,
        }
    ]


@pytest.mark.parametrize("value", ["example.com", "ftp://example.com", "   "])
def test_base_url_rejects_unsupported_values(value: str) -> None:
    with pytest.raises(ValueError, match="http"):
        normalise_base_url(value)


def test_requests_identify_the_load_test_client() -> None:
    request = build_request(
        "https://example.test/api/v1/sessions", method="POST", payload={"a": 1}
    )

    assert request.get_header("User-agent") == USER_AGENT
    assert not USER_AGENT.startswith("Python-urllib")
    assert request.get_header("Content-type") == "application/json"
    assert request.data == b'{"a": 1}'


def test_requests_without_payload_send_no_body() -> None:
    request = build_request("https://example.test/health")

    assert request.data is None
    assert request.get_method() == "GET"
    assert request.get_header("User-agent") == USER_AGENT


@pytest.mark.parametrize("status", ["answered", "clarification_required"])
def test_completed_turn_statuses_count_as_success(status: str) -> None:
    assert check_turn_result({"status": status}) == status


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (
            {"status": "failed", "termination_reason": "provider_error"},
            "provider_error",
        ),
        ({"status": "unknown"}, "status=unknown"),
        ({}, "status=None"),
        ({"status": 1}, "status=1"),
    ],
)
def test_other_turn_statuses_fail_with_reason(
    result: dict[str, object], expected: str
) -> None:
    with pytest.raises(ValueError, match=expected):
        check_turn_result(result)


def test_summary_counts_completed_statuses() -> None:
    summary = summarise(
        [
            UserResult(user=1, succeeded=True, latency_seconds=1.0, status="answered"),
            UserResult(
                user=2,
                succeeded=True,
                latency_seconds=1.0,
                status="clarification_required",
            ),
            UserResult(user=3, succeeded=True, latency_seconds=1.0, status="answered"),
            UserResult(user=4, succeeded=False, latency_seconds=1.0, error="403"),
        ],
        elapsed_seconds=2.0,
    )

    assert summary["succeeded"] == 3
    assert summary["failed"] == 1
    assert summary["statuses"] == {"answered": 2, "clarification_required": 1}
