"""Unit tests for deterministic load-test aggregation and validation."""

import pytest
from scripts.load_test import UserResult, normalise_base_url, percentile, summarise


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
        }
    ]


@pytest.mark.parametrize("value", ["example.com", "ftp://example.com", "   "])
def test_base_url_rejects_unsupported_values(value: str) -> None:
    with pytest.raises(ValueError, match="http"):
        normalise_base_url(value)
