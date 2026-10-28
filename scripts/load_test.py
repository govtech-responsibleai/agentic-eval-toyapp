"""Run concurrent user flows against a deployed toy app."""

import argparse
import json
import math
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from threading import Barrier
from typing import cast


@dataclass(frozen=True)
class UserResult:
    """Outcome and end-to-end latency for one isolated user flow."""

    user: int
    succeeded: bool
    latency_seconds: float
    error: str | None = None


def _request_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, object] | None = None,
    timeout_seconds: float,
) -> object:
    encoded = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url,
        data=encoded,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode())


def normalise_base_url(value: str) -> str:
    """Validate and normalise an HTTP(S) deployment URL."""
    base_url = value.strip().rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        raise ValueError("base URL must start with http:// or https://")
    return base_url


def _run_user(
    user: int,
    *,
    barrier: Barrier,
    base_url: str,
    message: str,
    timeout_seconds: float,
) -> UserResult:
    barrier.wait()
    started = time.perf_counter()
    try:
        created = _request_json(
            f"{base_url}/api/v1/sessions",
            method="POST",
            timeout_seconds=timeout_seconds,
        )
        if not isinstance(created, dict):
            raise ValueError("session response was not an object")
        session_id = cast(dict[str, object], created).get("session_id")
        if not isinstance(session_id, str):
            raise ValueError("session response did not contain a session ID")

        response = _request_json(
            f"{base_url}/api/v1/sessions/{session_id}/messages",
            method="POST",
            payload={"message": message},
            timeout_seconds=timeout_seconds,
        )
        if not isinstance(response, dict):
            raise ValueError("message response was not an object")
        result = cast(dict[str, object], response).get("result")
        if not isinstance(result, dict):
            raise ValueError("message response did not contain a result object")
        typed_result = cast(dict[str, object], result)
        if typed_result.get("status") != "answered":
            raise ValueError("message response was not answered")
        return UserResult(
            user=user,
            succeeded=True,
            latency_seconds=time.perf_counter() - started,
        )
    except (OSError, ValueError, urllib.error.URLError) as error:
        return UserResult(
            user=user,
            succeeded=False,
            latency_seconds=time.perf_counter() - started,
            error=f"{type(error).__name__}: {error}",
        )


def percentile(values: list[float], percentile_value: float) -> float:
    """Return the nearest-rank percentile for a non-empty sample."""
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile_value * len(ordered)))
    return ordered[rank - 1]


def summarise(
    results: list[UserResult], *, elapsed_seconds: float
) -> dict[str, object]:
    """Build stable aggregate evidence from individual user results."""
    if not results:
        raise ValueError("summary requires at least one result")
    if elapsed_seconds <= 0:
        raise ValueError("elapsed time must be positive")
    latencies = [result.latency_seconds for result in results]
    succeeded = sum(result.succeeded for result in results)
    return {
        "users": len(results),
        "succeeded": succeeded,
        "failed": len(results) - succeeded,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "throughput_users_per_second": round(len(results) / elapsed_seconds, 3),
        "latency_seconds": {
            "min": round(min(latencies), 3),
            "p50": round(percentile(latencies, 0.50), 3),
            "p95": round(percentile(latencies, 0.95), 3),
            "p99": round(percentile(latencies, 0.99), 3),
            "max": round(max(latencies), 3),
        },
        "failures": [asdict(result) for result in results if not result.succeeded],
    }


def run_load_test(
    *,
    base_url: str,
    users: int,
    message: str,
    timeout_seconds: float,
) -> dict[str, object]:
    """Run one isolated session and message flow per simultaneous user."""
    if users < 1:
        raise ValueError("users must be positive")
    if timeout_seconds <= 0:
        raise ValueError("timeout must be positive")
    if not message.strip():
        raise ValueError("message must not be blank")

    normalised_url = normalise_base_url(base_url)
    barrier = Barrier(users)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=users) as executor:
        futures = [
            executor.submit(
                _run_user,
                user,
                barrier=barrier,
                base_url=normalised_url,
                message=message,
                timeout_seconds=timeout_seconds,
            )
            for user in range(1, users + 1)
        ]
        results = [future.result() for future in futures]
    elapsed_seconds = time.perf_counter() - started
    return summarise(results, elapsed_seconds=elapsed_seconds)


def build_parser() -> argparse.ArgumentParser:
    """Return command-line arguments for a load-test run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--users", type=int, default=60)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument(
        "--message",
        default="If I have been working for 67 days, how many days of annual leave am I entitled to?",
    )
    return parser


def main() -> None:
    """Execute the load test and fail when any user flow fails."""
    arguments = build_parser().parse_args()
    summary = run_load_test(
        base_url=arguments.base_url,
        users=arguments.users,
        message=arguments.message,
        timeout_seconds=arguments.timeout_seconds,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
