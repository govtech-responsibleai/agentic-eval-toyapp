"""Run a deterministic end-to-end smoke test against the runtime image."""

import argparse
import json
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import cast
from uuid import uuid4


class SmokeTestError(RuntimeError):
    """Raised when the container does not satisfy its runtime contract."""


def _request_json(
    url: str,
    *,
    method: str = "GET",
    payload: Mapping[str, object] | None = None,
    timeout_seconds: float = 5.0,
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


def validate_smoke_turn(payload: object) -> None:
    """Validate the minimum deterministic RAG result required from the image."""
    if not isinstance(payload, dict):
        raise SmokeTestError("message response must be a JSON object")
    response = cast(dict[str, object], payload)
    result = response.get("result")
    if not isinstance(result, dict):
        raise SmokeTestError("message response is missing its result object")
    typed_result = cast(dict[str, object], result)
    if typed_result.get("status") != "answered":
        raise SmokeTestError("deterministic container turn was not answered")
    citations = typed_result.get("citations")
    if not isinstance(citations, list) or "LEAVE-001#1" not in citations:
        raise SmokeTestError("deterministic container turn missed LEAVE-001#1")
    chunks = typed_result.get("retrieved_chunks")
    if not isinstance(chunks, list) or not chunks:
        raise SmokeTestError("deterministic container turn returned no evidence")
    if typed_result.get("model_calls") != 2 or typed_result.get("searches") != 1:
        raise SmokeTestError("deterministic container turn used an unexpected path")


def wait_until_ready(base_url: str, *, timeout_seconds: float) -> None:
    """Wait until startup has loaded the model and built the FAISS index."""
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            payload = _request_json(f"{base_url}/ready")
            if isinstance(payload, dict):
                readiness = cast(dict[str, object], payload)
                if readiness.get("status") == "ready":
                    return
        except (OSError, ValueError, urllib.error.URLError) as error:
            last_error = error
        time.sleep(1.0)
    message = "container did not become ready before the smoke-test timeout"
    if last_error is not None:
        raise SmokeTestError(message) from last_error
    raise SmokeTestError(message)


def run_smoke_test(*, image: str, host_port: int, timeout_seconds: float) -> None:
    """Start the image, exercise one real retrieval turn, and remove it."""
    container_name = f"agentic-eval-toyapp-smoke-{uuid4().hex[:12]}"
    base_url = f"http://127.0.0.1:{host_port}"
    started = False
    try:
        subprocess.run(
            [
                "docker",
                "run",
                "--detach",
                "--rm",
                "--name",
                container_name,
                "--publish",
                f"127.0.0.1:{host_port}:8000",
                "--env",
                "TOYAPP_GENERATION_PROVIDER=deterministic",
                "--env",
                "TOYAPP_LANGFUSE_ENABLED=false",
                image,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        started = True
        wait_until_ready(base_url, timeout_seconds=timeout_seconds)

        created = _request_json(f"{base_url}/api/v1/sessions", method="POST")
        if not isinstance(created, dict):
            raise SmokeTestError("session endpoint returned no session ID")
        session_response = cast(dict[str, object], created)
        session_id = session_response.get("session_id")
        if not isinstance(session_id, str):
            raise SmokeTestError("session endpoint returned no session ID")
        turn = _request_json(
            f"{base_url}/api/v1/sessions/{session_id}/messages",
            method="POST",
            payload={"message": "How many annual leave days do employees receive?"},
            timeout_seconds=30.0,
        )
        validate_smoke_turn(turn)
    except Exception:
        if started:
            subprocess.run(
                ["docker", "logs", container_name],
                check=False,
            )
        raise
    finally:
        if started:
            subprocess.run(
                ["docker", "rm", "--force", container_name],
                check=False,
                capture_output=True,
                text=True,
            )


def build_parser() -> argparse.ArgumentParser:
    """Return smoke-test command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--host-port", type=int, default=18000)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    return parser


def main() -> None:
    """Execute the container smoke test."""
    arguments = build_parser().parse_args()
    if not 1 <= arguments.host_port <= 65_535:
        raise ValueError("host port must be between 1 and 65535")
    if arguments.timeout_seconds <= 0:
        raise ValueError("timeout must be positive")
    run_smoke_test(
        image=arguments.image,
        host_port=arguments.host_port,
        timeout_seconds=arguments.timeout_seconds,
    )


if __name__ == "__main__":
    main()
