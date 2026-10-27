"""Run a deterministic end-to-end smoke test against the runtime image."""

import argparse
import json
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast
from uuid import uuid4

REPORT_FIXTURE = Path(__file__).parents[1] / "tests" / "fixtures" / "test_report.md"


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


def _result(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise SmokeTestError("message response is missing its result object")
    result = cast(dict[str, object], payload).get("result")
    if not isinstance(result, dict):
        raise SmokeTestError("message response is missing its result object")
    return cast(dict[str, object], result)


def _actions(result: dict[str, object]) -> list[str]:
    steps = result.get("execution_steps")
    if not isinstance(steps, list):
        raise SmokeTestError("container turn has no execution steps")
    actions: list[str] = []
    for step in cast(list[object], steps):
        if not isinstance(step, dict):
            raise SmokeTestError("container turn has an invalid execution step")
        action = cast(dict[str, object], step).get("action")
        if not isinstance(action, str):
            raise SmokeTestError("container turn has an invalid execution step")
        actions.append(action)
    return actions


def validate_coordinator_policy_turn(payload: object) -> None:
    """The coordinator should return grounded policy evidence."""
    result = _result(payload)
    if result.get("status") != "answered":
        raise SmokeTestError("coordinator policy turn was not answered")
    citations = result.get("citations")
    if not isinstance(citations, list) or "LEAVE-001#1" not in citations:
        raise SmokeTestError("coordinator policy turn missed LEAVE-001#1")
    chunks = result.get("retrieved_chunks")
    if not isinstance(chunks, list) or not chunks:
        raise SmokeTestError("coordinator policy turn returned no evidence")
    if _actions(result) != [
        "inspect_input",
        "route_rag",
        "search",
        "answer",
        "return_response",
    ]:
        raise SmokeTestError("coordinator policy turn used an unexpected path")


def validate_coordinator_analysis_turn(payload: object, *, file_id: str) -> None:
    """A selected report should reach analysis and yield report references."""
    result = _result(payload)
    if result.get("status") != "answered" or result.get("searches") != 0:
        raise SmokeTestError("coordinator analysis turn did not finish as expected")
    references = result.get("report_references")
    if not isinstance(references, list) or not references:
        raise SmokeTestError("coordinator analysis turn returned no report references")
    for reference in cast(list[object], references):
        if (
            not isinstance(reference, dict)
            or cast(dict[str, object], reference).get("file_id") != file_id
        ):
            raise SmokeTestError("coordinator analysis cited a different report")
    if _actions(result) != [
        "inspect_input",
        "route_analysis",
        "inspect_content",
        "answer",
        "return_response",
    ]:
        raise SmokeTestError("coordinator analysis turn used an unexpected path")


def _upload_report(base_url: str, *, session_id: str) -> str:
    """Upload the shared test report without submitting a chat message."""
    boundary = f"toyapp-{uuid4().hex}"
    body = (
        (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="test_report.md"\r\n'
            "Content-Type: text/markdown\r\n\r\n"
        ).encode()
        + REPORT_FIXTURE.read_bytes()
        + f"\r\n--{boundary}--\r\n".encode()
    )
    request = urllib.request.Request(
        f"{base_url}/api/v1/sessions/{session_id}/files",
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(request, timeout=30.0) as response:
        payload: object = json.loads(response.read().decode())
    if not isinstance(payload, dict):
        raise SmokeTestError("report upload returned no file ID")
    file_id = cast(dict[str, object], payload).get("file_id")
    if not isinstance(file_id, str):
        raise SmokeTestError("report upload returned no file ID")
    return file_id


def _create_session(base_url: str) -> str:
    created = _request_json(f"{base_url}/api/v1/sessions", method="POST")
    if not isinstance(created, dict):
        raise SmokeTestError("session endpoint returned no session ID")
    session_id = cast(dict[str, object], created).get("session_id")
    if not isinstance(session_id, str):
        raise SmokeTestError("session endpoint returned no session ID")
    return session_id


def _check_browser(base_url: str) -> None:
    with urllib.request.urlopen(base_url, timeout=5.0) as response:
        html = response.read().decode()
    if not all(marker in html for marker in ('id="chat-form"', 'id="files-panel"')):
        raise SmokeTestError("container did not serve the browser chat shell")
    with urllib.request.urlopen(f"{base_url}/static/app.js", timeout=5.0) as response:
        script = response.read().decode()
    if "selectedFileId" not in script:
        raise SmokeTestError("container did not serve the browser client")


def _exercise_rag(base_url: str) -> str:
    _check_browser(base_url)
    session_id = _create_session(base_url)
    turn = _request_json(
        f"{base_url}/api/v1/sessions/{session_id}/messages",
        method="POST",
        payload={"message": "How many annual leave days do employees receive?"},
        timeout_seconds=30.0,
    )
    validate_smoke_turn(turn)
    return session_id


def _exercise_coordinator(base_url: str) -> str:
    _check_browser(base_url)
    session_id = _create_session(base_url)
    file_id = _upload_report(base_url, session_id=session_id)
    snapshot = _request_json(f"{base_url}/api/v1/sessions/{session_id}")
    if (
        not isinstance(snapshot, dict)
        or cast(dict[str, object], snapshot).get("messages") != []
    ):
        raise SmokeTestError("upload alone created an agent turn")
    listed = _request_json(f"{base_url}/api/v1/sessions/{session_id}/files")
    files = (
        cast(dict[str, object], listed).get("files")
        if isinstance(listed, dict)
        else None
    )
    if not isinstance(files, list) or len(cast(list[object], files)) != 1:
        raise SmokeTestError("uploaded report was not listed")
    analysed = _request_json(
        f"{base_url}/api/v1/sessions/{session_id}/messages",
        method="POST",
        payload={
            "message": "Summarise the main findings in the selected report.",
            "file_ids": [file_id],
        },
        timeout_seconds=30.0,
    )
    validate_coordinator_analysis_turn(analysed, file_id=file_id)
    policy = _request_json(
        f"{base_url}/api/v1/sessions/{session_id}/messages",
        method="POST",
        payload={
            "message": "How many annual leave days do employees receive?",
            "file_ids": [],
        },
        timeout_seconds=30.0,
    )
    validate_coordinator_policy_turn(policy)
    request = urllib.request.Request(
        f"{base_url}/api/v1/sessions/{session_id}/files/{file_id}", method="DELETE"
    )
    with urllib.request.urlopen(request, timeout=5.0) as response:
        if response.status != 204:
            raise SmokeTestError("report deletion did not return 204")
    listed = _request_json(f"{base_url}/api/v1/sessions/{session_id}/files")
    if (
        not isinstance(listed, dict)
        or cast(dict[str, object], listed).get("files") != []
    ):
        raise SmokeTestError("deleted report remained in the session")
    return session_id


def _verify_restart(base_url: str, *, old_session_id: str) -> str:
    """A new container must not inherit previous process-local state."""
    for suffix in ("", "/files"):
        try:
            _request_json(f"{base_url}/api/v1/sessions/{old_session_id}{suffix}")
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise SmokeTestError(
                    "restart returned an unexpected API error"
                ) from error
        else:
            raise SmokeTestError("process-local session survived container restart")
    return _create_session(base_url)


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


def _run_container(
    *,
    image: str,
    host_port: int,
    timeout_seconds: float,
    mode: str,
    exercise: Callable[[str], str],
) -> str:
    """Exercise one isolated deterministic runtime and remove it on exit."""
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
                f"TOYAPP_AGENT_MODE={mode}",
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
        return exercise(base_url)
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


def run_smoke_test(*, image: str, host_port: int, timeout_seconds: float) -> None:
    """Verify standalone RAG, coordinator API, browser assets, and restart."""
    _run_container(
        image=image,
        host_port=host_port,
        timeout_seconds=timeout_seconds,
        mode="rag",
        exercise=_exercise_rag,
    )
    old_session_id = _run_container(
        image=image,
        host_port=host_port,
        timeout_seconds=timeout_seconds,
        mode="coordinator",
        exercise=_exercise_coordinator,
    )
    _run_container(
        image=image,
        host_port=host_port,
        timeout_seconds=timeout_seconds,
        mode="coordinator",
        exercise=lambda base_url: _verify_restart(
            base_url, old_session_id=old_session_id
        ),
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
