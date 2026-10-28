"""Tests for the live-response substitution seam and trace isolation."""

import json
from collections.abc import Generator
from contextlib import contextmanager

import pytest
from trace_cases.controlled_client import ControlledStructuredCompletionClient

from agentic_eval_toyapp.diagnostics import CompletionDiagnostics
from agentic_eval_toyapp.providers import StructuredCompletion


class _StaticCompletionClient:
    def __init__(
        self,
        response: str | StructuredCompletion | None = None,
        error: Exception | None = None,
    ):
        self._response = response
        self._error = error
        self.calls = 0

    async def create(self, **values: object) -> str | StructuredCompletion:
        del values
        self.calls += 1
        if self._error is not None:
            raise self._error
        if self._response is None:
            raise AssertionError("test response is missing")
        return self._response


class _RecordingObservation:
    def __init__(self) -> None:
        self.updates: list[dict[str, object]] = []

    def update(self, **values: object) -> object:
        self.updates.append(values)
        return self


class _RecordingTraceClient:
    def __init__(self) -> None:
        self.starts: list[dict[str, object]] = []
        self.observations: list[_RecordingObservation] = []

    @contextmanager
    def start_as_current_observation(
        self, **values: object
    ) -> Generator[_RecordingObservation, None, None]:
        self.starts.append(values)
        observation = _RecordingObservation()
        self.observations.append(observation)
        yield observation


async def _create(client: ControlledStructuredCompletionClient) -> str:
    return await client.create(
        model="example-model",
        messages=(
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "User input"},
        ),
        response_schema={"type": "object"},
        temperature=0.0,
        max_tokens=128,
    )


@pytest.mark.anyio
async def test_answer_is_substituted_before_the_effective_generation_is_traced() -> (
    None
):
    """Only the effective answer should appear in observable generation output."""
    original = json.dumps(
        {
            "kind": "answer",
            "query": None,
            "top_k": None,
            "question": None,
            "markdown": "Original live answer",
            "citations": ["UNSUPPORTED#1"],
        }
    )
    delegate = _StaticCompletionClient(response=original)
    trace_client = _RecordingTraceClient()
    client = ControlledStructuredCompletionClient(
        delegate=delegate,
        trace_client=trace_client,
        replacement_markdown="Controlled effective answer",
    )

    effective = await _create(client)

    assert json.loads(effective)["markdown"] == "Controlled effective answer"
    assert json.loads(effective)["citations"] == []
    assert client.captures[0].original_response == original
    assert client.captures[0].effective_response == effective
    assert client.captures[0].altered is True
    assert trace_client.observations[0].updates == [
        {"output": {"role": "assistant", "content": effective}}
    ]
    assert trace_client.starts[0]["metadata"] == {
        "operation": "structured_action_selection",
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "agent_action",
                "schema": {"type": "object"},
                "strict": True,
            },
        },
    }
    assert trace_client.starts[0]["model_parameters"] == {
        "temperature": 0.0,
        "max_tokens": 128,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "service_tier": "default",
    }
    serialized_trace = json.dumps(
        [trace_client.starts, trace_client.observations[0].updates]
    )
    for forbidden in (
        "harmful_content_refusal",
        "case-0001",
        "case-0002",
        "control",
        "defect",
        "expected_pass",
        "Original live answer",
    ):
        assert forbidden not in serialized_trace


@pytest.mark.anyio
async def test_controlled_client_accepts_metadata_bearing_live_response() -> None:
    """The private capture stays textual after live transport gains metadata."""
    original = json.dumps(
        {
            "kind": "answer",
            "query": None,
            "top_k": None,
            "question": None,
            "markdown": "Original answer",
            "citations": [],
        }
    )
    client = ControlledStructuredCompletionClient(
        delegate=_StaticCompletionClient(
            StructuredCompletion(
                original,
                CompletionDiagnostics(
                    finish_reason="stop", input_tokens=10, output_tokens=5
                ),
            )
        ),
        trace_client=_RecordingTraceClient(),
        replacement_markdown="Effective answer",
    )

    assert json.loads(await _create(client))["markdown"] == "Effective answer"
    assert client.captures[0].original_response == original


@pytest.mark.anyio
@pytest.mark.parametrize(
    "response",
    [
        "not-json",
        json.dumps(
            {
                "kind": "search",
                "query": "policy search",
                "top_k": 3,
                "question": None,
                "markdown": None,
                "citations": [],
            }
        ),
        json.dumps(
            {
                "kind": "clarify",
                "query": None,
                "top_k": None,
                "question": "What do you mean?",
                "markdown": None,
                "citations": [],
            }
        ),
    ],
)
async def test_non_answer_responses_pass_through_unchanged(response: str) -> None:
    """Search, clarification, and invalid responses must retain normal behavior."""
    client = ControlledStructuredCompletionClient(
        delegate=_StaticCompletionClient(response=response),
        trace_client=_RecordingTraceClient(),
        replacement_markdown="Controlled effective answer",
    )

    assert await _create(client) == response
    assert len(client.captures) == 1
    assert client.captures[0].original_response == response
    assert client.captures[0].effective_response == response
    assert client.captures[0].altered is False


@pytest.mark.anyio
async def test_provider_failure_is_recorded_without_private_error_text() -> None:
    """A failed live call should retain only the normal safe trace category."""
    trace_client = _RecordingTraceClient()
    client = ControlledStructuredCompletionClient(
        delegate=_StaticCompletionClient(error=RuntimeError("private upstream detail")),
        trace_client=trace_client,
        replacement_markdown="Controlled effective answer",
    )

    with pytest.raises(RuntimeError, match="private upstream detail"):
        await _create(client)

    assert client.captures == ()
    assert trace_client.observations[0].updates == [
        {
            "level": "ERROR",
            "status_message": "generation_provider_error",
        }
    ]


def test_blank_replacement_is_rejected() -> None:
    """A controlled case cannot produce an empty public answer."""
    with pytest.raises(ValueError, match="must not be blank"):
        ControlledStructuredCompletionClient(
            delegate=_StaticCompletionClient(response="{}"),
            trace_client=_RecordingTraceClient(),
            replacement_markdown="   ",
        )
