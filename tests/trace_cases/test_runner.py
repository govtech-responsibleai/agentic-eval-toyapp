"""API and local-manifest tests for the isolated trace-case harness."""

import json
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import pytest
from trace_cases.catalog import load_metric_bundle
from trace_cases.controlled_client import ControlledStructuredCompletionClient
from trace_cases.models import CaseVariant, TraceCaseRun
from trace_cases.runner import (
    DEFAULT_RUN_DIRECTORY,
    TraceCaseExecutionError,
    execute_api_turn,
    write_run_manifest,
)

from agentic_eval_toyapp.agent import RagAgent
from agentic_eval_toyapp.app import create_app
from agentic_eval_toyapp.config import AppSettings, GenerationProviderKind
from agentic_eval_toyapp.models import ModelRequest, RetrievedChunk
from agentic_eval_toyapp.providers import OpenAIChatGenerationProvider


class _EmptyRetriever:
    def search(
        self, query: str, *, top_k: int | None = None
    ) -> tuple[RetrievedChunk, ...]:
        del query, top_k
        return ()


class _AnswerCompletionClient:
    def create(self, **values: object) -> str:
        del values
        return json.dumps(
            {
                "kind": "answer",
                "query": None,
                "top_k": None,
                "question": None,
                "markdown": "Original live refusal",
                "citations": [],
            }
        )


class _Observation:
    def __init__(self) -> None:
        self.updates: list[dict[str, object]] = []

    def update(self, **values: object) -> object:
        self.updates.append(values)
        return self


class _TraceClient:
    def __init__(self) -> None:
        self.starts: list[dict[str, object]] = []
        self.observation = _Observation()

    @contextmanager
    def start_as_current_observation(
        self, **values: object
    ) -> Generator[_Observation, None, None]:
        self.starts.append(values)
        yield self.observation


def test_default_run_directory_is_inside_the_trace_case_harness() -> None:
    """Generated manifests should stay grouped under the isolated harness."""
    assert Path("trace_cases/runs") == DEFAULT_RUN_DIRECTORY


@pytest.mark.parametrize("variant", tuple(CaseVariant))
def test_controlled_case_exercises_the_normal_session_message_api(
    variant: CaseVariant,
) -> None:
    """Both variants must use the public API and typed agent processing."""
    bundle = load_metric_bundle("harmful_content_refusal")
    case = next(case for case in bundle.cases.cases if case.variant is variant)
    trace_client = _TraceClient()
    controlled_client = ControlledStructuredCompletionClient(
        delegate=_AnswerCompletionClient(),
        trace_client=trace_client,
        replacement_markdown=case.replacement_markdown,
    )
    provider = OpenAIChatGenerationProvider(
        api_key="unused-test-key",
        base_url="https://provider.invalid/v1",
        model="test-model",
        revision="test-revision",
        temperature=0.0,
        max_tokens=128,
        timeout_seconds=1.0,
        client=controlled_client,
    )
    agent = RagAgent(
        provider=provider,
        retriever=_EmptyRetriever(),
        system_prompt="Return one structured action.",
    )
    settings = AppSettings(
        generation_provider=GenerationProviderKind.DETERMINISTIC,
        langfuse_enabled=False,
    )

    response = execute_api_turn(create_app(settings, agent=agent), case)

    assert response.result.message == case.replacement_markdown
    assert response.result.citations == ()
    assert response.result.model_calls == 1
    assert response.result.searches == 0
    serialized_observation = json.dumps(
        [trace_client.starts, trace_client.observation.updates]
    )
    assert bundle.cases.metric not in serialized_observation
    assert case.case_id not in serialized_observation
    assert case.variant.value not in serialized_observation


def test_run_manifest_retains_private_ground_truth_outside_trace(
    tmp_path: Path,
) -> None:
    """The session-to-label mapping should be local, complete, and atomic."""
    bundle = load_metric_bundle("harmful_content_refusal")
    case = next(
        case for case in bundle.cases.cases if case.variant is CaseVariant.CONTROL
    )
    label = next(
        label for label in bundle.ground_truth.labels if label.case_id == case.case_id
    )
    controlled_client = ControlledStructuredCompletionClient(
        delegate=_AnswerCompletionClient(),
        trace_client=_TraceClient(),
        replacement_markdown=case.replacement_markdown,
    )
    provider = OpenAIChatGenerationProvider(
        api_key="unused-test-key",
        base_url="https://provider.invalid/v1",
        model="test-model",
        revision="test-revision",
        temperature=0.0,
        max_tokens=128,
        timeout_seconds=1.0,
        client=controlled_client,
    )
    response = execute_api_turn(
        create_app(
            AppSettings(
                generation_provider=GenerationProviderKind.DETERMINISTIC,
                langfuse_enabled=False,
            ),
            agent=RagAgent(
                provider=provider,
                retriever=_EmptyRetriever(),
                system_prompt="Return one structured action.",
            ),
        ),
        case,
    )
    run = TraceCaseRun(
        generated_at=datetime.now(UTC),
        metric=bundle.cases.metric,
        case_id=case.case_id,
        variant=case.variant,
        expected_pass=label.expected_pass,
        expected_rationale=label.rationale,
        session_id=response.session_id,
        user_message=case.user_message,
        controlled_response_applied=True,
        result=response.result,
        model_completions=controlled_client.captures,
    )

    path = write_run_manifest(run, tmp_path)
    restored = TraceCaseRun.model_validate_json(path.read_text(encoding="utf-8"))

    assert path == tmp_path / f"{run.session_id}.json"
    assert restored == run
    assert list(tmp_path.glob("*.tmp")) == []

    with pytest.raises(TraceCaseExecutionError, match="already exists for session"):
        write_run_manifest(run, tmp_path)


def test_generation_provider_protocol_is_unchanged_for_trace_cases() -> None:
    """The harness should depend on the existing provider contract only."""
    request_annotations = ModelRequest.model_fields
    assert "user_message" in request_annotations
    assert "retrieved_chunks" in request_annotations
