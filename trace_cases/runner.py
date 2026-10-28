"""Generate realistic Langfuse traces through the normal toy-app API."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from uuid import UUID

from anyio import CapacityLimiter
from fastapi import FastAPI
from langfuse import Langfuse
from starlette.testclient import TestClient

from agentic_eval_toyapp import __version__
from agentic_eval_toyapp.agent import RagAgent, load_system_prompt
from agentic_eval_toyapp.app import create_app
from agentic_eval_toyapp.backpressure import RateLimitedCompletionClient
from agentic_eval_toyapp.config import AgentMode, AppSettings, GenerationProviderKind
from agentic_eval_toyapp.embeddings import SentenceTransformerEmbeddingProvider
from agentic_eval_toyapp.models import AnswerAction, SessionMessageResponse
from agentic_eval_toyapp.providers import (
    OpenAIChatGenerationProvider,
    OpenAICompatibleChatClient,
)
from agentic_eval_toyapp.retrieval import AsyncRetriever, FaissRetriever, load_corpus
from agentic_eval_toyapp.tracing import LangfuseTraceRecorder
from trace_cases.catalog import load_metric_bundle
from trace_cases.controlled_client import (
    ControlledStructuredCompletionClient,
    GenerationTraceClient,
)
from trace_cases.models import (
    CaseVariant,
    ControlledCase,
    GroundTruthLabel,
    TraceCaseRun,
)

DEFAULT_RUN_DIRECTORY = Path("trace_cases/runs")


class TraceCaseExecutionError(RuntimeError):
    """Raised when a live run cannot produce the controlled trace contract."""


def execute_api_turn(
    application: FastAPI, case: ControlledCase
) -> SessionMessageResponse:
    """Exercise the normal session and message routes for one controlled case."""
    with TestClient(application) as client:
        session_response = client.post("/api/v1/sessions")
        if session_response.status_code != 201:
            raise TraceCaseExecutionError("the API could not create a session")
        session_id = session_response.json().get("session_id")
        if not isinstance(session_id, str):
            raise TraceCaseExecutionError("the API returned no session ID")
        turn_response = client.post(
            f"/api/v1/sessions/{session_id}/messages",
            json={"message": case.user_message},
        )
        if turn_response.status_code != 200:
            raise TraceCaseExecutionError("the API could not complete the trace case")
        return SessionMessageResponse.model_validate(turn_response.json())


def _build_live_application(
    settings: AppSettings,
    case: ControlledCase,
) -> tuple[FastAPI, ControlledStructuredCompletionClient]:
    """Build a case-scoped app while keeping the normal app independent."""
    if settings.generation_provider is not GenerationProviderKind.OPENAI:
        raise TraceCaseExecutionError(
            "trace cases require TOYAPP_GENERATION_PROVIDER=openai"
        )
    if not settings.langfuse_enabled:
        raise TraceCaseExecutionError(
            "trace cases require TOYAPP_LANGFUSE_ENABLED=true"
        )
    api_key, openai_base_url = settings.require_openai_configuration()
    public_key, secret_key, langfuse_base_url = (
        settings.require_langfuse_configuration()
    )
    langfuse_client = Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        base_url=langfuse_base_url,
        environment=settings.langfuse_environment,
        release=__version__,
    )
    raw_client = RateLimitedCompletionClient(
        OpenAICompatibleChatClient(
            api_key=api_key,
            base_url=openai_base_url,
            timeout_seconds=settings.generation_timeout_seconds,
            instrument_with_langfuse=False,
        ),
        max_requests_per_minute=settings.llm_max_requests_per_minute,
        max_concurrent_calls=settings.llm_max_concurrent_calls,
    )
    controlled_client = ControlledStructuredCompletionClient(
        delegate=raw_client,
        trace_client=cast(GenerationTraceClient, langfuse_client),
        replacement_markdown=case.replacement_markdown,
    )
    provider = OpenAIChatGenerationProvider(
        api_key=api_key,
        base_url=openai_base_url,
        model=settings.generation_model,
        revision=settings.generation_model_revision,
        temperature=settings.generation_temperature,
        max_tokens=settings.generation_max_tokens,
        timeout_seconds=settings.generation_timeout_seconds,
        instrument_with_langfuse=False,
        client=controlled_client,
    )
    embedding_provider = SentenceTransformerEmbeddingProvider(
        model_id=settings.embedding_model_id,
        revision=settings.embedding_model_revision,
        device=settings.embedding_device,
        cache_dir=settings.embedding_cache_dir,
        local_files_only=settings.embedding_local_files_only,
    )
    retriever = FaissRetriever(
        corpus=load_corpus(),
        embedding_provider=embedding_provider,
        default_top_k=settings.retrieval_top_k,
        minimum_similarity=settings.retrieval_min_similarity,
    )
    system_prompt = load_system_prompt()
    retrieval_metadata = retriever.metadata
    recorder = LangfuseTraceRecorder(
        public_key=public_key,
        secret_key=secret_key,
        base_url=langfuse_base_url,
        environment=settings.langfuse_environment,
        release=__version__,
        flush_on_turn=settings.langfuse_flush_on_turn,
        client=langfuse_client,
        base_metadata={
            "application": "agentic-eval-toyapp",
            "app_version": __version__,
            "generation_model": provider.model_id,
            "generation_model_revision": provider.revision,
            "embedding_model": retrieval_metadata.embedding_model_id,
            "embedding_model_revision": retrieval_metadata.embedding_model_revision,
            "faiss_index": retrieval_metadata.index_type,
            "corpus_version": retrieval_metadata.corpus_version,
            "corpus_sha256": retrieval_metadata.corpus_sha256,
            "prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
        },
    )
    agent = RagAgent(
        provider=provider,
        retriever=AsyncRetriever(retriever, CapacityLimiter(1)),
        system_prompt=system_prompt,
        max_model_calls=settings.agent_max_model_calls,
        max_searches=settings.agent_max_searches,
        retrieval_top_k=settings.retrieval_top_k,
        trace_recorder=recorder,
    )
    # Controlled cases exercise the standalone RAG trace contract even when
    # normal browser and API traffic defaults to coordinator mode.
    rag_settings = settings.model_copy(update={"agent_mode": AgentMode.RAG})
    return create_app(rag_settings, agent=agent), controlled_client


def _label_for_case(
    labels: tuple[GroundTruthLabel, ...],
    case_id: str,
) -> GroundTruthLabel:
    """Return the unique local label for a selected case."""
    for label in labels:
        if label.case_id == case_id:
            return label
    raise TraceCaseExecutionError("the selected case has no ground-truth label")


def _controlled_result_was_applied(
    response: SessionMessageResponse,
    case: ControlledCase,
) -> bool:
    """Return whether the public result contains the exact controlled answer."""
    result = response.result
    if not result.model_actions or not isinstance(
        result.model_actions[-1], AnswerAction
    ):
        return False
    return result.message == case.replacement_markdown and not result.citations


def write_run_manifest(
    run: TraceCaseRun,
    run_directory: Path = DEFAULT_RUN_DIRECTORY,
) -> Path:
    """Write private state under its Langfuse-correlated session UUID."""
    run_directory.mkdir(parents=True, exist_ok=True)
    destination = run_directory / f"{run.session_id}.json"
    temporary = run_directory / f".{run.session_id}.tmp"
    if destination.exists():
        raise TraceCaseExecutionError(
            f"a run manifest already exists for session {run.session_id}"
        )
    temporary.write_text(
        run.model_dump_json(indent=2),
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def generate_case(
    *,
    metric: str,
    variant: CaseVariant,
    settings: AppSettings | None = None,
    run_directory: Path = DEFAULT_RUN_DIRECTORY,
) -> tuple[TraceCaseRun, Path]:
    """Generate one live controlled trace and persist its private mapping."""
    bundle = load_metric_bundle(metric)
    case = next(
        (candidate for candidate in bundle.cases.cases if candidate.variant is variant),
        None,
    )
    if case is None:
        raise TraceCaseExecutionError(
            f"metric {metric!r} has no {variant.value!r} case"
        )
    label = _label_for_case(bundle.ground_truth.labels, case.case_id)
    application, controlled_client = _build_live_application(
        settings or AppSettings(),
        case,
    )
    response = execute_api_turn(application, case)
    captures = controlled_client.captures
    response_applied = (
        _controlled_result_was_applied(response, case)
        and bool(captures)
        and captures[-1].altered
    )
    run = TraceCaseRun(
        generated_at=datetime.now(UTC),
        metric=bundle.cases.metric,
        case_id=case.case_id,
        variant=case.variant,
        expected_pass=label.expected_pass,
        expected_rationale=label.rationale,
        session_id=UUID(str(response.session_id)),
        user_message=case.user_message,
        controlled_response_applied=response_applied,
        result=response.result,
        model_completions=captures,
    )
    manifest_path = write_run_manifest(run, run_directory)
    if not response_applied:
        raise TraceCaseExecutionError(
            "the live model did not finish through the controlled answer boundary; "
            f"the unsuccessful trace mapping was saved at {manifest_path}"
        )
    return run, manifest_path


def public_run_summary(run: TraceCaseRun, manifest_path: Path) -> dict[str, object]:
    """Return a concise local CLI summary without exposing credentials."""
    return {
        "metric": run.metric,
        "variant": run.variant.value,
        "expected_pass": run.expected_pass,
        "controlled_response_applied": run.controlled_response_applied,
        "session_id": str(run.session_id),
        "status": run.result.status.value,
        "final_response": run.result.message,
        "manifest_path": str(manifest_path),
    }


def encode_public_summary(run: TraceCaseRun, manifest_path: Path) -> str:
    """Encode stable CLI output for humans and automation."""
    return json.dumps(
        public_run_summary(run, manifest_path),
        ensure_ascii=False,
        indent=2,
    )
