"""FastAPI application, dependency wiring, and versioned target endpoints."""

import hashlib
import logging
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from threading import Lock
from typing import Annotated
from uuid import UUID

from fastapi import FastAPI, File, Request, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from agentic_eval_toyapp import __version__
from agentic_eval_toyapp.agent import RagAgent, load_system_prompt
from agentic_eval_toyapp.config import (
    AppSettings,
    GenerationProviderKind,
    get_settings,
)
from agentic_eval_toyapp.embeddings import SentenceTransformerEmbeddingProvider
from agentic_eval_toyapp.models import (
    AgentActionKind,
    CapabilityResponse,
    CheckStatus,
    ErrorResponse,
    FeatureAvailability,
    HealthResponse,
    ReadinessCheck,
    ReadinessResponse,
    ReadinessStatus,
    ReportUploadResponse,
    SessionMessageRequest,
    SessionMessageResponse,
    SessionResponse,
)
from agentic_eval_toyapp.providers import (
    ExtractiveGenerationProvider,
    GenerationProvider,
    OpenAIChatGenerationProvider,
)
from agentic_eval_toyapp.reports import (
    InMemoryReportStore,
    ReportAttachmentLimitError,
    ReportCapacityError,
    ReportNotFoundError,
    ReportProcessingError,
    ReportProcessor,
    ReportTooLargeError,
    UnsupportedReportTypeError,
)
from agentic_eval_toyapp.retrieval import FaissRetriever, load_corpus
from agentic_eval_toyapp.sessions import (
    AgentRunner,
    InMemorySessionStore,
    SessionNotFoundError,
)
from agentic_eval_toyapp.tracing import (
    LangfuseTraceRecorder,
    NoOpTraceRecorder,
    TraceRecorder,
)

API_PREFIX = "/api/v1"
PACKAGE_DIRECTORY = Path(__file__).parent
STATIC_DIRECTORY = PACKAGE_DIRECTORY / "static"
TEMPLATE_DIRECTORY = PACKAGE_DIRECTORY / "templates"
TEMPLATES = Jinja2Templates(directory=str(TEMPLATE_DIRECTORY))
LOGGER = logging.getLogger("uvicorn.error")


class AgentInitialisationError(RuntimeError):
    """Raised when the configured RAG dependencies cannot be initialised."""


class _AgentResolver:
    """Initialise the default agent once without work at module import time."""

    def __init__(
        self,
        *,
        agent: AgentRunner | None,
        factory: Callable[[], AgentRunner] | None,
    ) -> None:
        if (agent is None) == (factory is None):
            raise ValueError("provide exactly one of agent or agent_factory")
        self._agent = agent
        self._factory = factory
        self._lock = Lock()

    def get(self) -> AgentRunner:
        """Return the injected agent or initialise the default exactly once."""
        if self._agent is not None:
            return self._agent
        with self._lock:
            if self._agent is not None:
                return self._agent
            if self._factory is None:  # Defensive; constructor rejects this state.
                raise AgentInitialisationError("agent factory is unavailable")
            try:
                candidate = self._factory()
            except Exception as error:
                raise AgentInitialisationError(
                    "agent dependencies could not be initialised"
                ) from error
            self._agent = candidate
            return candidate

    def shutdown(self) -> None:
        """Release resources owned by the default agent, if initialised."""
        if isinstance(self._agent, RagAgent):
            self._agent.shutdown()


def _build_generation_provider(settings: AppSettings) -> GenerationProvider:
    """Construct the explicitly selected process-level generation provider."""
    if settings.generation_provider is GenerationProviderKind.DETERMINISTIC:
        return ExtractiveGenerationProvider()
    api_key, base_url = settings.require_openai_configuration()
    return OpenAIChatGenerationProvider(
        api_key=api_key,
        base_url=base_url,
        model=settings.generation_model,
        revision=settings.generation_model_revision,
        temperature=settings.generation_temperature,
        max_tokens=settings.generation_max_tokens,
        timeout_seconds=settings.generation_timeout_seconds,
        instrument_with_langfuse=settings.langfuse_enabled,
    )


def _build_trace_recorder(
    settings: AppSettings,
    *,
    provider: GenerationProvider,
    retriever: FaissRetriever,
    system_prompt: str,
) -> TraceRecorder:
    """Build explicit opt-in tracing with immutable runtime provenance."""
    if not settings.langfuse_enabled:
        return NoOpTraceRecorder()
    public_key, secret_key, base_url = settings.require_langfuse_configuration()
    metadata = retriever.metadata
    return LangfuseTraceRecorder(
        public_key=public_key,
        secret_key=secret_key,
        base_url=base_url,
        environment=settings.langfuse_environment,
        release=__version__,
        flush_on_turn=settings.langfuse_flush_on_turn,
        base_metadata={
            "application": "agentic-eval-toyapp",
            "app_version": __version__,
            "generation_model": provider.model_id,
            "generation_model_revision": provider.revision,
            "embedding_model": metadata.embedding_model_id,
            "embedding_model_revision": metadata.embedding_model_revision,
            "faiss_index": metadata.index_type,
            "corpus_version": metadata.corpus_version,
            "corpus_sha256": metadata.corpus_sha256,
            "prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
        },
    )


def _build_default_agent(settings: AppSettings) -> RagAgent:
    """Construct the configured generation provider and semantic retriever."""
    generation_provider = _build_generation_provider(settings)
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
    trace_recorder = _build_trace_recorder(
        settings,
        provider=generation_provider,
        retriever=retriever,
        system_prompt=system_prompt,
    )
    return RagAgent(
        provider=generation_provider,
        retriever=retriever,
        system_prompt=system_prompt,
        max_model_calls=settings.agent_max_model_calls,
        max_searches=settings.agent_max_searches,
        retrieval_top_k=settings.retrieval_top_k,
        trace_recorder=trace_recorder,
    )


def _public_error(*, status_code: int, code: str, message: str) -> JSONResponse:
    error = ErrorResponse(code=code, message=message)
    return JSONResponse(
        status_code=status_code,
        content=error.model_dump(mode="json"),
    )


async def _handle_session_not_found(
    _request: Request, _error: Exception
) -> JSONResponse:
    return _public_error(
        status_code=404,
        code="session_not_found",
        message="The requested session was not found.",
    )


async def _handle_agent_initialisation_error(
    _request: Request, _error: Exception
) -> JSONResponse:
    return _public_error(
        status_code=503,
        code="agent_unavailable",
        message="The RAG agent is not ready.",
    )


async def _handle_request_validation_error(
    _request: Request, _error: Exception
) -> JSONResponse:
    return _public_error(
        status_code=422,
        code="invalid_request",
        message="The request did not match the API schema.",
    )


async def _handle_report_not_found(
    _request: Request, _error: Exception
) -> JSONResponse:
    return _public_error(
        status_code=404,
        code="report_not_found",
        message="The requested report was not found for this session.",
    )


async def _handle_unsupported_report_type(
    _request: Request, _error: Exception
) -> JSONResponse:
    return _public_error(
        status_code=415,
        code="unsupported_file_type",
        message="The uploaded file type is not supported.",
    )


async def _handle_report_too_large(
    _request: Request, _error: Exception
) -> JSONResponse:
    return _public_error(
        status_code=413,
        code="report_too_large",
        message="The uploaded report exceeds an accepted processing limit.",
    )


async def _handle_report_processing_error(
    _request: Request, _error: Exception
) -> JSONResponse:
    return _public_error(
        status_code=422,
        code="report_processing_failed",
        message="The uploaded report could not be processed.",
    )


async def _handle_report_capacity_error(
    _request: Request, _error: Exception
) -> JSONResponse:
    return _public_error(
        status_code=409,
        code="report_capacity_reached",
        message="This session cannot retain additional reports.",
    )


async def _handle_report_attachment_limit(
    _request: Request, _error: Exception
) -> JSONResponse:
    return _public_error(
        status_code=422,
        code="too_many_report_files",
        message="The message references too many reports.",
    )


def create_app(
    settings: AppSettings | None = None,
    *,
    session_store: InMemorySessionStore | None = None,
    agent: AgentRunner | None = None,
    agent_factory: Callable[[], AgentRunner] | None = None,
    report_processor: ReportProcessor | None = None,
    report_store: InMemoryReportStore | None = None,
) -> FastAPI:
    """Construct an application with explicit, injectable dependencies."""
    resolved_settings = settings or get_settings()
    resolved_store = session_store or InMemorySessionStore(
        max_turns=resolved_settings.session_max_turns
    )
    resolved_report_processor = report_processor or ReportProcessor(
        max_bytes=resolved_settings.upload_max_bytes,
        max_pdf_pages=resolved_settings.upload_max_pdf_pages,
        max_extracted_characters=(resolved_settings.upload_max_extracted_characters),
    )
    resolved_report_store = report_store or InMemoryReportStore(
        max_reports_per_session=resolved_settings.upload_max_reports_per_session
    )
    if agent is None and agent_factory is None:
        agent_factory = partial(_build_default_agent, resolved_settings)
    resolver = _AgentResolver(agent=agent, factory=agent_factory)

    @asynccontextmanager
    async def lifespan(_application: FastAPI) -> AsyncGenerator[None, None]:
        """Finish RAG initialisation before the server accepts requests."""
        resolver.get()
        LOGGER.info(
            "application_started generation_provider=%s langfuse_enabled=%s",
            resolved_settings.generation_provider.value,
            str(resolved_settings.langfuse_enabled).lower(),
        )
        try:
            yield
        finally:
            resolver.shutdown()

    application = FastAPI(
        title=resolved_settings.app_name,
        version=__version__,
        description="Controlled RAG target application for METIS evaluations.",
        lifespan=lifespan,
    )
    application.mount(
        "/static",
        StaticFiles(directory=STATIC_DIRECTORY),
        name="static",
    )
    application.add_exception_handler(
        SessionNotFoundError,
        _handle_session_not_found,
    )
    application.add_exception_handler(
        AgentInitialisationError,
        _handle_agent_initialisation_error,
    )
    application.add_exception_handler(
        RequestValidationError,
        _handle_request_validation_error,
    )
    application.add_exception_handler(ReportNotFoundError, _handle_report_not_found)
    application.add_exception_handler(
        UnsupportedReportTypeError,
        _handle_unsupported_report_type,
    )
    application.add_exception_handler(ReportTooLargeError, _handle_report_too_large)
    application.add_exception_handler(
        ReportProcessingError,
        _handle_report_processing_error,
    )
    application.add_exception_handler(
        ReportCapacityError,
        _handle_report_capacity_error,
    )
    application.add_exception_handler(
        ReportAttachmentLimitError,
        _handle_report_attachment_limit,
    )

    @application.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index(request: Request) -> Response:
        """Render the browser client after startup dependencies are ready."""
        return TEMPLATES.TemplateResponse(
            request=request,
            name="index.html",
            context={"app_name": resolved_settings.app_name},
            headers={
                "Content-Security-Policy": (
                    "default-src 'self'; script-src 'self'; style-src 'self'; "
                    "connect-src 'self'; img-src 'self'; base-uri 'none'; "
                    "frame-ancestors 'none'; form-action 'self'"
                ),
                "X-Content-Type-Options": "nosniff",
            },
        )

    @application.get("/health", response_model=HealthResponse, tags=["service"])
    def health() -> HealthResponse:
        """Report process liveness without checking external dependencies."""
        return HealthResponse(
            service=resolved_settings.app_name,
            version=__version__,
        )

    @application.get("/ready", response_model=ReadinessResponse, tags=["service"])
    def ready() -> ReadinessResponse:
        """Report the dependencies initialised during application startup."""
        checks = [
            ReadinessCheck(
                name="configuration",
                status=CheckStatus.PASS,
                detail="service settings validated",
            ),
            ReadinessCheck(
                name="session_store",
                status=CheckStatus.PASS,
                detail=f"retains up to {resolved_store.max_turns} complete turns",
            ),
            ReadinessCheck(
                name="report_store",
                status=CheckStatus.PASS,
                detail=(
                    "retains up to "
                    f"{resolved_report_store.max_reports_per_session} "
                    "processed reports per session"
                ),
            ),
            ReadinessCheck(
                name="langfuse_tracing",
                status=CheckStatus.PASS,
                detail=(
                    "enabled and configured"
                    if resolved_settings.langfuse_enabled
                    else "disabled"
                ),
            ),
        ]
        try:
            resolver.get()
        except AgentInitialisationError:
            checks.append(
                ReadinessCheck(
                    name="rag_agent",
                    status=CheckStatus.FAIL,
                    detail="agent dependencies could not be initialised",
                )
            )
            return ReadinessResponse(
                status=ReadinessStatus.NOT_READY,
                checks=tuple(checks),
            )
        checks.append(
            ReadinessCheck(
                name="rag_agent",
                status=CheckStatus.PASS,
                detail="agent dependencies initialised",
            )
        )
        return ReadinessResponse(
            status=ReadinessStatus.READY,
            checks=tuple(checks),
        )

    @application.get(
        f"{API_PREFIX}/capabilities",
        response_model=CapabilityResponse,
        tags=["target"],
    )
    def capabilities() -> CapabilityResponse:
        """Advertise the synchronous session and semantic-retrieval path."""
        return CapabilityResponse(
            service_version=__version__,
            features=FeatureAvailability(
                session_chat=True,
                semantic_retrieval=True,
                langfuse_tracing=resolved_settings.langfuse_enabled,
                file_uploads=True,
            ),
            actions=tuple(action.value for action in AgentActionKind),
        )

    @application.post(
        f"{API_PREFIX}/sessions",
        response_model=SessionResponse,
        status_code=201,
        responses={422: {"model": ErrorResponse}},
        tags=["chat"],
    )
    def create_session() -> SessionResponse:
        """Create one empty process-local conversation session."""
        return resolved_store.create()

    @application.get(
        f"{API_PREFIX}/sessions/{{session_id}}",
        response_model=SessionResponse,
        responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
        tags=["chat"],
    )
    def get_session(session_id: UUID) -> SessionResponse:
        """Return safe public messages for one existing session."""
        return resolved_store.get(session_id)

    @application.post(
        f"{API_PREFIX}/sessions/{{session_id}}/files",
        response_model=ReportUploadResponse,
        status_code=201,
        responses={
            404: {"model": ErrorResponse},
            409: {"model": ErrorResponse},
            413: {"model": ErrorResponse},
            415: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
        },
        tags=["files"],
    )
    async def upload_report(
        session_id: UUID,
        file: Annotated[UploadFile, File(description="One bounded report file")],
    ) -> ReportUploadResponse:
        """Validate and retain one report without invoking an agent turn."""
        resolved_store.get(session_id)
        try:
            content = await file.read(resolved_report_processor.max_bytes + 1)
            processed = await run_in_threadpool(
                resolved_report_processor.process,
                filename=file.filename or "",
                media_type=file.content_type or "",
                content=content,
            )
        finally:
            await file.close()
        return resolved_report_store.add(session_id, processed)

    @application.post(
        f"{API_PREFIX}/sessions/{{session_id}}/messages",
        response_model=SessionMessageResponse,
        responses={
            404: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
        tags=["chat"],
    )
    def submit_message(
        session_id: UUID,
        request: SessionMessageRequest,
    ) -> SessionMessageResponse:
        """Execute and retain one synchronous bounded agent turn."""
        resolved_store.get(session_id)
        if len(request.file_ids) > resolved_settings.upload_max_files_per_message:
            raise ReportAttachmentLimitError
        resolved_report_store.resolve(session_id, request.file_ids)
        resolved_agent = resolver.get()
        result = resolved_store.run_turn(
            session_id,
            request.message,
            agent=resolved_agent,
        )
        return SessionMessageResponse(session_id=session_id, result=result)

    return application


app = create_app()
