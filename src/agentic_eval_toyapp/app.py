"""FastAPI application, dependency wiring, and versioned target endpoints."""

import hashlib
import logging
from collections.abc import AsyncGenerator, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Annotated
from uuid import UUID

from anyio import CapacityLimiter
from fastapi import FastAPI, File, Request, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from agentic_eval_toyapp import __version__
from agentic_eval_toyapp.agent import RagAgent, RagSpecialistAdapter, load_system_prompt
from agentic_eval_toyapp.analysis import (
    DeterministicAnalysisGenerationProvider,
    OpenAIAnalysisGenerationProvider,
    ReportAnalysisAgent,
    load_analysis_prompt,
)
from agentic_eval_toyapp.backpressure import (
    RETRY_AFTER_SECONDS,
    RateLimitedCompletionClient,
    ServerBusyError,
    TurnAdmission,
)
from agentic_eval_toyapp.config import (
    AgentMode,
    AppSettings,
    GenerationProviderKind,
    SessionStoreBackend,
    get_settings,
)
from agentic_eval_toyapp.coordinator import (
    CoordinatorAgent,
    DeterministicCoordinatorGenerationProvider,
    build_openai_coordinator_provider,
    load_coordinator_prompt,
)
from agentic_eval_toyapp.demo import load_demo, seed_demo
from agentic_eval_toyapp.diagnostics import (
    classify_external_error,
    diagnostic_context,
    log_failure,
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
    ReportListResponse,
    ReportUploadResponse,
    SessionMessageRequest,
    SessionMessageResponse,
    SessionResponse,
)
from agentic_eval_toyapp.multi_agent_tracing import (
    CoordinatorTraceRecorder,
    LangfuseCoordinatorTraceRecorder,
    NoOpCoordinatorTraceRecorder,
)
from agentic_eval_toyapp.orchestration_models import CoordinatorActionKind
from agentic_eval_toyapp.providers import (
    ExtractiveGenerationProvider,
    GenerationProvider,
    OpenAIChatGenerationProvider,
    OpenAICompatibleChatClient,
    StructuredCompletionClient,
)
from agentic_eval_toyapp.reports import (
    InMemoryReportStore,
    RedisReportStore,
    ReportAttachmentLimitError,
    ReportCapacityError,
    ReportNotFoundError,
    ReportProcessingError,
    ReportProcessor,
    ReportStore,
    ReportStoreBusyError,
    ReportStoreUnavailableError,
    ReportTooLargeError,
    UnsupportedReportTypeError,
)
from agentic_eval_toyapp.retrieval import (
    AsyncRetriever,
    FaissRetriever,
    corpus_sha256,
    load_corpus,
)
from agentic_eval_toyapp.session_export import export_session
from agentic_eval_toyapp.sessions import (
    AgentRunner,
    InMemorySessionStore,
    RedisSessionStore,
    SessionBusyError,
    SessionNotFoundError,
    SessionStore,
    SessionStoreUnavailableError,
)
from agentic_eval_toyapp.tracing import (
    LangfuseTraceRecorder,
    NoOpTraceRecorder,
    TraceExportError,
    TraceRecorder,
)

API_PREFIX = "/api/v1"
PACKAGE_DIRECTORY = Path(__file__).parent
STATIC_DIRECTORY = PACKAGE_DIRECTORY / "static"
TEMPLATE_DIRECTORY = PACKAGE_DIRECTORY / "templates"
TEMPLATES = Jinja2Templates(directory=str(TEMPLATE_DIRECTORY))
BROWSER_ASSET_VERSION = hashlib.sha256(
    (STATIC_DIRECTORY / "app.js").read_bytes()
    + (STATIC_DIRECTORY / "style.css").read_bytes()
).hexdigest()[:12]
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
        self._retrieval_limiter: CapacityLimiter | None = None

    def bind_retrieval_limiter(self, limiter: CapacityLimiter) -> None:
        """Record the lifespan-owned limiter used by the default agent."""
        self._retrieval_limiter = limiter

    def require_retrieval_limiter(self) -> CapacityLimiter:
        """Return the bound limiter or fail rather than create a second slot."""
        if self._retrieval_limiter is None:
            raise RuntimeError("retrieval limiter is not bound; run the lifespan")
        return self._retrieval_limiter

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
                category, status_code = classify_external_error(error)
                log_failure(
                    event="startup",
                    agent="runtime",
                    operation="initialise_agent",
                    code=(
                        "configuration_error"
                        if isinstance(error, ValueError)
                        else category
                        if category != "unexpected_error"
                        else "initialisation_error"
                    ),
                    status_code=status_code,
                    error=error,
                )
                safe_cause: Exception = (
                    ValueError("invalid agent configuration")
                    if isinstance(error, ValueError)
                    else RuntimeError("agent initialisation failed")
                )
                raise AgentInitialisationError(
                    "agent dependencies could not be initialised"
                ) from safe_cause
            self._agent = candidate
            return candidate

    def shutdown(self) -> None:
        """Release the agent's tracing resources, if it has been initialised.

        This is the one place the final Langfuse flush happens when
        `TOYAPP_LANGFUSE_FLUSH_ON_TURN=false`: the recorder's `shutdown()` drains
        the SDK's background queue.
        """
        # Intentional: `AgentRunner` has no `shutdown()`, so only the two agent
        # types known to own a trace recorder are closed. An injected custom
        # runner owns its own resources and its caller must close them.
        if isinstance(self._agent, (RagAgent, CoordinatorAgent)):
            self._agent.shutdown()


def _build_completion_client(
    settings: AppSettings,
    *,
    instrument_with_langfuse: bool,
) -> StructuredCompletionClient:
    """Build the one rate-limited client that every live agent call shares."""
    api_key, base_url = settings.require_openai_configuration()
    return RateLimitedCompletionClient(
        OpenAICompatibleChatClient(
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=settings.generation_timeout_seconds,
            instrument_with_langfuse=instrument_with_langfuse,
        ),
        max_requests_per_minute=settings.llm_max_requests_per_minute,
        max_concurrent_calls=settings.llm_max_concurrent_calls,
    )


def _build_generation_provider(
    settings: AppSettings,
    *,
    instrument_with_langfuse: bool | None = None,
    completion_client: StructuredCompletionClient | None = None,
) -> GenerationProvider:
    """Construct the explicitly selected process-level generation provider."""
    if settings.generation_provider is GenerationProviderKind.DETERMINISTIC:
        return ExtractiveGenerationProvider()
    api_key, base_url = settings.require_openai_configuration()
    tracing_enabled = (
        settings.langfuse_enabled
        if instrument_with_langfuse is None
        else instrument_with_langfuse
    )
    completion_client = completion_client or _build_completion_client(
        settings, instrument_with_langfuse=tracing_enabled
    )
    return OpenAIChatGenerationProvider(
        api_key=api_key,
        base_url=base_url,
        model=settings.generation_model,
        revision=settings.generation_model_revision,
        temperature=settings.generation_temperature,
        max_tokens=settings.generation_max_tokens,
        timeout_seconds=settings.generation_timeout_seconds,
        instrument_with_langfuse=tracing_enabled,
        client=completion_client,
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


def _build_rag_agent(
    settings: AppSettings,
    retrieval_limiter: CapacityLimiter,
    *,
    trace_export_enabled: bool | None = None,
    completion_client: StructuredCompletionClient | None = None,
) -> RagAgent:
    """Construct the configured RAG provider, retriever, and tracing boundary."""
    export_traces = (
        settings.langfuse_enabled
        if trace_export_enabled is None
        else trace_export_enabled
    )
    generation_provider = _build_generation_provider(
        settings,
        instrument_with_langfuse=export_traces,
        completion_client=completion_client,
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
    trace_recorder = (
        _build_trace_recorder(
            settings,
            provider=generation_provider,
            retriever=retriever,
            system_prompt=system_prompt,
        )
        if export_traces
        else NoOpTraceRecorder()
    )
    return RagAgent(
        provider=generation_provider,
        retriever=AsyncRetriever(retriever, retrieval_limiter),
        system_prompt=system_prompt,
        max_model_calls=settings.agent_max_model_calls,
        max_searches=settings.agent_max_searches,
        retrieval_top_k=settings.retrieval_top_k,
        trace_recorder=trace_recorder,
    )


def _build_coordinator_agent(
    settings: AppSettings, retrieval_limiter: CapacityLimiter
) -> CoordinatorAgent:
    """Construct the coordinator with RAG and report-analysis specialists."""
    completion_client: StructuredCompletionClient | None = None
    if settings.generation_provider is GenerationProviderKind.DETERMINISTIC:
        provider = DeterministicCoordinatorGenerationProvider()
    else:
        api_key, base_url = settings.require_openai_configuration()
        # One client, so one pacing clock and one semaphore, serves the
        # coordinator, the RAG specialist and the analysis specialist.
        completion_client = _build_completion_client(
            settings, instrument_with_langfuse=False
        )
        provider = build_openai_coordinator_provider(
            api_key=api_key,
            base_url=base_url,
            model=settings.generation_model,
            revision=settings.generation_model_revision,
            temperature=settings.generation_temperature,
            max_tokens=settings.generation_max_tokens,
            timeout_seconds=settings.generation_timeout_seconds,
            client=completion_client,
        )
    rag_agent = _build_rag_agent(
        settings,
        retrieval_limiter,
        trace_export_enabled=False,
        completion_client=completion_client,
    )
    rag_specialist = RagSpecialistAdapter(rag_agent)
    if settings.generation_provider is GenerationProviderKind.DETERMINISTIC:
        analysis_provider = DeterministicAnalysisGenerationProvider()
    else:
        analysis_provider = OpenAIAnalysisGenerationProvider(
            model=settings.generation_model,
            revision=settings.generation_model_revision,
            temperature=settings.generation_temperature,
            max_tokens=settings.analysis_max_output_tokens,
            client=completion_client
            or _build_completion_client(settings, instrument_with_langfuse=False),
        )
    analysis_specialist = ReportAnalysisAgent(
        provider=analysis_provider,
        system_prompt=load_analysis_prompt(),
    )
    trace_recorder: CoordinatorTraceRecorder = NoOpCoordinatorTraceRecorder()
    if settings.langfuse_enabled:
        public_key, secret_key, langfuse_url = settings.require_langfuse_configuration()
        corpus = load_corpus()
        trace_recorder = LangfuseCoordinatorTraceRecorder(
            public_key=public_key,
            secret_key=secret_key,
            base_url=langfuse_url,
            environment=settings.langfuse_environment,
            release=__version__,
            rag_model_id=rag_agent.model_id,
            include_report_text=settings.langfuse_include_report_text,
            flush_on_turn=settings.langfuse_flush_on_turn,
            base_metadata={
                "application": "agentic-eval-toyapp",
                "app_version": __version__,
                "generation_model": provider.model_id,
                "generation_model_revision": provider.revision,
                "embedding_model": settings.embedding_model_id,
                "embedding_model_revision": settings.embedding_model_revision,
                "corpus_version": corpus.version,
                "corpus_sha256": corpus_sha256(corpus),
                "coordinator_prompt_sha256": hashlib.sha256(
                    load_coordinator_prompt().encode()
                ).hexdigest(),
                "analysis_prompt_sha256": hashlib.sha256(
                    load_analysis_prompt().encode()
                ).hexdigest(),
            },
        )
    return CoordinatorAgent(
        provider=provider,
        system_prompt=load_coordinator_prompt(),
        max_model_calls=settings.coordinator_max_model_calls,
        rag_specialist=rag_specialist,
        analysis_specialist=analysis_specialist,
        trace_recorder=trace_recorder,
    )


def _build_default_agent(
    settings: AppSettings, retrieval_limiter: CapacityLimiter
) -> AgentRunner:
    """Construct the explicitly selected top-level agent runtime."""
    if settings.agent_mode is AgentMode.COORDINATOR:
        return _build_coordinator_agent(settings, retrieval_limiter)
    return _build_rag_agent(settings, retrieval_limiter)


def _build_session_store(settings: AppSettings) -> SessionStore:
    """Construct the explicitly selected conversation-session backend."""
    if settings.session_store_backend is SessionStoreBackend.MEMORY:
        return InMemorySessionStore(max_turns=settings.session_max_turns)
    return RedisSessionStore.from_url(
        settings.require_redis_configuration(),
        max_turns=settings.session_max_turns,
        key_prefix=settings.redis_key_prefix,
        socket_timeout_seconds=settings.redis_socket_timeout_seconds,
        lock_timeout_seconds=settings.redis_lock_timeout_seconds,
        lock_blocking_timeout_seconds=(settings.redis_lock_blocking_timeout_seconds),
    )


def _build_report_store(settings: AppSettings) -> ReportStore:
    """Construct the report backend selected with conversation persistence."""
    if settings.session_store_backend is SessionStoreBackend.MEMORY:
        return InMemoryReportStore(
            max_reports_per_session=settings.upload_max_reports_per_session
        )
    return RedisReportStore.from_url(
        settings.require_redis_configuration(),
        max_reports_per_session=settings.upload_max_reports_per_session,
        key_prefix=settings.redis_key_prefix,
        socket_timeout_seconds=settings.redis_socket_timeout_seconds,
        lock_timeout_seconds=settings.redis_lock_timeout_seconds,
        lock_blocking_timeout_seconds=(settings.redis_lock_blocking_timeout_seconds),
    )


def _public_error(
    *,
    status_code: int,
    code: str,
    message: str,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    error = ErrorResponse(code=code, message=message)
    return JSONResponse(
        status_code=status_code,
        content=error.model_dump(mode="json"),
        headers=headers,
    )


async def _handle_session_not_found(
    _request: Request, _error: Exception
) -> JSONResponse:
    return _public_error(
        status_code=404,
        code="session_not_found",
        message="The requested session was not found.",
    )


async def _handle_session_store_unavailable(
    _request: Request, _error: Exception
) -> JSONResponse:
    return _public_error(
        status_code=503,
        code="session_store_unavailable",
        message="The conversation session store is temporarily unavailable.",
    )


async def _handle_session_busy(_request: Request, _error: Exception) -> JSONResponse:
    return _public_error(
        status_code=409,
        code="session_busy",
        message="Another request is currently updating this session.",
    )


async def _handle_report_store_unavailable(
    _request: Request, _error: Exception
) -> JSONResponse:
    return _public_error(
        status_code=503,
        code="report_store_unavailable",
        message="The report store is temporarily unavailable.",
    )


async def _handle_report_store_busy(
    _request: Request, _error: Exception
) -> JSONResponse:
    return _public_error(
        status_code=409,
        code="report_store_busy",
        message="Another request is currently updating this session's reports.",
    )


async def _handle_agent_initialisation_error(
    _request: Request, _error: Exception
) -> JSONResponse:
    return _public_error(
        status_code=503,
        code="agent_unavailable",
        message="The configured agent runtime is not ready.",
    )


async def _handle_server_busy(_request: Request, _error: Exception) -> JSONResponse:
    return _public_error(
        status_code=503,
        code="server_busy",
        message="The service is at capacity. Retry the request shortly.",
        headers={"Retry-After": str(RETRY_AFTER_SECONDS)},
    )


async def _handle_trace_export_error(
    _request: Request, _error: Exception
) -> JSONResponse:
    return _public_error(
        status_code=503,
        code="trace_export_failed",
        message=(
            "The turn ran but its trace could not be exported to Langfuse. "
            "Treat the trace as incomplete."
        ),
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
    session_store: SessionStore | None = None,
    agent: AgentRunner | None = None,
    agent_factory: Callable[[], AgentRunner] | None = None,
    report_processor: ReportProcessor | None = None,
    report_store: ReportStore | None = None,
) -> FastAPI:
    """Construct an application with explicit, injectable dependencies."""
    resolved_settings = settings or get_settings()
    resolved_store = session_store or _build_session_store(resolved_settings)
    resolved_report_processor = report_processor or ReportProcessor(
        max_bytes=resolved_settings.upload_max_bytes,
        max_pdf_pages=resolved_settings.upload_max_pdf_pages,
        max_extracted_characters=(resolved_settings.upload_max_extracted_characters),
    )
    resolved_report_store = report_store or _build_report_store(resolved_settings)
    demo = load_demo()
    admission = TurnAdmission(resolved_settings.max_pending_turns)
    resolver: _AgentResolver

    def default_agent_factory() -> AgentRunner:
        return _build_default_agent(
            resolved_settings, resolver.require_retrieval_limiter()
        )

    if agent is None and agent_factory is None:
        agent_factory = default_agent_factory
    resolver = _AgentResolver(agent=agent, factory=agent_factory)

    @asynccontextmanager
    async def lifespan(_application: FastAPI) -> AsyncGenerator[None, None]:
        """Finish RAG initialisation before the server accepts requests."""
        async with AsyncExitStack() as stack:
            # Redis connection pools are bound to this loop, so the stores build
            # their clients here and close them last, after the agent's flush.
            await resolved_store.open()
            stack.push_async_callback(resolved_store.aclose)
            await resolved_report_store.open()
            stack.push_async_callback(resolved_report_store.aclose)
            # Seed the bundled demonstration session before the agent
            # initialises so it is available from the first request.
            demo_seeded = await seed_demo(demo, resolved_store, resolved_report_store)
            if demo_seeded:
                LOGGER.info("demo_session_seeded session_id=%s", demo.session_id)
            # One slot for the whole process, created on the running loop and
            # shared by every agent built from the default factory.
            resolver.bind_retrieval_limiter(CapacityLimiter(1))
            resolver.get()
            LOGGER.info(
                "application_started agent_mode=%s generation_provider=%s "
                "langfuse_enabled=%s",
                resolved_settings.agent_mode.value,
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
        SessionStoreUnavailableError,
        _handle_session_store_unavailable,
    )
    application.add_exception_handler(SessionBusyError, _handle_session_busy)
    application.add_exception_handler(
        AgentInitialisationError,
        _handle_agent_initialisation_error,
    )
    application.add_exception_handler(ServerBusyError, _handle_server_busy)
    application.add_exception_handler(TraceExportError, _handle_trace_export_error)
    application.add_exception_handler(
        RequestValidationError,
        _handle_request_validation_error,
    )
    application.add_exception_handler(ReportNotFoundError, _handle_report_not_found)
    application.add_exception_handler(
        ReportStoreUnavailableError,
        _handle_report_store_unavailable,
    )
    application.add_exception_handler(ReportStoreBusyError, _handle_report_store_busy)
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

    def render_browser_client(request: Request) -> Response:
        """Render the shared browser shell with its response security headers."""
        return TEMPLATES.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "app_name": resolved_settings.app_name,
                "browser_asset_version": BROWSER_ASSET_VERSION,
                "demo_session_id": str(demo.session_id),
            },
            headers={
                "Cache-Control": "no-store",
                "Content-Security-Policy": (
                    "default-src 'self'; script-src 'self'; style-src 'self'; "
                    "connect-src 'self'; img-src 'self'; base-uri 'none'; "
                    "frame-ancestors 'none'; form-action 'self'"
                ),
                "X-Content-Type-Options": "nosniff",
            },
        )

    @application.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index(request: Request) -> Response:
        """Render the browser client after startup dependencies are ready."""
        return render_browser_client(request)

    @application.get(
        "/session/{session_id:uuid}",
        response_class=HTMLResponse,
        include_in_schema=False,
    )
    def session_index(request: Request, session_id: UUID) -> Response:
        """Render the browser client for a UUID selected by the URL path."""
        del session_id
        return render_browser_client(request)

    @application.get("/health", response_model=HealthResponse, tags=["service"])
    def health() -> HealthResponse:
        """Report process liveness without checking external dependencies."""
        return HealthResponse(
            service=resolved_settings.app_name,
            version=__version__,
        )

    @application.get("/ready", response_model=ReadinessResponse, tags=["service"])
    async def ready() -> ReadinessResponse:
        """Report the dependencies initialised during application startup."""
        session_store_ready = True
        try:
            await resolved_store.check()
            session_store_check = ReadinessCheck(
                name="session_store",
                status=CheckStatus.PASS,
                detail=resolved_store.readiness_detail,
            )
        except SessionStoreUnavailableError:
            session_store_ready = False
            session_store_check = ReadinessCheck(
                name="session_store",
                status=CheckStatus.FAIL,
                detail="configured session store is unavailable",
            )
        report_store_ready = True
        try:
            await resolved_report_store.check()
            report_store_check = ReadinessCheck(
                name="report_store",
                status=CheckStatus.PASS,
                detail=resolved_report_store.readiness_detail,
            )
        except ReportStoreUnavailableError:
            report_store_ready = False
            report_store_check = ReadinessCheck(
                name="report_store",
                status=CheckStatus.FAIL,
                detail="configured report store is unavailable",
            )
        checks = [
            ReadinessCheck(
                name="configuration",
                status=CheckStatus.PASS,
                detail="service settings validated",
            ),
            session_store_check,
            report_store_check,
            ReadinessCheck(
                name="langfuse_tracing",
                status=CheckStatus.PASS,
                detail=(
                    "disabled"
                    if not resolved_settings.langfuse_enabled
                    else "enabled and configured"
                ),
            ),
        ]
        agent_check_name = f"{resolved_settings.agent_mode.value}_agent"
        try:
            resolver.get()
        except AgentInitialisationError:
            checks.append(
                ReadinessCheck(
                    name=agent_check_name,
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
                name=agent_check_name,
                status=CheckStatus.PASS,
                detail=(
                    f"{resolved_settings.agent_mode.value} agent dependencies "
                    "initialised"
                ),
            )
        )
        return ReadinessResponse(
            status=(
                ReadinessStatus.READY
                if session_store_ready and report_store_ready
                else ReadinessStatus.NOT_READY
            ),
            checks=tuple(checks),
        )

    @application.get(
        f"{API_PREFIX}/capabilities",
        response_model=CapabilityResponse,
        tags=["target"],
    )
    def capabilities() -> CapabilityResponse:
        """Advertise the request/response session and semantic-retrieval path."""
        tracing_operational = resolved_settings.langfuse_enabled
        advertised_actions = (
            tuple(action.value for action in CoordinatorActionKind)
            if resolved_settings.agent_mode is AgentMode.COORDINATOR
            else tuple(action.value for action in AgentActionKind)
        )
        return CapabilityResponse(
            service_version=__version__,
            features=FeatureAvailability(
                session_chat=True,
                semantic_retrieval=True,
                langfuse_tracing=tracing_operational,
                file_uploads=True,
            ),
            actions=advertised_actions,
        )

    @application.post(
        f"{API_PREFIX}/sessions",
        response_model=SessionResponse,
        status_code=201,
        responses={
            422: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
        tags=["chat"],
    )
    async def create_session() -> SessionResponse:
        """Create one empty conversation session in the configured store."""
        return await resolved_store.create()

    @application.get(
        f"{API_PREFIX}/sessions/{{session_id}}",
        response_model=SessionResponse,
        responses={
            404: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
        tags=["chat"],
    )
    async def get_session(session_id: UUID) -> SessionResponse:
        """Return safe public messages for one existing session."""
        return await resolved_store.get(session_id)

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
            503: {"model": ErrorResponse},
        },
        tags=["files"],
    )
    async def upload_report(
        session_id: UUID,
        file: Annotated[UploadFile, File(description="One bounded report file")],
    ) -> ReportUploadResponse:
        """Validate and retain one report without invoking an agent turn."""
        with diagnostic_context(session_id):
            await resolved_store.get(session_id)
            try:
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
                return await resolved_report_store.add(session_id, processed)
            except (
                UnsupportedReportTypeError,
                ReportTooLargeError,
                ReportProcessingError,
                ReportCapacityError,
            ) as error:
                if isinstance(error, UnsupportedReportTypeError):
                    code = "unsupported_file_type"
                elif isinstance(error, ReportTooLargeError):
                    code = "report_too_large"
                elif isinstance(error, ReportCapacityError):
                    code = "report_capacity_reached"
                else:
                    code = "report_processing_failed"
                log_failure(
                    event="upload",
                    agent="api",
                    operation="process_report",
                    code=code,
                    level=logging.WARNING,
                    error=error,
                )
                raise
            except (ReportStoreUnavailableError, ReportStoreBusyError):
                raise
            except Exception as error:
                log_failure(
                    event="upload",
                    agent="api",
                    operation="process_report",
                    code="unexpected_upload_error",
                    error=error,
                )
                raise RuntimeError("report upload could not be processed") from None

    @application.get(
        f"{API_PREFIX}/sessions/{{session_id}}/files",
        response_model=ReportListResponse,
        responses={
            404: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
        tags=["files"],
    )
    async def list_reports(session_id: UUID) -> ReportListResponse:
        """List safe metadata for reports retained by this session."""
        await resolved_store.get(session_id)
        return ReportListResponse(
            files=await resolved_report_store.list_for_session(session_id)
        )

    @application.delete(
        f"{API_PREFIX}/sessions/{{session_id}}/files/{{file_id}}",
        status_code=204,
        responses={
            404: {"model": ErrorResponse},
            409: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
        tags=["files"],
    )
    async def delete_report(session_id: UUID, file_id: UUID) -> Response:
        """Remove one owned report without changing past conversation turns."""
        await resolved_store.get(session_id)
        await resolved_report_store.delete(session_id, file_id)
        return Response(status_code=204)

    @application.post(
        f"{API_PREFIX}/sessions/{{session_id}}/messages",
        response_model=SessionMessageResponse,
        responses={
            404: {"model": ErrorResponse},
            409: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
        tags=["chat"],
    )
    async def submit_message(
        session_id: UUID,
        request: SessionMessageRequest,
    ) -> SessionMessageResponse:
        """Execute and retain one bounded agent turn."""
        with diagnostic_context(session_id, turn=True):
            await resolved_store.get(session_id)
            if len(request.file_ids) > resolved_settings.upload_max_files_per_message:
                log_failure(
                    event="message",
                    agent="api",
                    operation="resolve_attachments",
                    code="too_many_report_files",
                    level=logging.WARNING,
                )
                raise ReportAttachmentLimitError
            try:
                reports = await resolved_report_store.resolve(
                    session_id, request.file_ids
                )
            except ReportNotFoundError as error:
                log_failure(
                    event="message",
                    agent="api",
                    operation="resolve_attachments",
                    code="report_not_found",
                    level=logging.WARNING,
                    error=error,
                )
                raise
            resolved_agent = resolver.get()
            # Refuse overload before any agent code, trace or stored turn exists.
            if not admission.try_acquire():
                log_failure(
                    event="message",
                    agent="api",
                    operation="admit_turn",
                    code="server_busy",
                    level=logging.WARNING,
                    status_code=503,
                )
                raise ServerBusyError
            try:
                result = await resolved_store.run_turn(
                    session_id,
                    request.message,
                    agent=resolved_agent,
                    reports=reports,
                )
            finally:
                admission.release()
            return SessionMessageResponse(session_id=session_id, result=result)

    @application.get(
        f"{API_PREFIX}/sessions/{{session_id}}/export",
        responses={
            404: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
        tags=["chat"],
    )
    async def export_session_endpoint(session_id: UUID) -> Response:
        """Export one session as a portable JSON or ZIP file."""
        content, content_type, filename = await export_session(
            session_id, resolved_store, resolved_report_store
        )
        return Response(
            content=content,
            media_type=content_type,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "X-Content-Type-Options": "nosniff",
            },
        )

    return application


app = create_app()
