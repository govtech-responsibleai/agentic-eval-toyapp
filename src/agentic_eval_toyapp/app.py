"""FastAPI application, dependency wiring, and versioned target endpoints."""

from collections.abc import Callable
from functools import partial
from threading import Lock
from uuid import UUID

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from agentic_eval_toyapp import __version__
from agentic_eval_toyapp.agent import RagAgent, load_system_prompt
from agentic_eval_toyapp.config import AppSettings, get_settings
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
    SessionMessageRequest,
    SessionMessageResponse,
    SessionResponse,
)
from agentic_eval_toyapp.providers import ExtractiveGenerationProvider
from agentic_eval_toyapp.retrieval import FaissRetriever, load_corpus
from agentic_eval_toyapp.sessions import (
    AgentRunner,
    InMemorySessionStore,
    SessionNotFoundError,
)

API_PREFIX = "/api/v1"


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


def _build_default_agent(settings: AppSettings) -> RagAgent:
    """Construct the local semantic retriever and deterministic MVP provider."""
    embedding_provider = SentenceTransformerEmbeddingProvider(
        model_id=settings.embedding_model_id,
        revision=settings.embedding_model_revision,
        device=settings.embedding_device,
        cache_dir=settings.embedding_cache_dir,
    )
    retriever = FaissRetriever(
        corpus=load_corpus(),
        embedding_provider=embedding_provider,
        default_top_k=settings.retrieval_top_k,
        minimum_similarity=settings.retrieval_min_similarity,
    )
    return RagAgent(
        provider=ExtractiveGenerationProvider(),
        retriever=retriever,
        system_prompt=load_system_prompt(),
        max_model_calls=settings.agent_max_model_calls,
        max_searches=settings.agent_max_searches,
        retrieval_top_k=settings.retrieval_top_k,
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


def create_app(
    settings: AppSettings | None = None,
    *,
    session_store: InMemorySessionStore | None = None,
    agent: AgentRunner | None = None,
    agent_factory: Callable[[], AgentRunner] | None = None,
) -> FastAPI:
    """Construct an application with explicit, injectable dependencies."""
    resolved_settings = settings or get_settings()
    resolved_store = session_store or InMemorySessionStore(
        max_turns=resolved_settings.session_max_turns
    )
    if agent is None and agent_factory is None:
        agent_factory = partial(_build_default_agent, resolved_settings)
    resolver = _AgentResolver(agent=agent, factory=agent_factory)

    application = FastAPI(
        title=resolved_settings.app_name,
        version=__version__,
        description="Controlled RAG target application for METIS evaluations.",
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

    @application.get("/health", response_model=HealthResponse, tags=["service"])
    def health() -> HealthResponse:
        """Report process liveness without checking external dependencies."""
        return HealthResponse(
            service=resolved_settings.app_name,
            version=__version__,
        )

    @application.get("/ready", response_model=ReadinessResponse, tags=["service"])
    def ready() -> ReadinessResponse:
        """Initialise and report the dependencies required for chat turns."""
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
        resolved_agent = resolver.get()
        result = resolved_store.run_turn(
            session_id,
            request.message,
            agent=resolved_agent,
        )
        return SessionMessageResponse(session_id=session_id, result=result)

    return application


app = create_app()
