"""Generation-provider boundary, deterministic fixtures, and live adapter."""

import json
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from importlib import import_module
from time import perf_counter
from typing import Any, Protocol, cast, runtime_checkable

from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from agentic_eval_toyapp.diagnostics import CompletionDiagnostics, log_provider_failure
from agentic_eval_toyapp.models import (
    AgentAction,
    AgentActionKind,
    AnswerAction,
    ClarifyAction,
    ModelRequest,
    NonEmptyString,
    SearchAction,
)
from agentic_eval_toyapp.multi_agent_tracing import record_completion_usage


class GenerationProviderError(RuntimeError):
    """Raised when a generation provider cannot produce a valid action."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "provider_error",
        completion: CompletionDiagnostics | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.completion = completion


class InvalidGenerationActionError(GenerationProviderError):
    """Raised when a provider response can be retried with schema feedback."""


@dataclass(frozen=True, slots=True)
class StructuredCompletion:
    """Model text plus optional safe metadata from the provider response."""

    content: str
    diagnostics: CompletionDiagnostics


def unpack_completion(
    value: str | StructuredCompletion,
) -> tuple[str, CompletionDiagnostics | None]:
    """Support deterministic string fakes and metadata-bearing live responses."""
    if isinstance(value, StructuredCompletion):
        return value.content, value.diagnostics
    return value, None


def _langfuse_openai_client_class() -> Any:
    """Load the globally instrumenting Langfuse wrapper only when requested."""
    langfuse_openai_module = cast(Any, import_module("langfuse.openai"))
    return langfuse_openai_module.AsyncOpenAI


@runtime_checkable
class GenerationProvider(Protocol):
    """Narrow interface for selecting one validated agent action."""

    @property
    def model_id(self) -> str:
        """Return the generation model or fixture identifier."""
        ...

    @property
    def revision(self) -> str:
        """Return the generation model or fixture revision."""
        ...

    async def generate_action(self, request: ModelRequest) -> AgentAction:
        """Return exactly one validated action for the current context."""
        ...


class StructuredCompletionClient(Protocol):
    """Transport boundary for one OpenAI-compatible structured completion."""

    async def create(
        self,
        *,
        model: str,
        messages: tuple[dict[str, str], ...],
        response_schema: dict[str, Any],
        temperature: float,
        max_tokens: int,
    ) -> str | StructuredCompletion:
        """Return text, optionally with provider-reported completion metadata."""
        ...


class OpenAICompatibleChatClient:
    """Small asynchronous client for an OpenAI-wire-compatible endpoint."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        timeout_seconds: float,
        instrument_with_langfuse: bool = False,
    ) -> None:
        client_class: Any = (
            _langfuse_openai_client_class() if instrument_with_langfuse else AsyncOpenAI
        )
        self._instrument_with_langfuse = instrument_with_langfuse
        self._client: Any = client_class(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
            max_retries=0,
        )

    async def create(
        self,
        *,
        model: str,
        messages: tuple[dict[str, str], ...],
        response_schema: dict[str, Any],
        temperature: float,
        max_tokens: int,
    ) -> StructuredCompletion:
        """Request one strict JSON-schema response and retain safe metadata."""
        tracing_arguments: dict[str, object] = {}
        if self._instrument_with_langfuse:
            tracing_arguments = {
                "name": "model-action",
                "metadata": {"operation": "structured_action_selection"},
            }
        response: Any = await self._client.chat.completions.create(
            model=model,
            messages=list(messages),
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "agent_action",
                    "schema": response_schema,
                    "strict": True,
                },
            },
            temperature=temperature,
            max_tokens=max_tokens,
            **tracing_arguments,
        )
        diagnostics = CompletionDiagnostics.from_response(response)
        record_completion_usage(diagnostics)
        if not response.choices:
            raise GenerationProviderError(
                "live model returned no choices",
                code="empty_choices",
                completion=diagnostics,
            )
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise GenerationProviderError(
                "live model returned no structured content",
                code="empty_content",
                completion=diagnostics,
            )
        return StructuredCompletion(content=content, diagnostics=diagnostics)


class _StructuredAgentAction(BaseModel):
    """Gateway-friendly flat representation of one domain agent action."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: AgentActionKind
    query: NonEmptyString | None
    top_k: int | None = None
    question: NonEmptyString | None
    markdown: NonEmptyString | None
    citations: tuple[NonEmptyString, ...]

    @model_validator(mode="after")
    def validate_action_fields(self) -> "_StructuredAgentAction":
        """Require exactly the fields that belong to the selected action."""
        if self.kind is AgentActionKind.SEARCH:
            if (
                self.query is None
                or self.top_k is None
                or any(value is not None for value in (self.question, self.markdown))
            ):
                raise ValueError("search requires only query and top_k")
            if not 1 <= self.top_k <= 8:
                raise ValueError("search top_k must be between 1 and 8")
            if self.citations:
                raise ValueError("search cannot declare citations")
        elif self.kind is AgentActionKind.CLARIFY:
            if self.question is None or any(
                value is not None for value in (self.query, self.top_k, self.markdown)
            ):
                raise ValueError("clarify requires only question")
            if self.citations:
                raise ValueError("clarify cannot declare citations")
        elif self.markdown is None or any(
            value is not None for value in (self.query, self.top_k, self.question)
        ):
            raise ValueError("answer requires only markdown and citations")
        return self

    def to_domain_action(self) -> AgentAction:
        """Convert the transport schema into the existing domain union."""
        if self.kind is AgentActionKind.SEARCH:
            if self.query is None or self.top_k is None:  # Defensive.
                raise AssertionError("validated search parameters are unavailable")
            return SearchAction(query=self.query, top_k=self.top_k)
        if self.kind is AgentActionKind.CLARIFY:
            if self.question is None:  # Defensive; validated above.
                raise AssertionError("validated clarification is unavailable")
            return ClarifyAction(question=self.question)
        if self.markdown is None:  # Defensive; validated above.
            raise AssertionError("validated answer Markdown is unavailable")
        return AnswerAction(markdown=self.markdown, citations=self.citations)


def make_object_schemas_strict(node: object) -> None:
    """Require every declared property in nested JSON-schema objects."""
    if isinstance(node, dict):
        mapping = cast(dict[str, object], node)
        properties = mapping.get("properties")
        if isinstance(properties, dict):
            typed_properties = cast(dict[str, object], properties)
            mapping["additionalProperties"] = False
            mapping["required"] = list(typed_properties)
        for value in mapping.values():
            make_object_schemas_strict(value)
    elif isinstance(node, list):
        for value in cast(list[object], node):
            make_object_schemas_strict(value)


def _agent_action_schema() -> dict[str, Any]:
    schema = _StructuredAgentAction.model_json_schema()
    make_object_schemas_strict(schema)
    return schema


class OpenAIChatGenerationProvider:
    """Select validated agent actions through an OpenAI-compatible model."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        revision: str,
        temperature: float,
        max_tokens: int,
        timeout_seconds: float,
        instrument_with_langfuse: bool = False,
        client: StructuredCompletionClient | None = None,
    ) -> None:
        if not api_key.strip() or not base_url.strip():
            raise ValueError("api_key and base_url must not be blank")
        if not model.strip() or not revision.strip():
            raise ValueError("model and revision must not be blank")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._model_id = model.strip()
        self._revision = revision.strip()
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = client or OpenAICompatibleChatClient(
            api_key=api_key.strip(),
            base_url=base_url.strip(),
            timeout_seconds=timeout_seconds,
            instrument_with_langfuse=instrument_with_langfuse,
        )
        self._response_schema = _agent_action_schema()

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def revision(self) -> str:
        return self._revision

    async def generate_action(self, request: ModelRequest) -> AgentAction:
        """Ask the model for one action and validate the structured response."""
        response_schema = deepcopy(self._response_schema)
        definitions = cast(dict[str, Any], response_schema["$defs"])
        action_kind = cast(dict[str, Any], definitions["AgentActionKind"])
        action_kind["enum"] = [action.value for action in request.available_actions]
        context = {
            "task": "Choose exactly one next action for the current agent turn.",
            "available_actions": [action.value for action in request.available_actions],
            "action_feedback": request.action_feedback,
            "user_message": request.user_message,
            "conversation_history": [
                message.model_dump(mode="json") for message in request.history
            ],
            "retrieval": {
                "attempted": request.retrieval_attempted,
                "remaining_searches": request.remaining_searches,
                "default_top_k": request.default_top_k,
                "maximum_top_k": 8,
                "allowed_citation_ids": [
                    chunk.chunk_id for chunk in request.retrieved_chunks
                ],
                "chunks": [
                    chunk.model_dump(mode="json") for chunk in request.retrieved_chunks
                ],
            },
            "model_call_number": request.model_call_number,
        }
        messages = (
            {"role": "system", "content": request.system_prompt},
            {
                "role": "user",
                "content": json.dumps(context, ensure_ascii=False),
            },
        )
        started_at = perf_counter()
        try:
            completion = await self._client.create(
                model=self._model_id,
                messages=messages,
                response_schema=response_schema,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            content, diagnostics = unpack_completion(completion)
        except GenerationProviderError as error:
            log_provider_failure(
                agent="rag",
                model=self._model_id,
                attempt=request.model_call_number,
                max_tokens=self._max_tokens,
                started_at=started_at,
                error=error,
                code=error.code,
            )
            raise
        except Exception as error:
            log_provider_failure(
                agent="rag",
                model=self._model_id,
                attempt=request.model_call_number,
                max_tokens=self._max_tokens,
                started_at=started_at,
                error=error,
            )
            raise GenerationProviderError("live model request failed") from error
        try:
            structured_action = _StructuredAgentAction.model_validate_json(content)
        except ValidationError as error:
            log_provider_failure(
                agent="rag",
                model=self._model_id,
                attempt=request.model_call_number,
                max_tokens=self._max_tokens,
                started_at=started_at,
                error=error,
                code="invalid_schema",
                completion=diagnostics,
            )
            raise InvalidGenerationActionError(
                "live model returned an invalid structured action"
            ) from error
        if structured_action.kind not in request.available_actions:
            error = InvalidGenerationActionError(
                "live model returned an unavailable structured action"
            )
            log_provider_failure(
                agent="rag",
                model=self._model_id,
                attempt=request.model_call_number,
                max_tokens=self._max_tokens,
                started_at=started_at,
                error=error,
                code="unavailable_action",
                completion=diagnostics,
            )
            raise error
        return structured_action.to_domain_action()


class ScriptedGenerationProvider:
    """Return a fixed action sequence for deterministic offline tests."""

    def __init__(
        self,
        actions: Sequence[AgentAction],
        *,
        model_id: str = "scripted-generation-fixture",
        revision: str = "v1",
    ) -> None:
        if not actions:
            raise ValueError("at least one scripted action is required")
        if not model_id.strip() or not revision.strip():
            raise ValueError("model_id and revision must not be blank")
        self._actions = tuple(actions)
        self._model_id = model_id.strip()
        self._revision = revision.strip()
        self._position = 0
        self._requests: list[ModelRequest] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def revision(self) -> str:
        return self._revision

    @property
    def requests(self) -> tuple[ModelRequest, ...]:
        """Return immutable request history for test assertions."""
        return tuple(self._requests)

    @property
    def actions_remaining(self) -> int:
        """Return the number of scripted actions not yet consumed."""
        return len(self._actions) - self._position

    async def generate_action(self, request: ModelRequest) -> AgentAction:
        self._requests.append(request)
        if self._position >= len(self._actions):
            raise GenerationProviderError("scripted action sequence is exhausted")
        action = self._actions[self._position]
        self._position += 1
        return action


class ExtractiveGenerationProvider:
    """Build deterministic search and cited extractive-answer actions."""

    @property
    def model_id(self) -> str:
        return "deterministic-extractive-provider"

    @property
    def revision(self) -> str:
        return "v1"

    async def generate_action(self, request: ModelRequest) -> AgentAction:
        """Search once, then quote all retrieved chunks with exact citations."""
        if not request.retrieved_chunks:
            if request.retrieval_attempted:
                return AnswerAction(
                    markdown="No relevant evidence was found in the knowledge base.",
                    citations=(),
                )
            if request.remaining_searches < 1:
                raise GenerationProviderError("no search action remains")
            return SearchAction(
                query=request.user_message,
                top_k=request.default_top_k,
            )

        citations = tuple(chunk.chunk_id for chunk in request.retrieved_chunks)
        sections = tuple(
            f"### {chunk.title}\n\n{chunk.text} [{chunk.chunk_id}]"
            for chunk in request.retrieved_chunks
        )
        return AnswerAction(
            markdown="## Retrieved policy information\n\n" + "\n\n".join(sections),
            citations=citations,
        )
