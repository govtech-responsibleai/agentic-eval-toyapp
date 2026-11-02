"""Tests for deterministic generation providers."""

import json
import logging
from datetime import date
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import ValidationError

import agentic_eval_toyapp.providers as providers_module
from agentic_eval_toyapp.diagnostics import CompletionDiagnostics
from agentic_eval_toyapp.models import (
    AgentActionKind,
    AnswerAction,
    ClarifyAction,
    ConversationMessage,
    ConversationRole,
    DocumentMetadata,
    ModelRequest,
    RetrievedChunk,
    SearchAction,
)
from agentic_eval_toyapp.providers import (
    ExtractiveGenerationProvider,
    GenerationProvider,
    GenerationProviderError,
    InvalidGenerationActionError,
    OpenAIChatGenerationProvider,
    OpenAICompatibleChatClient,
    ScriptedGenerationProvider,
    StructuredCompletion,
    StructuredCompletionClient,
    _StructuredAgentAction,  # pyright: ignore[reportPrivateUsage]
    make_object_schemas_strict,
)


@pytest.mark.anyio
async def test_openai_client_enables_langfuse_generation_capture_only_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Live tracing should capture model I/O, latency, tokens, and errors."""
    creation_arguments: list[dict[str, object]] = []
    completion_arguments: list[dict[str, object]] = []

    class _FakeCompletions:
        async def create(self, **values: object) -> object:
            completion_arguments.append(values)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'))
                ]
            )

    class _FakeLangfuseOpenAI:
        def __init__(self, **values: object) -> None:
            creation_arguments.append(values)
            self.chat = SimpleNamespace(completions=_FakeCompletions())

    monkeypatch.setattr(
        providers_module,
        "_langfuse_openai_client_class",
        lambda: _FakeLangfuseOpenAI,
    )
    client = OpenAICompatibleChatClient(
        api_key="test-key",
        base_url="https://platform.example/v1/",
        timeout_seconds=12,
        instrument_with_langfuse=True,
    )

    content = await client.create(
        model="platform-model",
        messages=({"role": "user", "content": "Question"},),
        response_schema={"type": "object"},
        temperature=0.0,
        max_tokens=100,
    )

    assert content.content == '{"ok":true}'
    assert content.diagnostics.finish_reason is None
    assert creation_arguments == [
        {
            "api_key": "test-key",
            "base_url": "https://platform.example/v1",
            "timeout": 12,
            "max_retries": 0,
        }
    ]
    assert completion_arguments[0]["name"] == "model-action"
    assert completion_arguments[0]["metadata"] == {
        "operation": "structured_action_selection"
    }


@pytest.mark.anyio
async def test_openai_client_does_not_load_langfuse_wrapper_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An uninstrumented call must not create a duplicate OpenAI generation."""
    completion_arguments: list[dict[str, object]] = []

    class _FakeCompletions:
        async def create(self, **values: object) -> object:
            completion_arguments.append(values)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'))
                ]
            )

    class _FakeOpenAI:
        def __init__(self, **values: object) -> None:
            del values
            self.chat = SimpleNamespace(completions=_FakeCompletions())

    def fail_if_loaded() -> None:
        raise AssertionError("Langfuse OpenAI wrapper should not be loaded")

    monkeypatch.setattr(providers_module, "AsyncOpenAI", _FakeOpenAI)
    monkeypatch.setattr(
        providers_module,
        "_langfuse_openai_client_class",
        fail_if_loaded,
    )
    client = OpenAICompatibleChatClient(
        api_key="test-key",
        base_url="https://platform.example/v1",
        timeout_seconds=12,
        instrument_with_langfuse=False,
    )

    completion = await client.create(
        model="platform-model",
        messages=({"role": "user", "content": "Question"},),
        response_schema={"type": "object"},
        temperature=0.0,
        max_tokens=100,
    )

    assert completion.content == '{"ok":true}'
    assert "name" not in completion_arguments[0]
    assert "metadata" not in completion_arguments[0]


@pytest.mark.anyio
async def test_empty_live_completion_logs_finish_reason_and_reported_usage(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An empty completion retains provider evidence without response text."""

    class _FakeCompletions:
        async def create(self, **values: object) -> object:
            del values
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="length",
                        message=SimpleNamespace(content=None),
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=721,
                    completion_tokens=32,
                    total_tokens=753,
                    completion_tokens_details=SimpleNamespace(reasoning_tokens=32),
                ),
            )

    class _FakeOpenAI:
        def __init__(self, **values: object) -> None:
            del values
            self.chat = SimpleNamespace(completions=_FakeCompletions())

    monkeypatch.setattr(providers_module, "AsyncOpenAI", _FakeOpenAI)
    provider = _live_provider(
        OpenAICompatibleChatClient(
            api_key="private-test-key",
            base_url="https://platform.example/v1",
            timeout_seconds=30,
        )
    )

    with (
        caplog.at_level(logging.ERROR, logger="uvicorn.error"),
        pytest.raises(GenerationProviderError) as failure,
    ):
        await provider.generate_action(_request())

    assert failure.value.code == "empty_content"
    assert len(caplog.records) == 1
    assert "finish_reason=length" in caplog.text
    assert "input_tokens=721 output_tokens=32 reasoning_tokens=32" in caplog.text
    assert "total_tokens=753" in caplog.text
    assert "private-test-key" not in caplog.text


def _request(call_number: int = 1) -> ModelRequest:
    return ModelRequest(
        system_prompt="Ground answers in evidence.",
        user_message="What is the allowance?",
        model_call_number=call_number,
        remaining_searches=2,
    )


@pytest.mark.anyio
async def test_scripted_provider_returns_actions_in_order_and_records_requests() -> (
    None
):
    """The offline provider must be exact and auditable."""
    search = SearchAction(query="travel allowance")
    clarify = ClarifyAction(question="Domestic or international travel?")
    provider = ScriptedGenerationProvider((search, clarify))

    assert isinstance(provider, GenerationProvider)
    assert await provider.generate_action(_request()) is search
    assert await provider.generate_action(_request(2)) is clarify
    assert provider.actions_remaining == 0
    assert [request.model_call_number for request in provider.requests] == [1, 2]


@pytest.mark.anyio
async def test_scripted_provider_fails_explicitly_when_exhausted() -> None:
    """A missing test fixture action must not become random behaviour."""
    provider = ScriptedGenerationProvider((SearchAction(query="policy"),))
    await provider.generate_action(_request())

    with pytest.raises(GenerationProviderError, match="exhausted"):
        await provider.generate_action(_request(2))
    assert len(provider.requests) == 2


def test_scripted_provider_rejects_empty_or_unidentified_fixtures() -> None:
    """Fixture provenance and at least one action are required."""
    with pytest.raises(ValueError, match="one scripted action"):
        ScriptedGenerationProvider(())
    with pytest.raises(ValueError, match="model_id"):
        ScriptedGenerationProvider(
            (SearchAction(query="policy"),),
            model_id=" ",
        )


def _retrieved_chunk(
    chunk_id: str = "TRAVEL-001#1",
    *,
    title: str = "Travel Policy",
    text: str = "Domestic meals are limited to $50 per day.",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=chunk_id.split("#", maxsplit=1)[0],
        title=title,
        text=text,
        metadata=DocumentMetadata(topic="travel", effective_date=date(2026, 1, 1)),
        similarity_score=0.9,
    )


@pytest.mark.anyio
async def test_extractive_provider_searches_with_the_full_user_message() -> None:
    """The deterministic API path should preserve arbitrary user search text."""
    provider = ExtractiveGenerationProvider()

    action = await provider.generate_action(
        ModelRequest(
            system_prompt="Use evidence.",
            user_message="休暇ポリシーは何ですか？",
            model_call_number=1,
            remaining_searches=2,
        )
    )

    assert isinstance(action, SearchAction)
    assert action.query == "休暇ポリシーは何ですか？"
    assert action.top_k == 3
    assert provider.model_id == "deterministic-extractive-provider"
    assert provider.revision == "v1"


@pytest.mark.anyio
async def test_extractive_provider_returns_verbatim_chunks_and_citations() -> None:
    """The pre-model answer should add no factual content beyond evidence."""
    provider = ExtractiveGenerationProvider()
    travel = _retrieved_chunk()
    leave = _retrieved_chunk(
        "LEAVE-001#1",
        title="Leave Policy",
        text="Employees receive 18 annual-leave days.",
    )

    action = await provider.generate_action(
        ModelRequest(
            system_prompt="Use evidence.",
            user_message="Summarise the policies.",
            retrieved_chunks=(travel, leave),
            retrieval_attempted=True,
            model_call_number=2,
            remaining_searches=1,
        )
    )

    assert isinstance(action, AnswerAction)
    assert action.citations == ("TRAVEL-001#1", "LEAVE-001#1")
    assert travel.text in action.markdown
    assert leave.text in action.markdown
    assert "[TRAVEL-001#1]" in action.markdown
    assert "[LEAVE-001#1]" in action.markdown


@pytest.mark.anyio
async def test_extractive_provider_returns_zero_evidence_response_after_search() -> (
    None
):
    """Offline mode should model a response after an empty retrieval."""
    action = await ExtractiveGenerationProvider().generate_action(
        ModelRequest(
            system_prompt="Use evidence.",
            user_message="Unknown policy",
            retrieval_attempted=True,
            model_call_number=2,
            remaining_searches=1,
        )
    )

    assert isinstance(action, AnswerAction)
    assert action.citations == ()
    assert action.markdown == "No relevant evidence was found in the knowledge base."


@pytest.mark.anyio
async def test_extractive_provider_fails_when_no_search_remains() -> None:
    """The provider must not answer without evidence or an available search."""
    provider = ExtractiveGenerationProvider()
    request = ModelRequest(
        system_prompt="Use evidence.",
        user_message="Question",
        available_actions=(AgentActionKind.CLARIFY, AgentActionKind.ANSWER),
        model_call_number=2,
        remaining_searches=0,
    )

    with pytest.raises(GenerationProviderError, match="no search"):
        await provider.generate_action(request)


class _RecordingCompletionClient:
    def __init__(
        self,
        response: str = (
            '{"kind":"search","query":"leave policy","top_k":6,"question":null,'
            '"markdown":null,"citations":[]}'
        ),
        *,
        error: Exception | None = None,
        diagnostics: CompletionDiagnostics | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.diagnostics = diagnostics
        self.calls: list[dict[str, object]] = []

    async def create(
        self,
        *,
        model: str,
        messages: tuple[dict[str, str], ...],
        response_schema: dict[str, Any],
        temperature: float,
        max_tokens: int,
    ) -> str | StructuredCompletion:
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "response_schema": response_schema,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        if self.error is not None:
            raise self.error
        if self.diagnostics is not None:
            return StructuredCompletion(self.response, self.diagnostics)
        return self.response


def _live_provider(
    client: StructuredCompletionClient,
) -> OpenAIChatGenerationProvider:
    return OpenAIChatGenerationProvider(
        api_key="test-key-not-used",
        base_url="https://platform.example/v1",
        model="platform-model",
        revision="revision-7",
        temperature=0.0,
        max_tokens=512,
        timeout_seconds=30.0,
        client=client,
    )


@pytest.mark.anyio
async def test_live_provider_sends_complete_context_and_parses_structured_action() -> (
    None
):
    """The live boundary should preserve public context and return one action."""
    client = _RecordingCompletionClient()
    provider = _live_provider(client)
    history = (
        ConversationMessage(role=ConversationRole.USER, content="Earlier question"),
        ConversationMessage(role=ConversationRole.ASSISTANT, content="Earlier answer"),
    )

    action = await provider.generate_action(
        ModelRequest(
            system_prompt="Search before answering.",
            user_message="休暇について教えてください",
            history=history,
            model_call_number=1,
            remaining_searches=2,
        )
    )

    assert action == SearchAction(query="leave policy", top_k=6)
    assert provider.model_id == "platform-model"
    assert provider.revision == "revision-7"
    call = client.calls[0]
    assert call["model"] == "platform-model"
    assert call["temperature"] == 0.0
    assert call["max_tokens"] == 512
    messages = cast(tuple[dict[str, str], ...], call["messages"])
    assert messages[0] == {
        "role": "system",
        "content": "Search before answering.",
    }
    context = json.loads(messages[1]["content"])
    assert context["available_actions"] == ["search", "clarify", "answer"]
    assert context["action_feedback"] is None
    assert context["user_message"] == "休暇について教えてください"
    assert context["conversation_history"] == [
        {"role": "user", "content": "Earlier question"},
        {"role": "assistant", "content": "Earlier answer"},
    ]
    assert context["retrieval"] == {
        "attempted": False,
        "remaining_searches": 2,
        "default_top_k": 3,
        "maximum_top_k": 8,
        "allowed_citation_ids": [],
        "chunks": [],
    }
    schema = cast(dict[str, Any], call["response_schema"])
    assert schema["additionalProperties"] is False
    assert schema["required"] == [
        "kind",
        "query",
        "top_k",
        "question",
        "markdown",
        "citations",
    ]


@pytest.mark.anyio
async def test_live_provider_accepts_model_generated_zero_evidence_response() -> None:
    """Irrelevant candidates can lead to an uncited observable response."""
    client = _RecordingCompletionClient(
        '{"kind":"answer","query":null,"top_k":null,"question":null,'
        '"markdown":"I found no policy evidence.","citations":[]}'
    )
    candidate = _retrieved_chunk("LEAVE-001#1", text="Annual leave is available.")

    action = await _live_provider(client).generate_action(
        ModelRequest(
            system_prompt="Use evidence.",
            user_message="I need help with sick leave",
            retrieved_chunks=(candidate,),
            retrieval_attempted=True,
            model_call_number=2,
            remaining_searches=1,
        )
    )

    assert action == AnswerAction(
        markdown="I found no policy evidence.",
        citations=(),
    )
    call = client.calls[0]
    messages = cast(tuple[dict[str, str], ...], call["messages"])
    context = json.loads(messages[1]["content"])
    assert context["retrieval"]["allowed_citation_ids"] == ["LEAVE-001#1"]


@pytest.mark.anyio
async def test_live_provider_schema_excludes_actions_unavailable_after_search_limit() -> (
    None
):
    """Structured generation must not offer search after its budget is spent."""
    client = _RecordingCompletionClient(
        '{"kind":"answer","query":null,"top_k":null,"question":null,'
        '"markdown":"No supporting evidence was found.","citations":[]}'
    )
    provider = _live_provider(client)

    await provider.generate_action(
        ModelRequest(
            system_prompt="Use evidence.",
            user_message="I need help with sick leave",
            retrieval_attempted=True,
            action_feedback=(
                "The previous search repeated a query. Answer or clarify instead."
            ),
            available_actions=(AgentActionKind.CLARIFY, AgentActionKind.ANSWER),
            model_call_number=3,
            remaining_searches=0,
        )
    )

    schema = cast(dict[str, Any], client.calls[0]["response_schema"])
    definitions = cast(dict[str, Any], schema["$defs"])
    action_kind = cast(dict[str, Any], definitions["AgentActionKind"])
    assert action_kind["enum"] == ["clarify", "answer"]
    messages = cast(tuple[dict[str, str], ...], client.calls[0]["messages"])
    context = json.loads(messages[1]["content"])
    assert context["action_feedback"] == (
        "The previous search repeated a query. Answer or clarify instead."
    )


@pytest.mark.anyio
async def test_live_provider_rejects_action_outside_current_allowed_set() -> None:
    """Local validation must hold if an upstream ignores the narrowed schema."""
    client = _RecordingCompletionClient(
        '{"kind":"search","query":"policy","top_k":3,"question":null,'
        '"markdown":null,"citations":[]}'
    )
    provider = _live_provider(client)

    with pytest.raises(InvalidGenerationActionError, match="unavailable"):
        await provider.generate_action(
            ModelRequest(
                system_prompt="Answer or clarify.",
                user_message="Question",
                available_actions=(
                    AgentActionKind.CLARIFY,
                    AgentActionKind.ANSWER,
                ),
                model_call_number=3,
                remaining_searches=0,
            )
        )


@pytest.mark.anyio
@pytest.mark.parametrize(
    "response",
    [
        "not-json",
        '{"kind":"unknown","query":null,"question":null,"markdown":null,"citations":[]}',
        '{"kind":"search","query":"","question":null,"markdown":null,"citations":[]}',
        (
            '{"kind":"search","query":"policy","question":"extra",'
            '"markdown":null,"citations":[]}'
        ),
    ],
)
async def test_live_provider_rejects_malformed_structured_output(
    response: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Malformed live output must not enter the bounded agent loop."""
    provider = _live_provider(_RecordingCompletionClient(response))

    with (
        caplog.at_level(logging.ERROR, logger="uvicorn.error"),
        pytest.raises(InvalidGenerationActionError, match="invalid structured action"),
    ):
        await provider.generate_action(_request())

    assert len(caplog.records) == 1
    assert "agent=rag" in caplog.text
    assert "code=invalid_schema" in caplog.text
    assert response not in caplog.text


@pytest.mark.anyio
async def test_invalid_structured_output_retains_provider_usage(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Post-response schema failures carry usage through the RAG adapter."""
    provider = _live_provider(
        _RecordingCompletionClient(
            "not-json",
            diagnostics=CompletionDiagnostics(
                finish_reason="stop", input_tokens=200, output_tokens=9
            ),
        )
    )

    with (
        caplog.at_level(logging.ERROR, logger="uvicorn.error"),
        pytest.raises(InvalidGenerationActionError),
    ):
        await provider.generate_action(_request())

    assert "code=invalid_schema" in caplog.text
    assert "finish_reason=stop input_tokens=200 output_tokens=9" in caplog.text


@pytest.mark.anyio
async def test_live_provider_converts_transport_errors_without_leaking_details(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Provider errors should retain their cause but expose no credential detail."""
    provider = _live_provider(
        _RecordingCompletionClient(error=RuntimeError("secret upstream detail"))
    )

    with (
        caplog.at_level(logging.ERROR, logger="uvicorn.error"),
        pytest.raises(
            GenerationProviderError, match="live model request failed"
        ) as exc,
    ):
        await provider.generate_action(_request())

    assert "secret upstream detail" not in str(exc.value)
    assert len(caplog.records) == 1
    assert "code=unexpected_error" in caplog.text
    assert "agent=rag" in caplog.text
    assert "secret upstream detail" not in caplog.text


@pytest.mark.anyio
async def test_live_provider_logs_empty_completion_category(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An empty gateway response should not look like a transport failure."""
    provider = _live_provider(
        _RecordingCompletionClient(
            error=GenerationProviderError(
                "live model returned no structured content", code="empty_content"
            )
        )
    )

    with (
        caplog.at_level(logging.ERROR, logger="uvicorn.error"),
        pytest.raises(GenerationProviderError),
    ):
        await provider.generate_action(_request())

    assert len(caplog.records) == 1
    assert "code=empty_content" in caplog.text


_AGENT_FULL: dict[str, object] = {
    "kind": "answer",
    "query": None,
    "top_k": None,
    "question": None,
    "markdown": "Leave is 14 days.",
    "citations": ["chunk-1"],
}


def test_structured_agent_action_omitted_nullables_equal_explicit_nulls() -> None:
    omitted = _StructuredAgentAction.model_validate(
        {"kind": "answer", "markdown": "Leave is 14 days.", "citations": ["chunk-1"]}
    )

    assert omitted == _StructuredAgentAction.model_validate(_AGENT_FULL)
    assert (omitted.query, omitted.top_k, omitted.question) == (None, None, None)


def test_structured_agent_action_accepts_omitted_nullables_for_search() -> None:
    action = _StructuredAgentAction.model_validate(
        {"kind": "search", "query": "leave policy", "top_k": 4, "citations": []}
    )

    assert action.to_domain_action() == SearchAction(query="leave policy", top_k=4)


@pytest.mark.parametrize(
    "reply",
    [
        {"kind": "search", "query": "leave", "markdown": "Hi", "citations": []},
        {"kind": "clarify", "question": "Which?", "top_k": 3, "citations": []},
        {"kind": "search", "query": "leave", "citations": []},
        {**_AGENT_FULL, "unexpected": 1},
        {"kind": "answer", "markdown": "Hi"},
    ],
    ids=[
        "search-with-markdown",
        "clarify-with-top-k",
        "search-without-top-k",
        "extra-key",
        "omitted-list",
    ],
)
def test_structured_agent_action_still_rejects_invalid_replies(
    reply: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        _StructuredAgentAction.model_validate(reply)


def test_structured_agent_action_schema_required_list_is_unchanged() -> None:
    schema = _StructuredAgentAction.model_json_schema()
    make_object_schemas_strict(schema)

    assert schema["required"] == [
        "kind",
        "query",
        "top_k",
        "question",
        "markdown",
        "citations",
    ]
    assert schema["additionalProperties"] is False
