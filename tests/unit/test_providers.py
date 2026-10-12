"""Tests for deterministic generation providers."""

import json
from datetime import date
from types import SimpleNamespace
from typing import Any, cast

import pytest

import agentic_eval_toyapp.providers as providers_module
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
)


def test_openai_client_enables_langfuse_generation_capture_only_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Live tracing should capture model I/O, latency, tokens, and errors."""
    creation_arguments: list[dict[str, object]] = []
    completion_arguments: list[dict[str, object]] = []

    class _FakeCompletions:
        def create(self, **values: object) -> object:
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

    content = client.create(
        model="platform-model",
        messages=({"role": "user", "content": "Question"},),
        response_schema={"type": "object"},
        temperature=0.0,
        max_tokens=100,
    )

    assert content == '{"ok":true}'
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


def test_openai_client_does_not_load_langfuse_wrapper_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An uninstrumented call must not create a duplicate OpenAI generation."""
    completion_arguments: list[dict[str, object]] = []

    class _FakeCompletions:
        def create(self, **values: object) -> object:
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

    monkeypatch.setattr(providers_module, "OpenAI", _FakeOpenAI)
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

    assert (
        client.create(
            model="platform-model",
            messages=({"role": "user", "content": "Question"},),
            response_schema={"type": "object"},
            temperature=0.0,
            max_tokens=100,
        )
        == '{"ok":true}'
    )
    assert "name" not in completion_arguments[0]
    assert "metadata" not in completion_arguments[0]


def _request(call_number: int = 1) -> ModelRequest:
    return ModelRequest(
        system_prompt="Ground answers in evidence.",
        user_message="What is the allowance?",
        model_call_number=call_number,
        remaining_searches=2,
    )


def test_scripted_provider_returns_actions_in_order_and_records_requests() -> None:
    """The offline provider must be exact and auditable."""
    search = SearchAction(query="travel allowance")
    clarify = ClarifyAction(question="Domestic or international travel?")
    provider = ScriptedGenerationProvider((search, clarify))

    assert isinstance(provider, GenerationProvider)
    assert provider.generate_action(_request()) is search
    assert provider.generate_action(_request(2)) is clarify
    assert provider.actions_remaining == 0
    assert [request.model_call_number for request in provider.requests] == [1, 2]


def test_scripted_provider_fails_explicitly_when_exhausted() -> None:
    """A missing test fixture action must not become random behaviour."""
    provider = ScriptedGenerationProvider((SearchAction(query="policy"),))
    provider.generate_action(_request())

    with pytest.raises(GenerationProviderError, match="exhausted"):
        provider.generate_action(_request(2))
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


def test_extractive_provider_searches_with_the_full_user_message() -> None:
    """The deterministic API path should preserve arbitrary user search text."""
    provider = ExtractiveGenerationProvider()

    action = provider.generate_action(
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


def test_extractive_provider_returns_verbatim_chunks_and_citations() -> None:
    """The pre-model answer should add no factual content beyond evidence."""
    provider = ExtractiveGenerationProvider()
    travel = _retrieved_chunk()
    leave = _retrieved_chunk(
        "LEAVE-001#1",
        title="Leave Policy",
        text="Employees receive 18 annual-leave days.",
    )

    action = provider.generate_action(
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


def test_extractive_provider_returns_zero_evidence_response_after_search() -> None:
    """Offline mode should model a response after an empty retrieval."""
    action = ExtractiveGenerationProvider().generate_action(
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


def test_extractive_provider_fails_when_no_search_remains() -> None:
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
        provider.generate_action(request)


class _RecordingCompletionClient:
    def __init__(
        self,
        response: str = (
            '{"kind":"search","query":"leave policy","top_k":6,"question":null,'
            '"markdown":null,"citations":[]}'
        ),
        *,
        error: Exception | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, object]] = []

    def create(
        self,
        *,
        model: str,
        messages: tuple[dict[str, str], ...],
        response_schema: dict[str, Any],
        temperature: float,
        max_tokens: int,
    ) -> str:
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
        return self.response


def _live_provider(
    client: _RecordingCompletionClient,
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


def test_live_provider_sends_complete_context_and_parses_structured_action() -> None:
    """The live boundary should preserve public context and return one action."""
    client = _RecordingCompletionClient()
    provider = _live_provider(client)
    history = (
        ConversationMessage(role=ConversationRole.USER, content="Earlier question"),
        ConversationMessage(role=ConversationRole.ASSISTANT, content="Earlier answer"),
    )

    action = provider.generate_action(
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


def test_live_provider_accepts_model_generated_zero_evidence_response() -> None:
    """Irrelevant candidates can lead to an uncited observable response."""
    client = _RecordingCompletionClient(
        '{"kind":"answer","query":null,"top_k":null,"question":null,'
        '"markdown":"I found no policy evidence.","citations":[]}'
    )
    candidate = _retrieved_chunk("LEAVE-001#1", text="Annual leave is available.")

    action = _live_provider(client).generate_action(
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


def test_live_provider_schema_excludes_actions_unavailable_after_search_limit() -> None:
    """Structured generation must not offer search after its budget is spent."""
    client = _RecordingCompletionClient(
        '{"kind":"answer","query":null,"top_k":null,"question":null,'
        '"markdown":"No supporting evidence was found.","citations":[]}'
    )
    provider = _live_provider(client)

    provider.generate_action(
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


def test_live_provider_rejects_action_outside_current_allowed_set() -> None:
    """Local validation must hold if an upstream ignores the narrowed schema."""
    client = _RecordingCompletionClient(
        '{"kind":"search","query":"policy","top_k":3,"question":null,'
        '"markdown":null,"citations":[]}'
    )
    provider = _live_provider(client)

    with pytest.raises(InvalidGenerationActionError, match="unavailable"):
        provider.generate_action(
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
def test_live_provider_rejects_malformed_structured_output(response: str) -> None:
    """Malformed live output must not enter the bounded agent loop."""
    provider = _live_provider(_RecordingCompletionClient(response))

    with pytest.raises(InvalidGenerationActionError, match="invalid structured action"):
        provider.generate_action(_request())


def test_live_provider_converts_transport_errors_without_leaking_details() -> None:
    """Provider errors should retain their cause but expose no credential detail."""
    provider = _live_provider(
        _RecordingCompletionClient(error=RuntimeError("secret upstream detail"))
    )

    with pytest.raises(
        GenerationProviderError, match="live model request failed"
    ) as exc:
        provider.generate_action(_request())

    assert "secret upstream detail" not in str(exc.value)
