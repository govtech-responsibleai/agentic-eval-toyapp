"""Tests for deterministic generation providers."""

from datetime import date

import pytest

from agentic_eval_toyapp.models import (
    AnswerAction,
    ClarifyAction,
    DocumentMetadata,
    ModelRequest,
    RetrievedChunk,
    SearchAction,
)
from agentic_eval_toyapp.providers import (
    ExtractiveGenerationProvider,
    GenerationProvider,
    GenerationProviderError,
    ScriptedGenerationProvider,
)


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


def test_extractive_provider_fails_when_no_search_remains() -> None:
    """The provider must not answer without evidence or an available search."""
    provider = ExtractiveGenerationProvider()
    request = ModelRequest(
        system_prompt="Use evidence.",
        user_message="Question",
        model_call_number=2,
        remaining_searches=0,
    )

    with pytest.raises(GenerationProviderError, match="no search"):
        provider.generate_action(request)
