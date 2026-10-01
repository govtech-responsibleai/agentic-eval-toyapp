"""Tests for deterministic generation providers."""

import pytest

from agentic_eval_toyapp.models import (
    ClarifyAction,
    ModelRequest,
    SearchAction,
)
from agentic_eval_toyapp.providers import (
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
