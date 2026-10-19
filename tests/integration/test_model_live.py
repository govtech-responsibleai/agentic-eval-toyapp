"""Opt-in connectivity test for the configured OpenAI-compatible provider."""

import pytest

from agentic_eval_toyapp.config import AppSettings, GenerationProviderKind
from agentic_eval_toyapp.coordinator import (
    build_openai_coordinator_provider,
    load_coordinator_prompt,
)
from agentic_eval_toyapp.models import (
    AnswerAction,
    ClarifyAction,
    ModelRequest,
    SearchAction,
)
from agentic_eval_toyapp.orchestration_models import (
    CoordinatorActionKind,
    CoordinatorRequest,
    CoordinatorRouteRagAction,
)
from agentic_eval_toyapp.providers import OpenAIChatGenerationProvider


@pytest.mark.model_live
def test_configured_model_returns_a_valid_agent_action() -> None:
    """Platform credentials and structured-output support should work together."""
    settings = AppSettings()
    if settings.generation_provider is not GenerationProviderKind.OPENAI:
        pytest.fail("TOYAPP_GENERATION_PROVIDER must be openai for the live model test")
    api_key, base_url = settings.require_openai_configuration()
    provider = OpenAIChatGenerationProvider(
        api_key=api_key,
        base_url=base_url,
        model=settings.generation_model,
        revision=settings.generation_model_revision,
        temperature=settings.generation_temperature,
        max_tokens=settings.generation_max_tokens,
        timeout_seconds=settings.generation_timeout_seconds,
    )

    action = provider.generate_action(
        ModelRequest(
            system_prompt=(
                "Return exactly one structured search action for a policy question."
            ),
            user_message="How many annual leave days do employees receive?",
            model_call_number=1,
            remaining_searches=2,
        )
    )

    assert isinstance(action, SearchAction | ClarifyAction | AnswerAction)


@pytest.mark.model_live
def test_configured_model_returns_a_valid_coordinator_route() -> None:
    """The gateway should accept the coordinator's strict structured schema."""
    settings = AppSettings()
    if settings.generation_provider is not GenerationProviderKind.OPENAI:
        pytest.fail("TOYAPP_GENERATION_PROVIDER must be openai for the live model test")
    api_key, base_url = settings.require_openai_configuration()
    provider = build_openai_coordinator_provider(
        api_key=api_key,
        base_url=base_url,
        model=settings.generation_model,
        revision=settings.generation_model_revision,
        temperature=settings.generation_temperature,
        max_tokens=settings.generation_max_tokens,
        timeout_seconds=settings.generation_timeout_seconds,
    )

    action = provider.generate_action(
        CoordinatorRequest(
            system_prompt=load_coordinator_prompt(),
            user_message="How many annual leave days do employees receive?",
            available_actions=(CoordinatorActionKind.ROUTE_RAG,),
            model_call_number=1,
        )
    )

    assert isinstance(action, CoordinatorRouteRagAction)
