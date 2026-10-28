"""Opt-in connectivity test for the configured OpenAI-compatible provider."""

from uuid import uuid4

import pytest

from agentic_eval_toyapp.analysis import (
    OpenAIAnalysisGenerationProvider,
    load_analysis_prompt,
)
from agentic_eval_toyapp.analysis_models import (
    AnalysisAnswerAction,
    AnalysisClarifyAction,
    AnalysisModelRequest,
)
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
    SectionReportLocator,
)
from agentic_eval_toyapp.orchestration_models import (
    CoordinatorActionKind,
    CoordinatorRequest,
    CoordinatorRouteRagAction,
)
from agentic_eval_toyapp.providers import (
    OpenAIChatGenerationProvider,
    OpenAICompatibleChatClient,
)
from agentic_eval_toyapp.report_sources import SourceBlock


@pytest.mark.anyio
@pytest.mark.model_live
async def test_configured_model_returns_a_valid_agent_action() -> None:
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

    action = await provider.generate_action(
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


@pytest.mark.anyio
@pytest.mark.model_live
async def test_configured_model_returns_a_valid_coordinator_route() -> None:
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

    action = await provider.generate_action(
        CoordinatorRequest(
            system_prompt=load_coordinator_prompt(),
            user_message="How many annual leave days do employees receive?",
            available_actions=(CoordinatorActionKind.ROUTE_RAG,),
            model_call_number=1,
        )
    )

    assert isinstance(action, CoordinatorRouteRagAction)


@pytest.mark.anyio
@pytest.mark.model_live
async def test_configured_model_returns_a_valid_report_analysis_action() -> None:
    """The gateway should accept the report specialist's strict action schema."""
    settings = AppSettings()
    if settings.generation_provider is not GenerationProviderKind.OPENAI:
        pytest.fail("TOYAPP_GENERATION_PROVIDER must be openai for the live model test")
    api_key, base_url = settings.require_openai_configuration()
    provider = OpenAIAnalysisGenerationProvider(
        model=settings.generation_model,
        revision=settings.generation_model_revision,
        temperature=settings.generation_temperature,
        max_tokens=settings.analysis_max_output_tokens,
        client=OpenAICompatibleChatClient(
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=settings.generation_timeout_seconds,
            instrument_with_langfuse=False,
        ),
    )
    action = await provider.generate_action(
        AnalysisModelRequest(
            system_prompt=load_analysis_prompt(),
            task="Summarise the revenue finding.",
            blocks=(
                SourceBlock(
                    file_id=uuid4(),
                    locator=SectionReportLocator(section="Document"),
                    text="Revenue increased by 18% in the quarter.",
                ),
            ),
            model_call_number=1,
        )
    )

    assert isinstance(action, AnalysisAnswerAction | AnalysisClarifyAction)
