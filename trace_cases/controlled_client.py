"""Live completion boundary with a local-only effective-response substitution."""

import json
from contextlib import AbstractContextManager
from typing import Any, Protocol, cast

from agentic_eval_toyapp.providers import StructuredCompletionClient
from trace_cases.models import CompletionCapture


class GenerationObservation(Protocol):
    """Subset of a Langfuse generation observation used by the harness."""

    def update(self, **values: object) -> object:
        """Update observable generation fields."""
        ...


class GenerationTraceClient(Protocol):
    """Trace-client boundary used to keep offline tests network-free."""

    def start_as_current_observation(
        self,
        **values: object,
    ) -> AbstractContextManager[GenerationObservation]:
        """Create a normal model-generation observation."""
        ...


class ControlledStructuredCompletionClient:
    """Substitute answer content before it enters normal agent processing.

    The original provider response and the case label are retained only by the
    local harness. Langfuse receives the same ordinary generation shape as the
    application, with the effective structured response as its output.
    """

    def __init__(
        self,
        *,
        delegate: StructuredCompletionClient,
        trace_client: GenerationTraceClient,
        replacement_markdown: str,
    ) -> None:
        stripped_replacement = replacement_markdown.strip()
        if not stripped_replacement:
            raise ValueError("replacement_markdown must not be blank")
        self._delegate = delegate
        self._trace_client = trace_client
        self._replacement_markdown = stripped_replacement
        self._captures: list[CompletionCapture] = []

    @property
    def captures(self) -> tuple[CompletionCapture, ...]:
        """Return private original/effective response pairs for the run manifest."""
        return tuple(self._captures)

    def create(
        self,
        *,
        model: str,
        messages: tuple[dict[str, str], ...],
        response_schema: dict[str, Any],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Make the live call, substitute an answer, and trace only the result."""
        with self._trace_client.start_as_current_observation(
            name="model-action",
            as_type="generation",
            input=list(messages),
            metadata={
                "operation": "structured_action_selection",
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "agent_action",
                        "schema": response_schema,
                        "strict": True,
                    },
                },
            },
            model=model,
            model_parameters={
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": 1,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "service_tier": "default",
            },
        ) as observation:
            try:
                original = self._delegate.create(
                    model=model,
                    messages=messages,
                    response_schema=response_schema,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception:
                observation.update(
                    level="ERROR",
                    status_message="generation_provider_error",
                )
                raise
            effective, altered = self._replace_answer(original)
            observation.update(output={"role": "assistant", "content": effective})
        self._captures.append(
            CompletionCapture(
                original_response=original,
                effective_response=effective,
                altered=altered,
            )
        )
        return effective

    def _replace_answer(self, content: str) -> tuple[str, bool]:
        """Replace only a valid JSON answer payload; preserve other actions."""
        try:
            decoded = cast(object, json.loads(content))
        except json.JSONDecodeError:
            return content, False
        if not isinstance(decoded, dict):
            return content, False
        effective = cast(dict[str, Any], decoded).copy()
        if effective.get("kind") != "answer":
            return content, False
        effective["markdown"] = self._replacement_markdown
        effective["citations"] = []
        encoded = json.dumps(effective, ensure_ascii=False, separators=(",", ":"))
        return encoded, encoded != content
