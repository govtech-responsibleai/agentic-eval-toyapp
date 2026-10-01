"""Generation-provider boundary and deterministic offline implementation."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from agentic_eval_toyapp.models import AgentAction, ModelRequest


class GenerationProviderError(RuntimeError):
    """Raised when a generation provider cannot produce a valid action."""


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

    def generate_action(self, request: ModelRequest) -> AgentAction:
        """Return exactly one validated action for the current context."""
        ...


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

    def generate_action(self, request: ModelRequest) -> AgentAction:
        self._requests.append(request)
        if self._position >= len(self._actions):
            raise GenerationProviderError("scripted action sequence is exhausted")
        action = self._actions[self._position]
        self._position += 1
        return action
