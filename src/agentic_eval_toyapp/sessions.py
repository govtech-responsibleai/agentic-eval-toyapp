"""Thread-safe, process-local conversation sessions for the MVP."""

from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from threading import Lock
from typing import Protocol
from uuid import UUID, uuid4

from agentic_eval_toyapp.models import (
    AgentTurnResult,
    ConversationMessage,
    ConversationRole,
    SessionMessageRequest,
    SessionResponse,
)


class SessionNotFoundError(LookupError):
    """Raised when a caller references an unknown application session."""


class AgentRunner(Protocol):
    """Narrow agent interface required by the session service."""

    def run(
        self,
        user_message: str,
        *,
        history: Sequence[ConversationMessage] = (),
    ) -> AgentTurnResult:
        """Execute one agent turn using explicit prior conversation history."""
        ...


@dataclass(frozen=True, slots=True)
class _StoredTurn:
    user: ConversationMessage
    assistant: ConversationMessage


@dataclass(slots=True)
class _SessionRecord:
    turns: deque[_StoredTurn]
    lock: Lock = field(default_factory=Lock)


class InMemorySessionStore:
    """Retain a bounded number of complete turns for each process-local session."""

    def __init__(
        self,
        *,
        max_turns: int = 10,
        id_factory: Callable[[], UUID] = uuid4,
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        self._max_turns = max_turns
        self._id_factory = id_factory
        self._records: dict[UUID, _SessionRecord] = {}
        self._records_lock = Lock()

    @property
    def max_turns(self) -> int:
        """Return the per-session retained-turn bound."""
        return self._max_turns

    def create(self) -> SessionResponse:
        """Create an isolated empty session with a unique UUID."""
        with self._records_lock:
            session_id: UUID | None = None
            for _ in range(100):
                candidate = self._id_factory()
                if candidate not in self._records:
                    session_id = candidate
                    self._records[candidate] = _SessionRecord(
                        turns=deque(maxlen=self._max_turns)
                    )
                    break
            if session_id is None:
                raise RuntimeError("could not allocate a unique session ID")
        return SessionResponse(session_id=session_id, retained_turns=0)

    def get(self, session_id: UUID) -> SessionResponse:
        """Return a consistent public snapshot of one session."""
        record = self._get_record(session_id)
        with record.lock:
            return self._snapshot(session_id, record)

    def run_turn(
        self,
        session_id: UUID,
        user_message: str,
        *,
        agent: AgentRunner,
    ) -> AgentTurnResult:
        """Run and append one complete turn atomically within its session."""
        validated_request = SessionMessageRequest(message=user_message)
        user = ConversationMessage(
            role=ConversationRole.USER,
            content=validated_request.message,
        )
        record = self._get_record(session_id)
        with record.lock:
            history = self._flatten(record.turns)
            result = agent.run(user.content, history=history)
            assistant = ConversationMessage(
                role=ConversationRole.ASSISTANT,
                content=result.message,
            )
            record.turns.append(_StoredTurn(user=user, assistant=assistant))
            return result

    def _get_record(self, session_id: UUID) -> _SessionRecord:
        with self._records_lock:
            try:
                return self._records[session_id]
            except KeyError as error:
                raise SessionNotFoundError from error

    @staticmethod
    def _flatten(turns: Iterable[_StoredTurn]) -> tuple[ConversationMessage, ...]:
        return tuple(
            message for turn in turns for message in (turn.user, turn.assistant)
        )

    @classmethod
    def _snapshot(cls, session_id: UUID, record: _SessionRecord) -> SessionResponse:
        return SessionResponse(
            session_id=session_id,
            messages=cls._flatten(record.turns),
            retained_turns=len(record.turns),
        )
