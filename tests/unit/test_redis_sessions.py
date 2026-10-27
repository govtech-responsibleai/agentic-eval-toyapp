"""Offline tests for persistent Redis-backed conversation sessions."""

from collections.abc import Sequence
from uuid import UUID, uuid4

import pytest
from redis.exceptions import ConnectionError

from agentic_eval_toyapp.models import (
    AgentTurnResult,
    AgentTurnStatus,
    ClarifyAction,
    ConversationMessage,
    ConversationRole,
    TerminationReason,
)
from agentic_eval_toyapp.reports import StoredReport
from agentic_eval_toyapp.sessions import (
    RedisSessionStore,
    SessionBusyError,
    SessionNotFoundError,
    SessionStoreUnavailableError,
)


class _RecordingAgent:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[ConversationMessage, ...]]] = []

    def run(
        self,
        user_message: str,
        *,
        history: Sequence[ConversationMessage] = (),
        session_id: str | None = None,
        reports: Sequence[StoredReport] = (),
    ) -> AgentTurnResult:
        del session_id, reports
        self.calls.append((user_message, tuple(history)))
        return AgentTurnResult(
            status=AgentTurnStatus.CLARIFICATION_REQUIRED,
            message=f"Please clarify: {user_message}",
            termination_reason=TerminationReason.CLARIFICATION_REQUIRED,
            model_calls=1,
            searches=0,
            model_actions=(ClarifyAction(question=f"Please clarify: {user_message}"),),
        )


class _FailingAgent:
    def run(
        self,
        user_message: str,
        *,
        history: Sequence[ConversationMessage] = (),
        session_id: str | None = None,
        reports: Sequence[StoredReport] = (),
    ) -> AgentTurnResult:
        del user_message, history, session_id, reports
        raise RuntimeError("agent failed")


class _FakeLock:
    def __init__(self, client: "_FakeRedisClient") -> None:
        self._client = client
        self._acquired = False

    def acquire(self) -> bool:
        self._client.ensure_available()
        self._acquired = self._client.lock_available
        return self._acquired

    def release(self) -> None:
        self._client.ensure_available()
        if not self._acquired:
            raise AssertionError("cannot release an unowned fake lock")
        self._acquired = False


class _FakeRedisClient:
    def __init__(self) -> None:
        self.data: dict[str, str] = {}
        self.available = True
        self.lock_available = True

    def ensure_available(self) -> None:
        if not self.available:
            raise ConnectionError("fake Redis unavailable")

    def ping(self) -> bool:
        self.ensure_available()
        return True

    def set(self, name: str, value: str, *, nx: bool = False) -> object:
        self.ensure_available()
        if nx and name in self.data:
            return False
        self.data[name] = value
        return True

    def get(self, name: str) -> str | bytes | None:
        self.ensure_available()
        return self.data.get(name)

    def lock(
        self,
        name: str,
        *,
        timeout: float,
        blocking_timeout: float,
    ) -> _FakeLock:
        del name, timeout, blocking_timeout
        return _FakeLock(self)


def test_redis_store_persists_sessions_across_store_instances() -> None:
    """A reconstructed app store should see sessions held by Redis."""
    client = _FakeRedisClient()
    first_store = RedisSessionStore(client)
    agent = _RecordingAgent()
    session = first_store.create()
    first_store.run_turn(session.session_id, "persistent question", agent=agent)

    reconstructed_store = RedisSessionStore(client)
    snapshot = reconstructed_store.get(session.session_id)

    assert snapshot.retained_turns == 1
    assert [(message.role, message.content) for message in snapshot.messages] == [
        (ConversationRole.USER, "persistent question"),
        (ConversationRole.ASSISTANT, "Please clarify: persistent question"),
    ]


def test_redis_store_passes_history_and_trims_complete_turns() -> None:
    """Redis retention should match the bounded in-memory contract."""
    store = RedisSessionStore(_FakeRedisClient(), max_turns=2)
    agent = _RecordingAgent()
    session = store.create()

    for message in ("first", "second", "third"):
        store.run_turn(session.session_id, message, agent=agent)

    snapshot = store.get(session.session_id)
    assert snapshot.retained_turns == 2
    assert [message.content for message in snapshot.messages] == [
        "second",
        "Please clarify: second",
        "third",
        "Please clarify: third",
    ]
    assert [message.content for message in agent.calls[2][1]] == [
        "first",
        "Please clarify: first",
        "second",
        "Please clarify: second",
    ]


def test_redis_store_rejects_missing_busy_and_unavailable_sessions() -> None:
    """Storage failures should remain distinct from unknown and busy sessions."""
    client = _FakeRedisClient()
    store = RedisSessionStore(client)
    agent = _RecordingAgent()

    with pytest.raises(SessionNotFoundError):
        store.get(uuid4())

    session = store.create()
    client.lock_available = False
    with pytest.raises(SessionBusyError):
        store.run_turn(session.session_id, "question", agent=agent)

    client.available = False
    with pytest.raises(SessionStoreUnavailableError):
        store.check()
    with pytest.raises(SessionStoreUnavailableError):
        store.get(session.session_id)
    assert agent.calls == []


def test_redis_store_does_not_persist_partial_or_invalid_turns() -> None:
    """Agent and payload failures must not produce partial conversation pairs."""
    client = _FakeRedisClient()
    store = RedisSessionStore(client)
    session = store.create()

    with pytest.raises(RuntimeError, match="agent failed"):
        store.run_turn(session.session_id, "question", agent=_FailingAgent())
    assert store.get(session.session_id).messages == ()

    session_key = next(key for key in client.data if str(session.session_id) in key)
    client.data[session_key] = '{"version":1,"turns":[{"user":null}]}'
    with pytest.raises(SessionStoreUnavailableError, match="invalid session data"):
        store.get(session.session_id)


def test_redis_store_retries_uuid_collisions_and_validates_configuration() -> None:
    """Persistent session creation must not overwrite an existing Redis key."""
    first_id = UUID("00000000-0000-0000-0000-000000000001")
    second_id = UUID("00000000-0000-0000-0000-000000000002")
    generated_ids = iter((first_id, first_id, second_id))
    store = RedisSessionStore(
        _FakeRedisClient(),
        id_factory=lambda: next(generated_ids),
    )

    assert store.create().session_id == first_id
    assert store.create().session_id == second_id

    with pytest.raises(ValueError, match="max_turns"):
        RedisSessionStore(_FakeRedisClient(), max_turns=0)
    with pytest.raises(ValueError, match="key_prefix"):
        RedisSessionStore(_FakeRedisClient(), key_prefix="  ")
    with pytest.raises(ValueError, match="lock_timeout_seconds"):
        RedisSessionStore(_FakeRedisClient(), lock_timeout_seconds=0)
