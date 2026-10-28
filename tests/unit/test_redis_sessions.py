"""Offline tests for persistent Redis-backed conversation sessions."""

import asyncio
import threading
from collections.abc import Sequence
from uuid import UUID, uuid4

import pytest
from anyio import to_thread
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

    async def run(
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
    async def run(
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
        self.acquired = False

    async def acquire(self) -> bool:
        self._client.ensure_available()
        self.acquired = self._client.lock_available
        return self.acquired

    async def release(self) -> None:
        self._client.ensure_available()
        if not self.acquired:
            raise AssertionError("cannot release an unowned fake lock")
        self.acquired = False


class _FakeRedisClient:
    def __init__(self) -> None:
        self.data: dict[str, str] = {}
        self.available = True
        self.lock_available = True
        self.locks: list[_FakeLock] = []
        self.closed = False

    def ensure_available(self) -> None:
        if not self.available:
            raise ConnectionError("fake Redis unavailable")

    async def ping(self) -> bool:
        self.ensure_available()
        return True

    async def set(self, name: str, value: str, *, nx: bool = False) -> object:
        self.ensure_available()
        await asyncio.sleep(0)
        if nx and name in self.data:
            return False
        self.data[name] = value
        return True

    async def get(self, name: str) -> str | bytes | None:
        self.ensure_available()
        await asyncio.sleep(0)
        return self.data.get(name)

    def lock(
        self,
        name: str,
        *,
        timeout: float,
        blocking_timeout: float,
    ) -> _FakeLock:
        del name, timeout, blocking_timeout
        lock = self._new_lock()
        self.locks.append(lock)
        return lock

    def _new_lock(self) -> _FakeLock:
        return _FakeLock(self)

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.anyio
async def test_redis_store_persists_sessions_across_store_instances() -> None:
    """A reconstructed app store should see sessions held by Redis."""
    client = _FakeRedisClient()
    first_store = RedisSessionStore(client)
    agent = _RecordingAgent()
    session = await first_store.create()
    await first_store.run_turn(session.session_id, "persistent question", agent=agent)

    reconstructed_store = RedisSessionStore(client)
    snapshot = await reconstructed_store.get(session.session_id)

    assert snapshot.retained_turns == 1
    assert [(message.role, message.content) for message in snapshot.messages] == [
        (ConversationRole.USER, "persistent question"),
        (ConversationRole.ASSISTANT, "Please clarify: persistent question"),
    ]


@pytest.mark.anyio
async def test_redis_store_passes_history_and_trims_complete_turns() -> None:
    """Redis retention should match the bounded in-memory contract."""
    store = RedisSessionStore(_FakeRedisClient(), max_turns=2)
    agent = _RecordingAgent()
    session = await store.create()

    for message in ("first", "second", "third"):
        await store.run_turn(session.session_id, message, agent=agent)

    snapshot = await store.get(session.session_id)
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


@pytest.mark.anyio
async def test_redis_store_rejects_missing_busy_and_unavailable_sessions() -> None:
    """Storage failures should remain distinct from unknown and busy sessions."""
    client = _FakeRedisClient()
    store = RedisSessionStore(client)
    agent = _RecordingAgent()

    with pytest.raises(SessionNotFoundError):
        await store.get(uuid4())

    session = await store.create()
    client.lock_available = False
    with pytest.raises(SessionBusyError):
        await store.run_turn(session.session_id, "question", agent=agent)

    client.available = False
    with pytest.raises(SessionStoreUnavailableError):
        await store.check()
    with pytest.raises(SessionStoreUnavailableError):
        await store.get(session.session_id)
    assert agent.calls == []


@pytest.mark.anyio
async def test_redis_store_does_not_persist_partial_or_invalid_turns() -> None:
    """Agent and payload failures must not produce partial conversation pairs."""
    client = _FakeRedisClient()
    store = RedisSessionStore(client)
    session = await store.create()

    with pytest.raises(RuntimeError, match="agent failed"):
        await store.run_turn(session.session_id, "question", agent=_FailingAgent())
    assert (await store.get(session.session_id)).messages == ()

    session_key = next(key for key in client.data if str(session.session_id) in key)
    client.data[session_key] = '{"version":1,"turns":[{"user":null}]}'
    with pytest.raises(SessionStoreUnavailableError, match="invalid session data"):
        await store.get(session.session_id)


@pytest.mark.anyio
async def test_redis_store_retries_uuid_collisions_and_validates_configuration() -> (
    None
):
    """Persistent session creation must not overwrite an existing Redis key."""
    first_id = UUID("00000000-0000-0000-0000-000000000001")
    second_id = UUID("00000000-0000-0000-0000-000000000002")
    generated_ids = iter((first_id, first_id, second_id))
    store = RedisSessionStore(
        _FakeRedisClient(),
        id_factory=lambda: next(generated_ids),
    )

    assert (await store.create()).session_id == first_id
    assert (await store.create()).session_id == second_id

    with pytest.raises(ValueError, match="max_turns"):
        RedisSessionStore(_FakeRedisClient(), max_turns=0)
    with pytest.raises(ValueError, match="key_prefix"):
        RedisSessionStore(_FakeRedisClient(), key_prefix="  ")
    with pytest.raises(ValueError, match="lock_timeout_seconds"):
        RedisSessionStore(_FakeRedisClient(), lock_timeout_seconds=0)


class _BlockingLock(_FakeLock):
    """Wait on an event in `acquire`, as a busy distributed lock would."""

    def __init__(self, client: "_BlockingLockClient") -> None:
        super().__init__(client)
        self._blocking_client = client

    async def acquire(self) -> bool:
        self._blocking_client.acquire_started.set()
        await self._blocking_client.proceed.wait()
        return await super().acquire()


class _BlockingLockClient(_FakeRedisClient):
    def __init__(self) -> None:
        super().__init__()
        self.proceed = asyncio.Event()
        self.acquire_started = asyncio.Event()

    def _new_lock(self) -> _FakeLock:
        return _BlockingLock(self)


@pytest.mark.anyio
async def test_redis_store_waiter_holds_no_thread_with_the_pool_saturated() -> None:
    """A turn waiting on a busy lock must not need a worker thread to wait."""
    client = _BlockingLockClient()
    store = RedisSessionStore(client)
    session = await store.create()
    limiter = to_thread.current_default_thread_limiter()
    original = limiter.total_tokens
    limiter.total_tokens = 1
    try:
        # Occupy the only worker thread; any threadpool-based wait would hang.
        gate = threading.Event()
        blocker = asyncio.create_task(to_thread.run_sync(gate.wait, 5.0))
        await asyncio.sleep(0.01)
        assert limiter.borrowed_tokens == 1

        waiter = asyncio.create_task(
            store.run_turn(session.session_id, "held", agent=_RecordingAgent())
        )
        await asyncio.wait_for(client.acquire_started.wait(), 1.0)
        client.proceed.set()
        result = await asyncio.wait_for(waiter, 1.0)
        assert result.message == "Please clarify: held"
        gate.set()
        await blocker
    finally:
        limiter.total_tokens = original


@pytest.mark.anyio
async def test_redis_store_releases_the_lock_when_the_turn_raises() -> None:
    """A failing agent must still free the distributed lock."""
    client = _FakeRedisClient()
    store = RedisSessionStore(client)
    session = await store.create()

    with pytest.raises(RuntimeError, match="agent failed"):
        await store.run_turn(session.session_id, "q", agent=_FailingAgent())

    assert len(client.locks) == 1
    assert client.locks[0].acquired is False


class _HangingAgent:
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def run(
        self,
        user_message: str,
        *,
        history: Sequence[ConversationMessage] = (),
        session_id: str | None = None,
        reports: Sequence[StoredReport] = (),
    ) -> AgentTurnResult:
        del user_message, history, session_id, reports
        self.started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


@pytest.mark.anyio
async def test_redis_store_releases_the_lock_when_the_turn_is_cancelled() -> None:
    """Cancelling a running turn must free the lock and persist no partial turn."""
    client = _FakeRedisClient()
    store = RedisSessionStore(client)
    session = await store.create()
    agent = _HangingAgent()
    turn = asyncio.create_task(store.run_turn(session.session_id, "q", agent=agent))
    await asyncio.wait_for(agent.started.wait(), 1.0)

    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn

    assert client.locks[0].acquired is False
    assert (await store.get(session.session_id)).retained_turns == 0


@pytest.mark.anyio
async def test_redis_store_cancel_while_waiting_for_the_lock_leaves_no_lock() -> None:
    """A turn cancelled before acquiring never holds, or tries to release, a lock."""
    client = _BlockingLockClient()
    store = RedisSessionStore(client)
    session = await store.create()
    agent = _RecordingAgent()
    turn = asyncio.create_task(store.run_turn(session.session_id, "q", agent=agent))
    await asyncio.wait_for(client.acquire_started.wait(), 1.0)

    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn

    assert client.locks[0].acquired is False
    assert agent.calls == []
    assert (await store.get(session.session_id)).retained_turns == 0


@pytest.mark.anyio
async def test_redis_store_serialises_same_session_turns_in_order() -> None:
    """Two simultaneous same-session turns both complete, one after the other."""
    client = _FakeRedisClient()
    shared = asyncio.Lock()

    class _SerialLock(_FakeLock):
        async def acquire(self) -> bool:
            await shared.acquire()
            return await super().acquire()

        async def release(self) -> None:
            await super().release()
            shared.release()

    client._new_lock = lambda: _SerialLock(client)  # pyright: ignore[reportPrivateUsage]
    store = RedisSessionStore(client)
    session = await store.create()
    agent = _RecordingAgent()

    first = asyncio.create_task(store.run_turn(session.session_id, "one", agent=agent))
    await asyncio.sleep(0)
    second = asyncio.create_task(store.run_turn(session.session_id, "two", agent=agent))
    await asyncio.gather(first, second)

    snapshot = await store.get(session.session_id)
    assert [m.content for m in snapshot.messages][::2] == ["one", "two"]
    assert [m.content for m in agent.calls[1][1]] == ["one", "Please clarify: one"]


@pytest.mark.anyio
async def test_redis_store_client_lifecycle_is_lazy_idempotent_and_reopenable() -> None:
    """from_url builds no client until open(); aclose closes only an owned client."""
    clients: list[_FakeRedisClient] = []

    def factory() -> _FakeRedisClient:
        clients.append(_FakeRedisClient())
        return clients[-1]

    store = RedisSessionStore(client_factory=factory)
    assert clients == []
    with pytest.raises(SessionStoreUnavailableError, match="not open"):
        await store.check()

    await store.open()
    await store.open()
    assert len(clients) == 1
    await store.check()

    await store.aclose()
    await store.aclose()
    assert clients[0].closed is True
    with pytest.raises(SessionStoreUnavailableError, match="not open"):
        await store.check()

    await store.open()
    assert len(clients) == 2
    await store.aclose()


@pytest.mark.anyio
async def test_redis_store_does_not_close_an_injected_client() -> None:
    """A caller-owned client must survive the store's aclose."""
    client = _FakeRedisClient()
    store = RedisSessionStore(client)

    await store.open()
    await store.aclose()

    assert client.closed is False


def test_redis_store_requires_exactly_one_of_client_or_factory() -> None:
    """Neither or both sources of a client is a configuration error."""
    with pytest.raises(ValueError, match="exactly one"):
        RedisSessionStore()
    with pytest.raises(ValueError, match="exactly one"):
        RedisSessionStore(_FakeRedisClient(), client_factory=_FakeRedisClient)
