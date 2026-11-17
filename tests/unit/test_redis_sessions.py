"""Offline tests for persistent Redis-backed conversation sessions."""

import asyncio
import threading
from collections.abc import Mapping, Sequence
from uuid import UUID, uuid4

import pytest
from anyio import to_thread
from redis.exceptions import ConnectionError
from tests.support.fake_redis_memory import FakeMemoryState

from agentic_eval_toyapp.models import (
    AgentTurnResult,
    AgentTurnStatus,
    ClarifyAction,
    ConversationMessage,
    ConversationRole,
    TerminationReason,
)
from agentic_eval_toyapp.redis_memory import RedisMemoryGuard, StorageCapacityError
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


class _FakeBookkeepingPipeline:
    def __init__(self, client: "_FakeRedisClient") -> None:
        self._client = client
        self._commands: list[tuple[str, tuple[object, ...]]] = []

    def zadd(self, name: str, mapping: Mapping[str, float]) -> object:
        self._commands.append(("zadd", (name, dict(mapping))))
        return self

    def hset(self, name: str, key: str, value: str) -> object:
        self._commands.append(("hset", (name, key, value)))
        return self

    def hdel(self, name: str, *keys: str) -> object:
        self._commands.append(("hdel", (name, *keys)))
        return self

    async def execute(self) -> list[object]:
        self._client.ensure_available()
        await asyncio.sleep(0)
        return [self._client.memory.apply(c, a) for c, a in self._commands]


class _FakeRedisClient:
    """Model the Redis TTL rules the store relies on.

    A plain `SET` clears a key's TTL, `SET ... EX` replaces it, and `EXPIRE`
    on a missing key changes nothing and returns 0.
    """

    def __init__(self) -> None:
        self.data: dict[str, str] = {}
        self.ttls: dict[str, int] = {}
        self.memory = FakeMemoryState()
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

    async def set(
        self, name: str, value: str, *, nx: bool = False, ex: int | None = None
    ) -> object:
        self.ensure_available()
        await asyncio.sleep(0)
        if nx and name in self.data:
            return False
        self.data[name] = value
        if ex is None:
            self.ttls.pop(name, None)
        else:
            self.ttls[name] = ex
        return True

    async def info(self, section: str) -> Mapping[str, object]:
        self.ensure_available()
        return await self.memory.info(section)

    async def eval(self, script: str, numkeys: int, *keys_and_args: str) -> object:
        self.ensure_available()
        return await self.memory.eval(script, numkeys, *keys_and_args)

    def pipeline(self, transaction: bool = True) -> _FakeBookkeepingPipeline:
        assert transaction
        return _FakeBookkeepingPipeline(self)

    async def expire(self, name: str, time: int) -> bool:
        self.ensure_available()
        await asyncio.sleep(0)
        if name not in self.data:
            return False
        self.ttls[name] = time
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
    with pytest.raises(ValueError, match="idle_ttl_seconds"):
        RedisSessionStore(_FakeRedisClient(), idle_ttl_seconds=0)


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


@pytest.mark.anyio
async def test_redis_seed_read_only_returns_read_only_true() -> None:
    """A session seeded with read_only=True reports read_only in its snapshot."""
    client = _FakeRedisClient()
    store = RedisSessionStore(client)
    session_id = UUID("00000000-0000-0000-0000-000000000001")
    user = ConversationMessage(role=ConversationRole.USER, content="hi")
    assistant = ConversationMessage(role=ConversationRole.ASSISTANT, content="hello")
    created = await store.seed(session_id, ((user, assistant),), read_only=True)
    assert created is True

    snapshot = await store.get(session_id)
    assert snapshot.read_only is True


@pytest.mark.anyio
async def test_redis_created_session_is_mutable() -> None:
    """A session created through create() is mutable."""
    store = RedisSessionStore(_FakeRedisClient())
    session = await store.create()
    assert session.read_only is False


@pytest.mark.anyio
async def test_redis_v1_payload_loads_as_mutable() -> None:
    """A version-1 payload (no read_only field) loads as read_only=False."""
    import json

    client = _FakeRedisClient()
    store = RedisSessionStore(client)
    session_id = UUID("00000000-0000-0000-0000-000000000002")

    # Write a v1 payload directly into the fake client
    v1_payload = json.dumps(
        {
            "version": 1,
            "turns": [
                {
                    "user": {"role": "user", "content": "hi"},
                    "assistant": {"role": "assistant", "content": "hello"},
                }
            ],
        }
    )
    key = f"agentic-eval-toyapp:session:{session_id}"
    await client.set(key, v1_payload)

    snapshot = await store.get(session_id)
    assert snapshot.read_only is False
    assert snapshot.retained_turns == 1


_TTL = 3_600


def _session_key(session_id: UUID) -> str:
    return f"agentic-eval-toyapp:session:{session_id}"


def _pair() -> tuple[ConversationMessage, ConversationMessage]:
    return (
        ConversationMessage(role=ConversationRole.USER, content="hi"),
        ConversationMessage(role=ConversationRole.ASSISTANT, content="hello"),
    )


@pytest.mark.anyio
async def test_create_sets_the_idle_ttl_on_the_session_key() -> None:
    client = _FakeRedisClient()
    store = RedisSessionStore(client, idle_ttl_seconds=_TTL)

    session = await store.create()

    assert client.ttls == {_session_key(session.session_id): _TTL}


@pytest.mark.anyio
async def test_every_turn_rewrites_the_session_with_the_idle_ttl() -> None:
    """A plain SET would clear the TTL; every turn must write it again."""
    client = _FakeRedisClient()
    store = RedisSessionStore(client, idle_ttl_seconds=_TTL)
    session_id = (await store.create()).session_id
    key = _session_key(session_id)

    for message in ("one", "two"):
        client.ttls[key] = 5  # as if the session had been idle for a while
        await store.run_turn(session_id, message, agent=_RecordingAgent())
        assert client.ttls[key] == _TTL


@pytest.mark.anyio
async def test_get_refreshes_the_idle_ttl_of_a_mutable_session() -> None:
    client = _FakeRedisClient()
    store = RedisSessionStore(client, idle_ttl_seconds=_TTL)
    session_id = (await store.create()).session_id
    client.ttls[_session_key(session_id)] = 5

    await store.get(session_id)

    assert client.ttls[_session_key(session_id)] == _TTL


@pytest.mark.anyio
async def test_a_seeded_read_only_session_never_gains_a_ttl() -> None:
    client = _FakeRedisClient()
    store = RedisSessionStore(client, idle_ttl_seconds=_TTL)
    session_id = UUID("00000000-0000-0000-0000-0000000000d0")

    assert await store.seed(session_id, (_pair(),), read_only=True) is True
    await store.get(session_id)

    assert client.ttls == {}


@pytest.mark.anyio
async def test_unseed_leaves_an_expiring_empty_session() -> None:
    """A rolled-back demo is an ordinary session, so it must expire."""
    client = _FakeRedisClient()
    store = RedisSessionStore(client, idle_ttl_seconds=_TTL)
    session_id = UUID("00000000-0000-0000-0000-0000000000d1")
    await store.seed(session_id, (_pair(),), read_only=True)

    await store.unseed(session_id)

    assert client.ttls == {_session_key(session_id): _TTL}
    assert (await store.get(session_id)).retained_turns == 0


@pytest.mark.anyio
async def test_get_of_a_missing_session_creates_no_key() -> None:
    client = _FakeRedisClient()
    store = RedisSessionStore(client, idle_ttl_seconds=_TTL)

    with pytest.raises(SessionNotFoundError):
        await store.get(uuid4())

    assert client.data == {}
    assert client.ttls == {}


def _guarded(client: _FakeRedisClient, *, budget: int) -> RedisSessionStore:
    client.memory.maxmemory = budget
    return RedisSessionStore(
        client,
        idle_ttl_seconds=_TTL,
        memory_guard=RedisMemoryGuard(high_water=1.0, clock=lambda: 1_000.0),
    )


@pytest.mark.anyio
async def test_guarded_create_tracks_access_time_and_size() -> None:
    client = _FakeRedisClient()
    store = _guarded(client, budget=1 << 30)

    session_id = str((await store.create()).session_id)

    assert client.memory.lru == {session_id: 1_000.0}
    stored = len(client.data[_session_key(UUID(session_id))])
    assert int(client.memory.sizes[f"{session_id}:session"]) > stored
    assert client.memory.eval_calls == []


@pytest.mark.anyio
async def test_guarded_create_evicts_with_no_exempt_session_when_full() -> None:
    client = _FakeRedisClient()
    store = _guarded(client, budget=10_000)
    client.memory.used_memory = 10_000

    def evict(_args: tuple[str, ...]) -> object:
        client.memory.used_memory = 0
        return ["old", "memory_budget", "5000", "0"]

    client.memory.evict = evict
    await store.create()

    (call,) = client.memory.eval_calls
    assert call[-1] == ""


@pytest.mark.anyio
async def test_guarded_create_refuses_when_nothing_can_be_evicted() -> None:
    client = _FakeRedisClient()
    store = _guarded(client, budget=10_000)
    client.memory.used_memory = 10_000

    with pytest.raises(StorageCapacityError):
        await store.create()
    assert client.data == {}
    assert client.memory.lru == {}


@pytest.mark.anyio
async def test_guarded_turn_is_refused_before_the_agent_runs() -> None:
    """A refused turn must not cost a model call, a trace, or a stored pair."""
    client = _FakeRedisClient()
    store = _guarded(client, budget=1 << 20)
    session_id = (await store.create()).session_id
    client.memory.used_memory = 1 << 20
    agent = _RecordingAgent()

    with pytest.raises(StorageCapacityError):
        await store.run_turn(session_id, "hello", agent=agent)

    assert agent.calls == []
    assert [lock.acquired for lock in client.locks] == [False]
    client.memory.used_memory = 0
    assert (await store.get(session_id)).retained_turns == 0


@pytest.mark.anyio
async def test_guarded_turn_exempts_its_own_session_and_updates_size() -> None:
    client = _FakeRedisClient()
    store = _guarded(client, budget=1 << 20)
    session_id = (await store.create()).session_id
    before = int(client.memory.sizes[f"{session_id}:session"])
    client.memory.used_memory = (1 << 20) - 100

    def evict(_args: tuple[str, ...]) -> object:
        client.memory.used_memory = 0
        return ["other", "memory_budget", "70000", "0"]

    client.memory.evict = evict
    await store.run_turn(session_id, "hello", agent=_RecordingAgent())

    assert client.memory.eval_calls[0][-1] == str(session_id)
    assert int(client.memory.sizes[f"{session_id}:session"]) > before


@pytest.mark.anyio
async def test_guarded_get_renews_access_time_but_not_for_demos() -> None:
    client = _FakeRedisClient()
    now = [1_000.0]
    client.memory.maxmemory = 1 << 30
    store = RedisSessionStore(
        client,
        idle_ttl_seconds=_TTL,
        memory_guard=RedisMemoryGuard(clock=lambda: now[0]),
    )
    session_id = str((await store.create()).session_id)
    demo_id = UUID("00000000-0000-0000-0000-0000000000d2")
    await store.seed(demo_id, (_pair(),), read_only=True)

    now[0] = 2_000.0
    await store.get(UUID(session_id))
    await store.get(demo_id)

    assert client.memory.lru == {session_id: 2_000.0}


@pytest.mark.anyio
async def test_memory_usage_is_none_without_a_guard_and_reports_the_budget() -> None:
    client = _FakeRedisClient()
    assert await RedisSessionStore(client).memory_usage() is None

    usage = await _guarded(client, budget=4_096).memory_usage()
    assert usage is not None
    assert (usage.budget_bytes, usage.source) == (4_096, "maxmemory")
