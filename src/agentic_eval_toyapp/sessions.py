"""Event-loop-safe conversation sessions backed by memory or Redis."""

import asyncio
import json
from collections import deque
from collections.abc import AsyncGenerator, Callable, Iterable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from threading import Lock
from typing import Protocol, cast
from uuid import UUID, uuid4

import anyio
from redis.asyncio import Redis
from redis.exceptions import LockError, RedisError

from agentic_eval_toyapp.models import (
    AgentTurnResult,
    ConversationMessage,
    ConversationRole,
    SessionMessageRequest,
    SessionResponse,
)
from agentic_eval_toyapp.reports import StoredReport


class SessionNotFoundError(LookupError):
    """Raised when a caller references an unknown application session."""


class SessionStoreUnavailableError(RuntimeError):
    """Raised when the configured persistent session store is unavailable."""


class SessionBusyError(RuntimeError):
    """Raised when another worker holds a session turn lock too long."""


class AgentRunner(Protocol):
    """Narrow agent interface required by the session service."""

    async def run(
        self,
        user_message: str,
        *,
        history: Sequence[ConversationMessage] = (),
        session_id: str | None = None,
        reports: Sequence[StoredReport] = (),
    ) -> AgentTurnResult:
        """Execute one turn using explicit history and authorized reports."""
        ...


class SessionStore(Protocol):
    """Storage boundary required by the session API."""

    @property
    def max_turns(self) -> int:
        """Return the per-session retained-turn bound."""
        ...

    @property
    def readiness_detail(self) -> str:
        """Return a non-sensitive description of the configured backend."""
        ...

    async def open(self) -> None:
        """Acquire loop-bound resources; call from the running event loop."""
        ...

    async def aclose(self) -> None:
        """Release resources acquired by `open`."""
        ...

    async def check(self) -> None:
        """Raise when the backing store cannot serve requests."""
        ...

    async def create(self) -> SessionResponse:
        """Create one empty session."""
        ...

    async def get(self, session_id: UUID) -> SessionResponse:
        """Return one safe public session snapshot."""
        ...

    async def run_turn(
        self,
        session_id: UUID,
        user_message: str,
        *,
        agent: AgentRunner,
        reports: Sequence[StoredReport] = (),
    ) -> AgentTurnResult:
        """Run and append one complete turn."""
        ...


@dataclass(frozen=True, slots=True)
class _StoredTurn:
    user: ConversationMessage
    assistant: ConversationMessage


@dataclass(slots=True)
class _SessionRecord:
    turns: deque[_StoredTurn]
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def _flatten_turns(turns: Iterable[_StoredTurn]) -> tuple[ConversationMessage, ...]:
    return tuple(message for turn in turns for message in (turn.user, turn.assistant))


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

    @property
    def readiness_detail(self) -> str:
        """Describe the process-local store without exposing internal state."""
        return f"retains up to {self.max_turns} complete turns"

    async def open(self) -> None:
        """The process-local store has no loop-bound resources."""

    async def aclose(self) -> None:
        """The process-local store has nothing to release."""

    async def check(self) -> None:
        """The process-local store has no external dependency to check."""

    async def create(self) -> SessionResponse:
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

    async def get(self, session_id: UUID) -> SessionResponse:
        """Return a consistent public snapshot of one session."""
        # No await occurs here, so the snapshot is atomic on the event loop and
        # needs no turn lock; callers must therefore run on the loop thread.
        return self._snapshot(session_id, self._get_record(session_id))

    async def run_turn(
        self,
        session_id: UUID,
        user_message: str,
        *,
        agent: AgentRunner,
        reports: Sequence[StoredReport] = (),
    ) -> AgentTurnResult:
        """Run and append one complete turn atomically within its session."""
        validated_request = SessionMessageRequest(message=user_message)
        user = ConversationMessage(
            role=ConversationRole.USER,
            content=validated_request.message,
        )
        record = self._get_record(session_id)
        async with record.lock:
            history = _flatten_turns(record.turns)
            result = await agent.run(
                user.content,
                history=history,
                session_id=str(session_id),
                reports=reports,
            )
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

    def seed(
        self,
        session_id: UUID,
        messages: tuple[tuple[ConversationMessage, ConversationMessage], ...],
    ) -> bool:
        """Insert a session with a predetermined ID; return False if it exists.

        This method is intended for deterministic startup seeding (e.g. a
        bundled demo session).  It is *not* part of the ``SessionStore``
        protocol because only startup-time code should call it.
        """
        with self._records_lock:
            if session_id in self._records:
                return False
            turns: deque[_StoredTurn] = deque(maxlen=self._max_turns)
            for user_msg, assistant_msg in messages:
                turns.append(_StoredTurn(user=user_msg, assistant=assistant_msg))
            self._records[session_id] = _SessionRecord(turns=turns)
        return True

    def unseed(self, session_id: UUID) -> None:
        """Remove a session inserted by ``seed``; no-op if absent."""
        with self._records_lock:
            self._records.pop(session_id, None)

    @classmethod
    def _snapshot(cls, session_id: UUID, record: _SessionRecord) -> SessionResponse:
        return SessionResponse(
            session_id=session_id,
            messages=_flatten_turns(record.turns),
            retained_turns=len(record.turns),
        )


class _RedisLock(Protocol):
    """Subset of the redis.asyncio distributed lock used by the session store."""

    async def acquire(self) -> bool:
        """Acquire the configured lock within its blocking timeout."""
        ...

    async def release(self) -> None:
        """Release a lock owned by this client."""
        ...


class _RedisClient(Protocol):
    """Small redis.asyncio surface kept injectable for offline tests."""

    async def ping(self) -> bool:
        """Check connectivity to the Redis service."""
        ...

    async def set(self, name: str, value: str, *, nx: bool = False) -> object:
        """Set a string value, optionally only when the key is absent."""
        ...

    async def get(self, name: str) -> str | bytes | None:
        """Return one stored string value."""
        ...

    def lock(
        self,
        name: str,
        *,
        timeout: float,
        blocking_timeout: float,
    ) -> _RedisLock:
        """Return a distributed lock for one session."""
        ...

    async def aclose(self) -> None:
        """Close the client and its connection pool."""
        ...


class RedisSessionStore:
    """Persist bounded complete turns in an external Redis service.

    Every Redis call and lock wait is awaited on the event loop, so a turn
    waiting on a busy session holds no worker thread. The async connection pool
    is bound to one event loop, so a store built by `from_url` creates its
    client in `open()` (the application lifespan) and closes it in `aclose()`.
    A client passed to the constructor is owned by the caller and never closed.
    """

    _PAYLOAD_VERSION = 1

    def __init__(
        self,
        client: _RedisClient | None = None,
        *,
        client_factory: Callable[[], _RedisClient] | None = None,
        max_turns: int = 10,
        key_prefix: str = "agentic-eval-toyapp",
        lock_timeout_seconds: float = 1_800.0,
        lock_blocking_timeout_seconds: float = 10.0,
        id_factory: Callable[[], UUID] = uuid4,
    ) -> None:
        if (client is None) == (client_factory is None):
            raise ValueError("provide exactly one of client or client_factory")
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        normalised_key_prefix = key_prefix.strip().rstrip(":")
        if not normalised_key_prefix:
            raise ValueError("key_prefix must not be blank")
        if lock_timeout_seconds <= 0:
            raise ValueError("lock_timeout_seconds must be positive")
        if lock_blocking_timeout_seconds < 0:
            raise ValueError("lock_blocking_timeout_seconds must not be negative")
        self._client = client
        self._client_factory = client_factory
        self._owns_client = False
        self._max_turns = max_turns
        self._key_prefix = normalised_key_prefix
        self._lock_timeout_seconds = lock_timeout_seconds
        self._lock_blocking_timeout_seconds = lock_blocking_timeout_seconds
        self._id_factory = id_factory

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        max_turns: int = 10,
        key_prefix: str = "agentic-eval-toyapp",
        socket_timeout_seconds: float = 5.0,
        lock_timeout_seconds: float = 1_800.0,
        lock_blocking_timeout_seconds: float = 10.0,
    ) -> "RedisSessionStore":
        """Describe a Redis client that `open()` creates on the running loop."""

        def build_client() -> _RedisClient:
            client = Redis.from_url(  # pyright: ignore[reportUnknownMemberType]
                url,
                decode_responses=True,
                socket_connect_timeout=socket_timeout_seconds,
                socket_timeout=socket_timeout_seconds,
            )
            return cast(_RedisClient, client)

        return cls(
            client_factory=build_client,
            max_turns=max_turns,
            key_prefix=key_prefix,
            lock_timeout_seconds=lock_timeout_seconds,
            lock_blocking_timeout_seconds=lock_blocking_timeout_seconds,
        )

    @property
    def max_turns(self) -> int:
        """Return the per-session retained-turn bound."""
        return self._max_turns

    @property
    def readiness_detail(self) -> str:
        """Describe Redis persistence without exposing its connection URL."""
        return f"Redis reachable; retains up to {self.max_turns} complete turns"

    async def open(self) -> None:
        """Create the loop-bound client once; later calls change nothing."""
        if self._client is None and self._client_factory is not None:
            self._client = self._client_factory()
            self._owns_client = True

    async def aclose(self) -> None:
        """Close a client this store created; an injected client is left open."""
        client, self._client = self._client, None
        owned, self._owns_client = self._owns_client, False
        if client is not None and owned:
            try:
                await client.aclose()
            except RedisError as error:
                raise SessionStoreUnavailableError(
                    "Redis client could not be closed"
                ) from error

    def _require_client(self) -> _RedisClient:
        if self._client is None:
            raise SessionStoreUnavailableError(
                "Redis client is not open; run the application lifespan"
            )
        return self._client

    async def check(self) -> None:
        """Verify Redis connectivity without reading application data."""
        client = self._require_client()
        try:
            if not await client.ping():
                raise SessionStoreUnavailableError("Redis ping did not succeed")
        except RedisError as error:
            raise SessionStoreUnavailableError("Redis is unavailable") from error

    async def create(self) -> SessionResponse:
        """Create an empty persistent session using an atomic unique-key claim."""
        client = self._require_client()
        for _ in range(100):
            session_id = self._id_factory()
            try:
                created = await client.set(
                    self._session_key(session_id),
                    self._serialize(()),
                    nx=True,
                )
            except RedisError as error:
                raise SessionStoreUnavailableError("Redis is unavailable") from error
            if created:
                return SessionResponse(session_id=session_id, retained_turns=0)
        raise RuntimeError("could not allocate a unique session ID")

    async def seed(
        self,
        session_id: UUID,
        messages: tuple[tuple[ConversationMessage, ConversationMessage], ...],
    ) -> bool:
        """Atomically insert a session with a predetermined ID via SET NX.

        Returns ``False`` when the key already exists in Redis (another
        worker seeded it first).  Intended for deterministic startup
        seeding; not part of the ``SessionStore`` protocol.
        """
        client = self._require_client()
        stored: list[_StoredTurn] = [
            _StoredTurn(user=user_msg, assistant=assistant_msg)
            for user_msg, assistant_msg in messages
        ]
        try:
            created = await client.set(
                self._session_key(session_id),
                self._serialize(stored),
                nx=True,
            )
        except RedisError as error:
            raise SessionStoreUnavailableError("Redis is unavailable") from error
        return bool(created)

    async def unseed(self, session_id: UUID) -> None:
        """Overwrite a seeded session with an empty payload for rollback."""
        await self._write_turns(session_id, [])

    async def get(self, session_id: UUID) -> SessionResponse:
        """Return one atomically read persistent session snapshot."""
        turns = await self._read_turns(session_id)
        return SessionResponse(
            session_id=session_id,
            messages=_flatten_turns(turns),
            retained_turns=len(turns),
        )

    async def run_turn(
        self,
        session_id: UUID,
        user_message: str,
        *,
        agent: AgentRunner,
        reports: Sequence[StoredReport] = (),
    ) -> AgentTurnResult:
        """Run against stable history and persist a complete pair under a lock."""
        validated_request = SessionMessageRequest(message=user_message)
        user = ConversationMessage(
            role=ConversationRole.USER,
            content=validated_request.message,
        )
        async with self._session_lock(session_id):
            turns = await self._read_turns(session_id)
            result = await agent.run(
                user.content,
                history=_flatten_turns(turns),
                session_id=str(session_id),
                reports=reports,
            )
            assistant = ConversationMessage(
                role=ConversationRole.ASSISTANT,
                content=result.message,
            )
            turns.append(_StoredTurn(user=user, assistant=assistant))
            await self._write_turns(session_id, turns)
            return result

    def _session_key(self, session_id: UUID) -> str:
        return f"{self._key_prefix}:session:{session_id}"

    def _lock_key(self, session_id: UUID) -> str:
        return f"{self._session_key(session_id)}:turn-lock"

    async def _read_turns(self, session_id: UUID) -> deque[_StoredTurn]:
        client = self._require_client()
        try:
            raw = await client.get(self._session_key(session_id))
        except RedisError as error:
            raise SessionStoreUnavailableError("Redis is unavailable") from error
        if raw is None:
            raise SessionNotFoundError
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return self._deserialize(raw)

    async def _write_turns(
        self, session_id: UUID, turns: Iterable[_StoredTurn]
    ) -> None:
        client = self._require_client()
        try:
            stored = await client.set(
                self._session_key(session_id),
                self._serialize(turns),
            )
        except RedisError as error:
            raise SessionStoreUnavailableError("Redis is unavailable") from error
        if not stored:
            raise SessionStoreUnavailableError("Redis did not persist the session")

    @asynccontextmanager
    async def _session_lock(self, session_id: UUID) -> AsyncGenerator[None, None]:
        client = self._require_client()
        try:
            lock = client.lock(
                self._lock_key(session_id),
                timeout=self._lock_timeout_seconds,
                blocking_timeout=self._lock_blocking_timeout_seconds,
            )
            acquired = await lock.acquire()
        except (LockError, RedisError) as error:
            raise SessionStoreUnavailableError(
                "Redis locking is unavailable"
            ) from error
        if not acquired:
            raise SessionBusyError
        try:
            yield
        finally:
            try:
                # Shielded so a cancelled turn still frees the distributed lock.
                with anyio.CancelScope(shield=True):
                    await lock.release()
            except (LockError, RedisError) as error:
                raise SessionStoreUnavailableError(
                    "Redis session lock could not be released"
                ) from error

    def _serialize(self, turns: Iterable[_StoredTurn]) -> str:
        retained = tuple(turns)[-self.max_turns :]
        return json.dumps(
            {
                "version": self._PAYLOAD_VERSION,
                "turns": [
                    {
                        "user": turn.user.model_dump(mode="json"),
                        "assistant": turn.assistant.model_dump(mode="json"),
                    }
                    for turn in retained
                ],
            },
            separators=(",", ":"),
        )

    def _deserialize(self, raw: str) -> deque[_StoredTurn]:
        try:
            decoded = cast(object, json.loads(raw))
            if not isinstance(decoded, dict):
                raise ValueError("session payload must be an object")
            payload = cast(dict[str, object], decoded)
            if payload.get("version") != self._PAYLOAD_VERSION:
                raise ValueError("unsupported session payload version")
            decoded_turns = payload.get("turns")
            if not isinstance(decoded_turns, list):
                raise ValueError("session turns must be a list")
            raw_turns = cast(list[object], decoded_turns)
            turns: deque[_StoredTurn] = deque(maxlen=self.max_turns)
            for decoded_turn in raw_turns:
                if not isinstance(decoded_turn, dict):
                    raise ValueError("session turn must be an object")
                raw_turn = cast(dict[str, object], decoded_turn)
                user = ConversationMessage.model_validate(raw_turn.get("user"))
                assistant = ConversationMessage.model_validate(
                    raw_turn.get("assistant")
                )
                if user.role is not ConversationRole.USER:
                    raise ValueError("session user message has the wrong role")
                if assistant.role is not ConversationRole.ASSISTANT:
                    raise ValueError("session assistant message has the wrong role")
                turns.append(_StoredTurn(user=user, assistant=assistant))
            return turns
        except (TypeError, ValueError) as error:
            raise SessionStoreUnavailableError(
                "Redis contains invalid session data"
            ) from error
