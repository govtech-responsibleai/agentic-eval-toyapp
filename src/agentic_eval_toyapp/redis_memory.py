"""Keep Redis under an app-side memory budget by evicting idle sessions.

The budget is derived from `INFO memory` only, never `CONFIG`, so it works on
providers that block admin commands, under any eviction policy, and with any
number of app instances sharing one Redis. All bookkeeping lives in Redis:

    <prefix>:sessions:lru    sorted set, session ID -> last-access time
    <prefix>:sessions:bytes  hash, "<session>:session" / "<session>:reports"
                             -> estimated stored bytes

Eviction runs as one Lua script, so a session is either removed completely or
not at all, and two app instances cannot evict the same session twice. The
script builds key names from the prefix, which requires a single Redis node
(not Redis Cluster).
"""

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol, cast
from uuid import UUID

from redis.exceptions import RedisError

from agentic_eval_toyapp.diagnostics import log_storage_event

LOGGER = logging.getLogger("uvicorn.error")

# Per-key allowance for Redis's own bookkeeping, added to every size estimate.
KEY_OVERHEAD_BYTES = 256
# Fixed allowance for one report's content-key framing (field names, version).
REPORT_FRAMING_BYTES = 512
# Space reserved for an assistant reply when a turn is admitted before the
# agent runs; the exact reply size is not known until afterwards.
TURN_REPLY_ALLOWANCE_BYTES = 64 * 1024


class StorageCapacityError(RuntimeError):
    """Raised when idle sessions cannot free enough room for a write."""


class StorageUnavailableError(RuntimeError):
    """Raised when Redis memory usage cannot be read."""


class StorageBudgetUnavailableError(StorageUnavailableError):
    """Raised when neither `maxmemory` nor the fallback budget is available."""


class RedisMemoryClient(Protocol):
    """Redis commands the guard needs from a store's client."""

    async def info(self, section: str) -> Mapping[str, object]:
        """Return one parsed `INFO` section."""
        ...

    async def eval(self, script: str, numkeys: int, *keys_and_args: str) -> object:
        """Run a Lua script atomically."""
        ...


class BookkeepingPipeline(Protocol):
    """Pipeline commands used to record access time and size."""

    def zadd(self, name: str, mapping: Mapping[str, float]) -> object:
        """Queue a sorted-set score update."""
        ...

    def hset(self, name: str, key: str, value: str) -> object:
        """Queue one hash field write."""
        ...

    def hdel(self, name: str, *keys: str) -> object:
        """Queue removal of hash fields."""
        ...

    async def execute(self) -> list[object]:
        """Apply every queued command atomically, or raise and apply none."""
        ...


@dataclass(frozen=True, slots=True)
class MemoryUsage:
    """Current Redis memory use against the derived budget."""

    used_bytes: int
    budget_bytes: int
    source: str


@dataclass(frozen=True, slots=True)
class EvictionRecord:
    """One session removed by an eviction pass."""

    session_id: str
    reason: str
    bytes_freed: int
    reports_removed: int


def estimate_report_bytes(
    *, byte_size: int, extracted_text_length: int, has_pages: bool
) -> int:
    """Estimate one report's stored size from its public metadata.

    Base64 grows the original by a third, and PDF page text repeats the
    extracted text. Estimates only steer how much an eviction pass frees;
    `INFO used_memory` decides whether a write fits.
    """
    base64_bytes = 4 * ((byte_size + 2) // 3)
    text_bytes = extracted_text_length * (2 if has_pages else 1)
    return base64_bytes + text_bytes + REPORT_FRAMING_BYTES + KEY_OVERHEAD_BYTES


def compute_budget(
    *, maxmemory: int | None, high_water: float, fallback_bytes: int | None
) -> tuple[int, str]:
    """Return the budget and its source; the smaller limit wins when both exist."""
    candidates: list[tuple[int, str]] = []
    if maxmemory is not None and maxmemory > 0:
        candidates.append((max(1, int(maxmemory * high_water)), "maxmemory"))
    if fallback_bytes is not None:
        candidates.append((fallback_bytes, "fallback"))
    if not candidates:
        raise StorageBudgetUnavailableError(
            "Redis maxmemory is 0 or unreadable and no fallback budget is set"
        )
    return min(candidates)


def _info_int(info: Mapping[str, object], field: str) -> int | None:
    value = info.get(field)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


# KEYS: lru, bytes.  ARGV: prefix, needed bytes, scan limit, exempt session ID.
# Returns a flat list: session ID, reason, bytes freed, reports removed, ...
_EVICT_SCRIPT = """
local lru = KEYS[1]
local sizes = KEYS[2]
local prefix = ARGV[1]
local needed = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local exempt = ARGV[4]
local freed = 0
local out = {}
local candidates = redis.call('ZRANGE', lru, 0, limit - 1)
for _, sid in ipairs(candidates) do
  if freed >= needed then
    break
  end
  local session_key = prefix .. ':session:' .. sid
  local index_key = prefix .. ':reports:' .. sid .. ':index'
  local locked = redis.call(
    'EXISTS',
    session_key .. ':turn-lock',
    prefix .. ':reports:' .. sid .. ':mutation-lock'
  ) > 0
  if sid ~= exempt and not locked then
    local bytes = (tonumber(redis.call('HGET', sizes, sid .. ':session')) or 0)
      + (tonumber(redis.call('HGET', sizes, sid .. ':reports')) or 0)
    local alive = redis.call('EXISTS', session_key) == 1
    local removed = 0
    local raw = redis.call('GET', index_key)
    if raw then
      local ok, index = pcall(cjson.decode, raw)
      if ok and type(index) == 'table' and type(index.reports) == 'table' then
        for _, entry in ipairs(index.reports) do
          if type(entry) == 'table' and type(entry.file_id) == 'string' then
            removed = removed
              + redis.call('DEL', prefix .. ':report:' .. sid .. ':' .. entry.file_id)
          end
        end
      end
    end
    redis.call('DEL', session_key, index_key)
    redis.call('ZREM', lru, sid)
    redis.call('HDEL', sizes, sid .. ':session', sid .. ':reports')
    local reason = 'expired'
    if alive then
      reason = 'memory_budget'
      freed = freed + bytes
    else
      -- Redis already reclaimed an expired session; only leftovers were removed.
      bytes = 0
    end
    table.insert(out, sid)
    table.insert(out, reason)
    table.insert(out, tostring(bytes))
    table.insert(out, tostring(removed))
  end
end
return out
"""


def _parse_records(raw: object) -> tuple[EvictionRecord, ...]:
    if not isinstance(raw, list):
        return ()
    items = [
        item.decode() if isinstance(item, bytes) else str(item)
        for item in cast(list[object], raw)
    ]
    records: list[EvictionRecord] = []
    for offset in range(0, len(items) - len(items) % 4, 4):
        session_id, reason, freed, removed = items[offset : offset + 4]
        records.append(
            EvictionRecord(
                session_id=session_id,
                reason=reason,
                bytes_freed=int(freed) if freed.isdigit() else 0,
                reports_removed=int(removed) if removed.isdigit() else 0,
            )
        )
    return tuple(records)


class RedisMemoryGuard:
    """Derive the memory budget and evict the longest-idle sessions to meet it.

    The guard holds no connection and no per-session state; each store passes
    its own client, so it is safe to share between stores and instances.
    """

    def __init__(
        self,
        *,
        key_prefix: str = "agentic-eval-toyapp",
        high_water: float = 0.75,
        budget_bytes: int | None = None,
        scan_limit: int = 64,
        max_rounds: int = 3,
        clock: Callable[[], float] = time.time,
    ) -> None:
        normalised_key_prefix = key_prefix.strip().rstrip(":")
        if not normalised_key_prefix:
            raise ValueError("key_prefix must not be blank")
        if not 0 < high_water <= 1:
            raise ValueError("high_water must be greater than 0 and at most 1")
        if budget_bytes is not None and budget_bytes < 1:
            raise ValueError("budget_bytes must be at least 1")
        if scan_limit < 1:
            raise ValueError("scan_limit must be at least 1")
        if max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")
        self._key_prefix = normalised_key_prefix
        self._high_water = high_water
        self._budget_bytes = budget_bytes
        self._scan_limit = scan_limit
        self._max_rounds = max_rounds
        self._clock = clock

    @property
    def lru_key(self) -> str:
        """Return the sorted set of tracked sessions by last access."""
        return f"{self._key_prefix}:sessions:lru"

    @property
    def bytes_key(self) -> str:
        """Return the hash of estimated stored bytes per session."""
        return f"{self._key_prefix}:sessions:bytes"

    def track(
        self,
        pipeline: BookkeepingPipeline,
        session_id: UUID,
        *,
        session_bytes: int | None = None,
        reports_bytes: int | None = None,
    ) -> None:
        """Queue a last-access update and any size fields on `pipeline`."""
        member = str(session_id)
        pipeline.zadd(self.lru_key, {member: self._clock()})
        if session_bytes is not None:
            pipeline.hset(self.bytes_key, f"{member}:session", str(session_bytes))
        if reports_bytes is not None:
            pipeline.hset(self.bytes_key, f"{member}:reports", str(reports_bytes))

    def forget_reports(self, pipeline: BookkeepingPipeline, session_id: UUID) -> None:
        """Queue removal of a session's report size after its last report."""
        pipeline.hdel(self.bytes_key, f"{session_id}:reports")

    async def usage(self, client: RedisMemoryClient) -> MemoryUsage:
        """Read `used_memory` and `maxmemory` and derive the budget."""
        try:
            info = await client.info("memory")
        except RedisError as error:
            raise StorageUnavailableError("Redis memory could not be read") from error
        used = _info_int(info, "used_memory")
        if used is None:
            raise StorageUnavailableError("Redis did not report used_memory")
        budget, source = compute_budget(
            maxmemory=_info_int(info, "maxmemory"),
            high_water=self._high_water,
            fallback_bytes=self._budget_bytes,
        )
        return MemoryUsage(used_bytes=used, budget_bytes=budget, source=source)

    async def ensure_room(
        self,
        client: RedisMemoryClient,
        *,
        incoming_bytes: int,
        current_session: UUID | None = None,
    ) -> None:
        """Evict idle sessions until `incoming_bytes` fits, or raise capacity."""
        usage = await self.usage(client)
        if incoming_bytes > usage.budget_bytes:
            self._refuse(usage, incoming_bytes, reason="larger_than_budget")
        for _ in range(self._max_rounds):
            over = usage.used_bytes + incoming_bytes - usage.budget_bytes
            if over <= 0:
                return
            records = await self._evict(client, over, current_session)
            if not records:
                self._refuse(usage, incoming_bytes, reason="nothing_evictable")
            usage = await self.usage(client)
        if usage.used_bytes + incoming_bytes > usage.budget_bytes:
            self._refuse(usage, incoming_bytes, reason="rounds_exhausted")

    async def _evict(
        self,
        client: RedisMemoryClient,
        needed_bytes: int,
        current_session: UUID | None,
    ) -> tuple[EvictionRecord, ...]:
        try:
            raw = await client.eval(
                _EVICT_SCRIPT,
                2,
                self.lru_key,
                self.bytes_key,
                self._key_prefix,
                str(needed_bytes),
                str(self._scan_limit),
                str(current_session) if current_session is not None else "",
            )
        except RedisError as error:
            raise StorageUnavailableError("Redis eviction failed") from error
        records = _parse_records(raw)
        for record in records:
            log_storage_event(
                event="evict" if record.reason == "memory_budget" else "expire",
                session_id=record.session_id,
                reason=record.reason,
                bytes_freed=record.bytes_freed,
                reports_removed=record.reports_removed,
            )
        return records

    @staticmethod
    def _refuse(usage: MemoryUsage, incoming_bytes: int, *, reason: str) -> None:
        LOGGER.warning(
            "toyapp_storage event=capacity_refused reason=%s used_bytes=%s "
            "budget_bytes=%s incoming_bytes=%s",
            reason,
            usage.used_bytes,
            usage.budget_bytes,
            incoming_bytes,
        )
        raise StorageCapacityError("Redis memory budget cannot fit the write")
