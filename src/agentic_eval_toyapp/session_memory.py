"""Keep process-local sessions under a memory budget by evicting idle ones.

One `SessionMemoryLedger` is shared by `InMemorySessionStore` and
`InMemoryReportStore`, so a session's turns and reports are counted together.
The estimate is deliberately simple: `len()` of each stored string or byte
string plus a fixed allowance per session. It ignores Python object overhead
and undercounts non-ASCII text, so it approximates stored data, not RSS.

Before a write, `ensure_room` first removes sessions idle for longer than the
idle TTL (lazily: there is no background task), then plans which
longest-idle sessions to evict and evicts them only when the plan frees
enough room; otherwise it evicts nothing
and raises `StorageCapacityError`. Each store registers a callback that
removes one session's data, so evicting a session removes its reports too.
Everything runs on one event loop with no `await` between planning and
removal, so no other request can interleave.

This module must not import `sessions` or `reports`; both stores import it.
"""

import logging
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from threading import Lock
from typing import Final
from uuid import UUID

from agentic_eval_toyapp.diagnostics import log_process_memory, log_storage_event
from agentic_eval_toyapp.redis_memory import StorageCapacityError

LOGGER = logging.getLogger("uvicorn.error")

# Fixed cost of one session, so a flood of empty sessions still uses budget.
SESSION_ALLOWANCE_BYTES: Final = 1024


class SessionNotFoundError(LookupError):
    """Raised when a caller references an unknown application session."""


def turn_footprint(user_text: str, assistant_text: str) -> int:
    """Estimate one stored turn as the length of both messages."""
    return len(user_text) + len(assistant_text)


def report_footprint(
    original_bytes: bytes, extracted_text: str, page_texts: Iterable[str]
) -> int:
    """Estimate one stored report: original bytes, extracted text and pages.

    PDF page text is counted on top of the extracted text because the store
    keeps both; a text report has no pages and pays for its text once.
    """
    return (
        len(original_bytes)
        + len(extracted_text)
        + sum(len(page_text) for page_text in page_texts)
    )


@dataclass(frozen=True, slots=True)
class SessionFootprint:
    """Snapshot of one tracked session's estimated size and last access."""

    session_bytes: int
    report_bytes: int
    last_access: float

    @property
    def total_bytes(self) -> int:
        """Return the session's turns, allowance and reports together."""
        return self.session_bytes + self.report_bytes


@dataclass(slots=True)
class _Entry:
    session_bytes: int
    report_bytes: int
    last_access: float


def _never_busy(session_id: UUID) -> bool:
    del session_id
    return False


def _remove_nothing(session_id: UUID) -> int:
    del session_id
    return 0


class SessionMemoryLedger:
    """Keep per-session size estimates and an idle order, updated incrementally.

    Sessions are held longest-idle first. Only sessions registered with
    `track` are counted; updates for any other ID, such as a seeded demo
    session, are ignored, so untracked sessions can never be evicted.

    `budget_bytes=None` disables the budget: sizes are still tracked, but
    `ensure_room` never evicts for size or refuses. `idle_ttl_seconds=None`
    disables idle expiry. The two are independent.
    """

    def __init__(
        self,
        *,
        budget_bytes: int | None = None,
        idle_ttl_seconds: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if budget_bytes is not None and budget_bytes < 1:
            raise ValueError("budget_bytes must be at least 1")
        if idle_ttl_seconds is not None and idle_ttl_seconds <= 0:
            raise ValueError("idle_ttl_seconds must be positive")
        self._budget_bytes = budget_bytes
        self._idle_ttl_seconds = idle_ttl_seconds
        self._clock = clock
        self._entries: OrderedDict[UUID, _Entry] = OrderedDict()
        self._total_bytes = 0
        self._reserved_bytes = 0
        self._lock = Lock()
        self._remove_session: Callable[[UUID], object] = _remove_nothing
        self._is_busy: Callable[[UUID], bool] = _never_busy
        self._remove_reports: Callable[[UUID], int] = _remove_nothing

    @property
    def budget_bytes(self) -> int | None:
        """Return the configured budget, or None when it is disabled."""
        return self._budget_bytes

    @property
    def total_bytes(self) -> int:
        """Return the estimated bytes of every tracked session."""
        with self._lock:
            return self._total_bytes

    @property
    def reserved_bytes(self) -> int:
        """Return room held for replies of turns that are still running."""
        with self._lock:
            return self._reserved_bytes

    def bind_session_store(
        self,
        remove: Callable[[UUID], object],
        is_busy: Callable[[UUID], bool],
    ) -> None:
        """Register how to drop a session's turns and detect a running turn."""
        self._remove_session = remove
        self._is_busy = is_busy

    def bind_report_store(self, remove: Callable[[UUID], int]) -> None:
        """Register how to drop a session's reports; it returns the count."""
        self._remove_reports = remove

    def is_tracked(self, session_id: UUID) -> bool:
        """Return whether the session is counted (and so still exists)."""
        with self._lock:
            return session_id in self._entries

    def reserve(self, size_bytes: int) -> None:
        """Hold room for a write whose final size is not yet known."""
        with self._lock:
            self._reserved_bytes += size_bytes

    def release(self, size_bytes: int) -> None:
        """Give back room taken by `reserve`."""
        with self._lock:
            if size_bytes > self._reserved_bytes:
                raise ValueError("released more than was reserved")
            self._reserved_bytes -= size_bytes

    def ensure_room(
        self, incoming_bytes: int, *, current_session: UUID | None = None
    ) -> None:
        """Expire idle sessions, then evict so `incoming_bytes` fits, or raise.

        Never removes `current_session`, a session whose turn is running, or
        an untracked (demo) session. When the eligible sessions cannot free
        enough room, nothing is evicted and `StorageCapacityError` is raised.
        """
        self._expire_idle(current_session)
        budget = self._budget_bytes
        if budget is None:
            return
        with self._lock:
            used = self._total_bytes + self._reserved_bytes
            candidates = tuple(
                (session_id, entry.session_bytes + entry.report_bytes)
                for session_id, entry in self._entries.items()
            )
        if incoming_bytes > budget:
            self._refuse(used, budget, incoming_bytes, reason="larger_than_budget")
        needed = used + incoming_bytes - budget
        if needed <= 0:
            return
        victims: list[tuple[UUID, int]] = []
        freed = 0
        for session_id, size_bytes in candidates:
            if session_id == current_session or self._is_busy(session_id):
                continue
            victims.append((session_id, size_bytes))
            freed += size_bytes
            if freed >= needed:
                break
        if freed < needed:
            self._refuse(used, budget, incoming_bytes, reason="nothing_evictable")
        for session_id, size_bytes in victims:
            self._remove(session_id, size_bytes, event="evict", reason="budget")
        log_process_memory(event="evict", ledger_bytes=self.total_bytes)

    def _expire_idle(self, current_session: UUID | None) -> None:
        """Remove sessions idle for at least the TTL, longest-idle first.

        Sessions are held in last-access order, so the walk stops at the first
        one still within the TTL; with nothing expired it costs one check.
        """
        ttl = self._idle_ttl_seconds
        if ttl is None:
            return
        now = self._clock()
        with self._lock:
            expired: list[tuple[UUID, int]] = []
            for session_id, entry in self._entries.items():
                if now - entry.last_access < ttl:
                    break
                expired.append((session_id, entry.session_bytes + entry.report_bytes))
        removed = 0
        for session_id, size_bytes in expired:
            if session_id == current_session or self._is_busy(session_id):
                continue
            self._remove(session_id, size_bytes, event="expire", reason="idle")
            removed += 1
        if removed:
            log_process_memory(event="expire", ledger_bytes=self.total_bytes)

    def _remove(
        self, session_id: UUID, size_bytes: int, *, event: str, reason: str
    ) -> None:
        with self._lock:
            entry = self._entries.pop(session_id)
            self._total_bytes -= entry.session_bytes + entry.report_bytes
        # Called without the ledger lock, because each store takes its own.
        self._remove_session(session_id)
        reports_removed = self._remove_reports(session_id)
        log_storage_event(
            event=event,
            session_id=str(session_id),
            reason=reason,
            bytes_freed=size_bytes,
            reports_removed=reports_removed,
        )

    @staticmethod
    def _refuse(used: int, budget: int, incoming_bytes: int, *, reason: str) -> None:
        LOGGER.warning(
            "toyapp_storage event=capacity_refused reason=%s used_bytes=%s "
            "budget_bytes=%s incoming_bytes=%s",
            reason,
            used,
            budget,
            incoming_bytes,
        )
        raise StorageCapacityError("session memory budget cannot fit the write")

    def track(self, session_id: UUID) -> None:
        """Start counting a new session at the fixed allowance."""
        with self._lock:
            if session_id in self._entries:
                raise ValueError("session is already tracked")
            self._entries[session_id] = _Entry(
                session_bytes=SESSION_ALLOWANCE_BYTES,
                report_bytes=0,
                last_access=self._clock(),
            )
            self._total_bytes += SESSION_ALLOWANCE_BYTES

    def touch(self, session_id: UUID) -> None:
        """Record an access, making the session the most recently used."""
        with self._lock:
            entry = self._entries.get(session_id)
            if entry is None:
                return
            entry.last_access = self._clock()
            self._entries.move_to_end(session_id)

    def adjust(
        self, session_id: UUID, *, session_delta: int = 0, report_delta: int = 0
    ) -> None:
        """Add or subtract stored bytes; a negative result means drift."""
        with self._lock:
            entry = self._entries.get(session_id)
            if entry is None:
                return
            session_bytes = entry.session_bytes + session_delta
            report_bytes = entry.report_bytes + report_delta
            if session_bytes < SESSION_ALLOWANCE_BYTES or report_bytes < 0:
                raise ValueError("session size estimate would become negative")
            entry.session_bytes = session_bytes
            entry.report_bytes = report_bytes
            self._total_bytes += session_delta + report_delta

    def footprint(self, session_id: UUID) -> SessionFootprint | None:
        """Return one session's snapshot, or None when it is not tracked."""
        with self._lock:
            entry = self._entries.get(session_id)
            if entry is None:
                return None
            return SessionFootprint(
                session_bytes=entry.session_bytes,
                report_bytes=entry.report_bytes,
                last_access=entry.last_access,
            )

    def idle_order(self) -> tuple[UUID, ...]:
        """Return tracked session IDs, longest-idle first."""
        with self._lock:
            return tuple(self._entries)
