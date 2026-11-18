"""Track the estimated size and last access of every process-local session.

One `SessionMemoryLedger` is shared by `InMemorySessionStore` and
`InMemoryReportStore`, so a session's turns and reports are counted together.
The estimate is deliberately simple: `len()` of each stored string or byte
string plus a fixed allowance per session. It ignores Python object overhead
and undercounts non-ASCII text, so it approximates stored data, not RSS.

This module must not import `sessions` or `reports`; both stores import it.
"""

import time
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from threading import Lock
from typing import Final
from uuid import UUID

# Fixed cost of one session, so a flood of empty sessions still uses budget.
SESSION_ALLOWANCE_BYTES: Final = 1024


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


class SessionMemoryLedger:
    """Keep per-session size estimates and an idle order, updated incrementally.

    Sessions are held longest-idle first. Only sessions registered with
    `track` are counted; updates for any other ID, such as a seeded demo
    session, are ignored, so untracked sessions can never be evicted.
    """

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._entries: OrderedDict[UUID, _Entry] = OrderedDict()
        self._total_bytes = 0
        self._lock = Lock()

    @property
    def total_bytes(self) -> int:
        """Return the estimated bytes of every tracked session."""
        with self._lock:
            return self._total_bytes

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
