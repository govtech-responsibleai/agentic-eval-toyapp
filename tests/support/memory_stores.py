"""Shared in-memory store fixtures for eviction and expiry tests."""

import hashlib
from collections.abc import Sequence
from uuid import UUID, uuid4

import anyio
from tests.support.fake_clock import FakeClock

from agentic_eval_toyapp.models import (
    AgentTurnResult,
    AgentTurnStatus,
    ClarifyAction,
    ConversationMessage,
    ConversationRole,
    ReportMediaType,
    TerminationReason,
)
from agentic_eval_toyapp.reports import (
    InMemoryReportStore,
    ProcessedReport,
    StoredReport,
)
from agentic_eval_toyapp.session_memory import SessionMemoryLedger
from agentic_eval_toyapp.sessions import InMemorySessionStore

KIB = 1024


class ShortAgent:
    """Reply with a fixed short message and count calls."""

    def __init__(self) -> None:
        self.calls = 0

    async def run(
        self,
        user_message: str,
        *,
        history: Sequence[ConversationMessage] = (),
        session_id: str | None = None,
        reports: Sequence[StoredReport] = (),
    ) -> AgentTurnResult:
        del user_message, history, session_id, reports
        self.calls += 1
        return _result("ok")


class BlockingAgent:
    """Hold a turn open until the test releases it."""

    def __init__(self) -> None:
        self.started = anyio.Event()
        self.release = anyio.Event()

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
        await self.release.wait()
        return _result("ok")


class FailingAgent:
    """Raise from every turn."""

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


def _result(message: str) -> AgentTurnResult:
    return AgentTurnResult(
        status=AgentTurnStatus.CLARIFICATION_REQUIRED,
        message=message,
        termination_reason=TerminationReason.CLARIFICATION_REQUIRED,
        model_calls=1,
        searches=0,
        model_actions=(ClarifyAction(question=message),),
    )


def text_report(size: int) -> ProcessedReport:
    """A text report whose footprint is exactly `2 * size` bytes."""
    content = b"x" * size
    return ProcessedReport(
        filename="report.txt",
        media_type=ReportMediaType.TEXT,
        byte_size=size,
        original_bytes=content,
        extracted_text="x" * size,
        sha256=hashlib.sha256(content).hexdigest(),
    )


class MemoryStores:
    """Both in-memory stores sharing one ledger and a fake clock."""

    def __init__(
        self, budget_bytes: int | None, *, idle_ttl_seconds: float | None = None
    ) -> None:
        self.clock = FakeClock()
        self.ledger = SessionMemoryLedger(
            budget_bytes=budget_bytes,
            idle_ttl_seconds=idle_ttl_seconds,
            clock=self.clock,
        )
        self.sessions = InMemorySessionStore(ledger=self.ledger)
        self.reports = InMemoryReportStore(
            max_reports_per_session=10, ledger=self.ledger
        )

    async def create(self) -> UUID:
        """Create a session, then advance the clock by one second."""
        session = await self.sessions.create()
        self.clock.advance(1)
        return session.session_id

    def seed_demo(self) -> UUID:
        """Seed an untracked read-only session, like the bundled demo."""
        demo_id = uuid4()
        user = ConversationMessage(role=ConversationRole.USER, content="Q")
        assistant = ConversationMessage(role=ConversationRole.ASSISTANT, content="A")
        self.sessions.seed(demo_id, ((user, assistant),), read_only=True)
        return demo_id
