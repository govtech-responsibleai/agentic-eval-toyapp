"""HTTP-level proof that the Redis stores run on the loop and own their client."""

import asyncio
from collections.abc import Sequence
from typing import Any, cast

import pytest
from httpx2 import ASGITransport, AsyncClient
from starlette.testclient import TestClient

from agentic_eval_toyapp.app import create_app
from agentic_eval_toyapp.config import AppSettings
from agentic_eval_toyapp.models import (
    AgentTurnResult,
    AgentTurnStatus,
    ClarifyAction,
    ConversationMessage,
    TerminationReason,
)
from agentic_eval_toyapp.reports import RedisReportStore, StoredReport
from agentic_eval_toyapp.sessions import RedisSessionStore


class _Agent:
    async def run(
        self,
        user_message: str,
        *,
        history: Sequence[ConversationMessage] = (),
        session_id: str | None = None,
        reports: Sequence[StoredReport] = (),
    ) -> AgentTurnResult:
        del history, session_id, reports
        question = f"Clarify: {user_message}"
        return AgentTurnResult(
            status=AgentTurnStatus.CLARIFICATION_REQUIRED,
            message=question,
            termination_reason=TerminationReason.CLARIFICATION_REQUIRED,
            model_calls=1,
            searches=0,
            model_actions=(ClarifyAction(question=question),),
        )


class _Lock:
    def __init__(self, guard: asyncio.Lock) -> None:
        self._guard = guard

    async def acquire(self) -> bool:
        await self._guard.acquire()
        return True

    async def release(self) -> None:
        self._guard.release()


class _FakeRedis:
    """Async stand-in for `redis.asyncio.Redis`, bound to whichever loop calls it."""

    def __init__(self) -> None:
        self.data: dict[str, str] = {}
        self.closed = False
        self._guards: dict[str, asyncio.Lock] = {}

    async def ping(self) -> bool:
        return True

    async def set(self, name: str, value: str, *, nx: bool = False) -> object:
        await asyncio.sleep(0)
        if nx and name in self.data:
            return False
        self.data[name] = value
        return True

    async def get(self, name: str) -> str | None:
        await asyncio.sleep(0)
        return self.data.get(name)

    def lock(self, name: str, *, timeout: float, blocking_timeout: float) -> _Lock:
        del timeout, blocking_timeout
        return _Lock(self._guards.setdefault(name, asyncio.Lock()))

    async def aclose(self) -> None:
        self.closed = True


def _app(
    clients: list[_FakeRedis],
) -> tuple[Any, RedisSessionStore, RedisReportStore]:
    def factory() -> _FakeRedis:
        clients.append(_FakeRedis())
        return clients[-1]

    sessions = RedisSessionStore(client_factory=factory)
    reports = RedisReportStore(client_factory=factory, max_reports_per_session=3)
    application = create_app(
        AppSettings(langfuse_enabled=False),
        session_store=sessions,
        report_store=reports,
        agent=_Agent(),
    )
    return application, sessions, reports


def test_creating_the_app_opens_no_client() -> None:
    """The loop-bound pool must not exist until the lifespan runs."""
    clients: list[_FakeRedis] = []
    _app(clients)
    assert clients == []


def test_ready_before_the_lifespan_reports_the_stores_unavailable() -> None:
    """Without a lifespan there is no client, so readiness must say not ready."""
    clients: list[_FakeRedis] = []
    application, _, _ = _app(clients)

    body = TestClient(application).get("/ready").json()

    assert body["status"] == "not_ready"
    assert [c["status"] for c in body["checks"] if c["name"].endswith("_store")] == [
        "fail",
        "fail",
    ]
    assert "Redis client" not in str(body)


def test_lifespan_opens_serves_and_closes_clients_and_can_run_twice() -> None:
    """TestClient runs the lifespan: stores open, serve, close, then reopen."""
    clients: list[_FakeRedis] = []
    application, _, _ = _app(clients)

    for run in (1, 2):
        with TestClient(application) as client:
            ready = client.get("/ready").json()
            assert ready["status"] == "ready"
            assert len(clients) == 2 * run
            created = client.post("/api/v1/sessions").json()["session_id"]
            reply = client.post(
                f"/api/v1/sessions/{created}/messages", json={"message": "hi"}
            )
            assert reply.status_code == 200
        assert all(c.closed for c in clients)


def test_a_failed_startup_still_closes_the_opened_clients() -> None:
    """If agent start-up raises, the pools opened before it must be closed."""
    clients: list[_FakeRedis] = []

    def factory() -> _FakeRedis:
        clients.append(_FakeRedis())
        return clients[-1]

    def broken_agent() -> Any:
        raise ValueError("boom")

    application = create_app(
        AppSettings(langfuse_enabled=False),
        session_store=RedisSessionStore(client_factory=factory),
        report_store=RedisReportStore(
            client_factory=factory, max_reports_per_session=1
        ),
        agent_factory=broken_agent,
    )

    with (
        pytest.raises(Exception, match="could not be initialised"),
        TestClient(application),
    ):
        pass

    assert len(clients) == 2
    assert all(c.closed for c in clients)


@pytest.mark.anyio
async def test_injected_fakes_serve_requests_under_async_client() -> None:
    """Injected clients work without a lifespan, and same-session turns queue."""
    client_fake = _FakeRedis()
    application = create_app(
        AppSettings(langfuse_enabled=False),
        session_store=RedisSessionStore(client_fake),
        report_store=RedisReportStore(client_fake, max_reports_per_session=3),
        agent=_Agent(),
    )
    async with AsyncClient(
        transport=ASGITransport(app=application), base_url="http://testserver"
    ) as client:
        session_id = (await client.post("/api/v1/sessions")).json()["session_id"]
        uploaded = await client.post(
            f"/api/v1/sessions/{session_id}/files",
            files={"file": ("r.md", b"# Report\n\nBody.", "text/markdown")},
        )
        assert uploaded.status_code == 201

        first, second = await asyncio.gather(
            client.post(
                f"/api/v1/sessions/{session_id}/messages", json={"message": "one"}
            ),
            client.post(
                f"/api/v1/sessions/{session_id}/messages", json={"message": "two"}
            ),
        )
        assert first.status_code == second.status_code == 200

        snapshot = (await client.get(f"/api/v1/sessions/{session_id}")).json()
        contents = [cast(str, m["content"]) for m in snapshot["messages"]]
        assert sorted(contents[::2]) == ["one", "two"]
        assert len(contents) == 4
        listed = (await client.get(f"/api/v1/sessions/{session_id}/files")).json()
        assert len(listed["files"]) == 1
        deleted = await client.delete(
            f"/api/v1/sessions/{session_id}/files/{listed['files'][0]['file_id']}"
        )
        assert deleted.status_code == 204
        assert (await client.get("/ready")).json()["status"] == "ready"
