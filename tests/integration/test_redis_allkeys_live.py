"""Opt-in check that Redis's own allkeys-lru eviction never causes a 500.

Run against a dedicated, disposable container configured so Redis itself
evicts keys before the app's budget is reached:

    docker run --rm -p 6391:6379 redis:8.2-alpine \\
        redis-server --maxmemory 8mb --maxmemory-policy allkeys-lru
    TOYAPP_TEST_REDIS_ALLKEYS_URL=redis://127.0.0.1:6391/0 make test-live

The app's own budget is set far above `maxmemory`, so every key Redis removes
is outside the app's control, which is what the missing-key rules must absorb.
"""

import os
from collections.abc import Sequence
from typing import Any
from uuid import uuid4

import pytest
from httpx2 import ASGITransport, AsyncClient
from redis.asyncio import Redis

from agentic_eval_toyapp.app import create_app
from agentic_eval_toyapp.config import AppSettings
from agentic_eval_toyapp.models import (
    AgentTurnResult,
    AgentTurnStatus,
    ClarifyAction,
    ConversationMessage,
    TerminationReason,
)
from agentic_eval_toyapp.redis_memory import RedisMemoryGuard
from agentic_eval_toyapp.reports import RedisReportStore, StoredReport
from agentic_eval_toyapp.sessions import RedisSessionStore

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not os.environ.get("TOYAPP_TEST_REDIS_ALLKEYS_URL"),
        reason="set TOYAPP_TEST_REDIS_ALLKEYS_URL to a disposable allkeys-lru Redis",
    ),
]


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
        return AgentTurnResult(
            status=AgentTurnStatus.CLARIFICATION_REQUIRED,
            message=f"Clarify: {user_message}",
            termination_reason=TerminationReason.CLARIFICATION_REQUIRED,
            model_calls=1,
            searches=0,
            model_actions=(ClarifyAction(question=f"Clarify: {user_message}"),),
        )


@pytest.mark.anyio
async def test_allkeys_lru_eviction_never_returns_a_server_error() -> None:
    redis: Any = Redis.from_url(  # pyright: ignore[reportUnknownMemberType]
        os.environ["TOYAPP_TEST_REDIS_ALLKEYS_URL"], decode_responses=True
    )
    info = await redis.info("memory")
    if info.get("maxmemory_policy") != "allkeys-lru":
        await redis.aclose()
        pytest.skip("start the container with --maxmemory-policy allkeys-lru")
    prefix = f"toyapp-allkeys-{uuid4().hex}"
    guard = RedisMemoryGuard(key_prefix=prefix, budget_bytes=1 << 40)
    application = create_app(
        AppSettings(langfuse_enabled=False),
        session_store=RedisSessionStore(redis, key_prefix=prefix, memory_guard=guard),
        report_store=RedisReportStore(
            redis,
            key_prefix=prefix,
            max_reports_per_session=5,
            memory_guard=guard,
        ),
        agent=_Agent(),
    )
    # Stay inside the default upload limits (100,000 extracted characters), or
    # every upload is refused with 413 and Redis never fills up. 80 sessions of
    # about 210 KB each is roughly twice the 8 MB maxmemory.
    body = b"# Report\n\n" + b"x" * 90_000
    sessions: list[tuple[str, str]] = []
    statuses: list[tuple[str, int]] = []

    def record(route: str, status: int) -> None:
        statuses.append((route, status))

    try:
        async with AsyncClient(
            transport=ASGITransport(app=application), base_url="http://testserver"
        ) as client:
            for _ in range(80):
                created = await client.post("/api/v1/sessions")
                record("create", created.status_code)
                if created.status_code != 201:
                    continue
                session_id = created.json()["session_id"]
                uploaded = await client.post(
                    f"/api/v1/sessions/{session_id}/files",
                    files={"file": ("r.md", body, "text/markdown")},
                )
                record("upload", uploaded.status_code)
                file_id = (
                    uploaded.json()["file_id"] if uploaded.status_code == 201 else ""
                )
                sessions.append((session_id, file_id))

            # Revisit every session after Redis has evicted keys on its own.
            for session_id, file_id in sessions:
                base = f"/api/v1/sessions/{session_id}"
                record("get", (await client.get(base)).status_code)
                record("list", (await client.get(f"{base}/files")).status_code)
                sent = await client.post(
                    f"{base}/messages",
                    json={"message": "hi", "file_ids": [file_id] if file_id else []},
                )
                record("message", sent.status_code)
                record("export", (await client.get(f"{base}/export")).status_code)
                if file_id:
                    deleted = await client.delete(f"{base}/files/{file_id}")
                    record("delete", deleted.status_code)
    finally:
        keys = [key async for key in redis.scan_iter(f"{prefix}:*")]
        if keys:
            await redis.delete(*keys)
        await redis.aclose()

    server_errors = [(r, s) for r, s in statuses if s >= 500 and s != 503]
    assert server_errors == []
    uploads = [status for route, status in statuses if route == "upload"]
    assert uploads.count(201) > len(uploads) // 2, f"uploads were refused: {uploads}"
    # The run is only meaningful if Redis actually removed some keys.
    assert any(status == 404 for _route, status in statuses), (
        "Redis evicted nothing; lower maxmemory"
    )
