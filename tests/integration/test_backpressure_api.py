"""HTTP-level proof of turn admission and the shared completion client wiring."""

import asyncio
import json
from collections.abc import MutableMapping, Sequence
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from httpx2 import ASGITransport, AsyncClient

from agentic_eval_toyapp.app import (
    _build_coordinator_agent,  # pyright: ignore[reportPrivateUsage]
    create_app,
)
from agentic_eval_toyapp.backpressure import RateLimitedCompletionClient
from agentic_eval_toyapp.config import AppSettings, GenerationProviderKind
from agentic_eval_toyapp.models import (
    AgentTurnResult,
    AgentTurnStatus,
    ClarifyAction,
    ConversationMessage,
    TerminationReason,
)
from agentic_eval_toyapp.reports import StoredReport
from agentic_eval_toyapp.sessions import InMemorySessionStore

_TIMEOUT_SECONDS = 5.0
_BASE_URL = "http://testserver"


class _GatedAgent:
    """Counts calls and holds every turn open until the test opens the gate."""

    def __init__(self) -> None:
        self.calls = 0
        self.gate = asyncio.Event()
        self.fail_with: Exception | None = None
        self._entered = asyncio.Condition()

    async def wait_for_calls(self, count: int) -> None:
        async with self._entered:
            await asyncio.wait_for(
                self._entered.wait_for(lambda: self.calls >= count), _TIMEOUT_SECONDS
            )

    async def run(
        self,
        user_message: str,
        *,
        history: Sequence[ConversationMessage] = (),
        session_id: str | None = None,
        reports: Sequence[StoredReport] = (),
    ) -> AgentTurnResult:
        del history, session_id, reports
        async with self._entered:
            self.calls += 1
            self._entered.notify_all()
        await self.gate.wait()
        if self.fail_with is not None:
            raise self.fail_with
        question = f"Clarify: {user_message}"
        return AgentTurnResult(
            status=AgentTurnStatus.CLARIFICATION_REQUIRED,
            message=question,
            termination_reason=TerminationReason.CLARIFICATION_REQUIRED,
            model_calls=1,
            searches=0,
            model_actions=(ClarifyAction(question=question),),
        )


def _application(agent: _GatedAgent, *, cap: int) -> Any:
    return create_app(
        AppSettings(max_pending_turns=cap, langfuse_enabled=False),
        session_store=InMemorySessionStore(),
        agent=agent,
    )


def _client(application: Any) -> AsyncClient:
    return AsyncClient(
        transport=ASGITransport(app=application, raise_app_exceptions=False),
        base_url=_BASE_URL,
    )


async def _new_session(client: AsyncClient) -> str:
    return (await client.post("/api/v1/sessions")).json()["session_id"]


def _post_message(client: AsyncClient, session_id: str, text: str = "hello") -> Any:
    return asyncio.ensure_future(
        client.post(f"/api/v1/sessions/{session_id}/messages", json={"message": text})
    )


async def _stored_turns(client: AsyncClient, session_id: str) -> int:
    return (await client.get(f"/api/v1/sessions/{session_id}")).json()["retained_turns"]


@pytest.mark.anyio
async def test_exactly_at_the_cap_every_turn_is_admitted() -> None:
    agent = _GatedAgent()
    async with _client(_application(agent, cap=2)) as client:
        sessions = [await _new_session(client) for _ in range(2)]
        turns = [_post_message(client, session) for session in sessions]
        await agent.wait_for_calls(2)

        agent.gate.set()
        responses = await asyncio.wait_for(asyncio.gather(*turns), _TIMEOUT_SECONDS)

    assert [response.status_code for response in responses] == [200, 200]
    assert agent.calls == 2


@pytest.mark.anyio
async def test_one_over_the_cap_is_refused_before_any_agent_code_runs() -> None:
    agent = _GatedAgent()
    async with _client(_application(agent, cap=2)) as client:
        held_sessions = [await _new_session(client) for _ in range(2)]
        rejected_session = await _new_session(client)
        held = [_post_message(client, session) for session in held_sessions]
        await agent.wait_for_calls(2)

        rejected = await asyncio.wait_for(
            _post_message(client, rejected_session), _TIMEOUT_SECONDS
        )

        assert rejected.status_code == 503
        assert rejected.json()["code"] == "server_busy"
        assert rejected.headers["Retry-After"] == "10"
        assert agent.calls == 2  # the provider-side call count did not move
        assert await _stored_turns(client, rejected_session) == 0

        agent.gate.set()
        await asyncio.wait_for(asyncio.gather(*held), _TIMEOUT_SECONDS)


@pytest.mark.anyio
async def test_a_rejected_request_leaves_other_sessions_unaffected() -> None:
    agent = _GatedAgent()
    async with _client(_application(agent, cap=2)) as client:
        held_sessions = [await _new_session(client) for _ in range(2)]
        bystander = await _new_session(client)
        held = [_post_message(client, session) for session in held_sessions]
        await agent.wait_for_calls(2)
        for _ in range(3):
            refused = await _post_message(client, bystander)
            assert refused.status_code == 503

        agent.gate.set()
        results = await asyncio.wait_for(asyncio.gather(*held), _TIMEOUT_SECONDS)

        assert [result.status_code for result in results] == [200, 200]
        assert [await _stored_turns(client, s) for s in held_sessions] == [1, 1]
        assert await _stored_turns(client, bystander) == 0
        # The bystander is not poisoned: once capacity frees it is served.
        retry = await asyncio.wait_for(
            _post_message(client, bystander), _TIMEOUT_SECONDS
        )
        assert retry.status_code == 200
        assert await _stored_turns(client, bystander) == 1


@pytest.mark.anyio
async def test_404_and_422_are_decided_before_admission_and_consume_no_slot() -> None:
    agent = _GatedAgent()
    async with _client(_application(agent, cap=1)) as client:
        held_session = await _new_session(client)
        held = _post_message(client, held_session)
        await agent.wait_for_calls(1)

        unknown = await client.post(
            "/api/v1/sessions/00000000-0000-0000-0000-000000000000/messages",
            json={"message": "hi"},
        )
        invalid = await client.post(
            f"/api/v1/sessions/{held_session}/messages", json={"message": ""}
        )

        # The cap is full, yet these answer with their own status, not 503.
        assert (unknown.status_code, unknown.json()["code"]) == (
            404,
            "session_not_found",
        )
        assert (invalid.status_code, invalid.json()["code"]) == (422, "invalid_request")

        agent.gate.set()
        assert (await asyncio.wait_for(held, _TIMEOUT_SECONDS)).status_code == 200


@pytest.mark.anyio
async def test_server_busy_is_distinguishable_from_agent_unavailable() -> None:
    agent = _GatedAgent()
    unavailable = create_app(
        AppSettings(langfuse_enabled=False),
        session_store=InMemorySessionStore(),
        agent_factory=_failing_factory,
    )
    async with _client(unavailable) as client:
        session = await _new_session(client)
        broken = await client.post(
            f"/api/v1/sessions/{session}/messages", json={"message": "hi"}
        )
    async with _client(_application(agent, cap=1)) as client:
        first = await _new_session(client)
        second = await _new_session(client)
        held = _post_message(client, first)
        await agent.wait_for_calls(1)
        busy = await _post_message(client, second)
        agent.gate.set()
        await asyncio.wait_for(held, _TIMEOUT_SECONDS)

    assert broken.status_code == busy.status_code == 503
    assert broken.json()["code"] == "agent_unavailable"
    assert busy.json()["code"] == "server_busy"
    assert "Retry-After" not in broken.headers


def _failing_factory() -> Any:
    raise ValueError("not configured")


@pytest.mark.anyio
async def test_the_slot_is_released_after_success() -> None:
    agent = _GatedAgent()
    agent.gate.set()
    async with _client(_application(agent, cap=1)) as client:
        session = await _new_session(client)
        for _ in range(3):
            response = await _post_message(client, session)
            assert response.status_code == 200


@pytest.mark.anyio
async def test_the_slot_is_released_after_an_agent_exception() -> None:
    agent = _GatedAgent()
    agent.gate.set()
    agent.fail_with = RuntimeError("agent exploded")
    async with _client(_application(agent, cap=1)) as client:
        session = await _new_session(client)
        failed = await _post_message(client, session)
        assert failed.status_code == 500

        agent.fail_with = None
        recovered = await _post_message(client, session)

    assert recovered.status_code == 200


@pytest.mark.anyio
async def test_the_slot_is_released_after_a_cancellation() -> None:
    agent = _GatedAgent()
    async with _client(_application(agent, cap=1)) as client:
        first = await _new_session(client)
        second = await _new_session(client)
        held = _post_message(client, first)
        await agent.wait_for_calls(1)
        assert (await _post_message(client, second)).status_code == 503

        held.cancel()
        with pytest.raises(asyncio.CancelledError):
            await held

        agent.gate.set()
        after = await asyncio.wait_for(_post_message(client, second), _TIMEOUT_SECONDS)

    assert after.status_code == 200


async def _raw_post(
    application: Any,
    session_id: str,
    *,
    disconnect: asyncio.Event,
) -> list[MutableMapping[str, Any]]:
    """Drive the ASGI app directly so the client can vanish mid-request."""
    body = json.dumps({"message": "hello"}).encode()
    scope: dict[str, Any] = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": f"/api/v1/sessions/{session_id}/messages",
        "raw_path": f"/api/v1/sessions/{session_id}/messages".encode(),
        "query_string": b"",
        "headers": [
            (b"host", b"testserver"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ],
        "client": ("127.0.0.1", 1),
        "server": ("testserver", 80),
    }
    sent_body = False
    sent: list[MutableMapping[str, Any]] = []

    async def receive() -> MutableMapping[str, Any]:
        nonlocal sent_body
        if not sent_body:
            sent_body = True
            return {"type": "http.request", "body": body, "more_body": False}
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message: MutableMapping[str, Any]) -> None:
        sent.append(message)

    await application(scope, receive, send)
    return sent


@pytest.mark.anyio
async def test_a_client_disconnect_keeps_the_slot_until_the_turn_finishes() -> None:
    """Pins the behaviour the issue believed but had not verified."""
    agent = _GatedAgent()
    application = _application(agent, cap=1)
    async with _client(application) as client:
        first = await _new_session(client)
        second = await _new_session(client)
        disconnect = asyncio.Event()
        raw = asyncio.ensure_future(
            _raw_post(application, first, disconnect=disconnect)
        )
        await agent.wait_for_calls(1)

        disconnect.set()  # the client goes away while the turn is running
        for _ in range(20):
            await asyncio.sleep(0)

        # The handler was not cancelled, and its slot is still held.
        assert not raw.done()
        assert (await _post_message(client, second)).status_code == 503

        agent.gate.set()
        sent = await asyncio.wait_for(raw, _TIMEOUT_SECONDS)

        assert sent[0]["status"] == 200  # the turn ran to completion
        assert await _stored_turns(client, first) == 1  # and its result was stored
        after = await asyncio.wait_for(_post_message(client, second), _TIMEOUT_SECONDS)
        assert after.status_code == 200  # the slot is free again


def test_create_app_does_not_need_the_lifespan_to_build_the_admission_cap() -> None:
    application = create_app(
        AppSettings(max_pending_turns=3, langfuse_enabled=False),
        session_store=InMemorySessionStore(),
        agent=_GatedAgent(),
    )

    assert application is not None


def test_one_shared_completion_client_serves_every_live_agent() -> None:
    settings = AppSettings(
        generation_provider=GenerationProviderKind.OPENAI,
        openai_api_key="test-key",  # pyright: ignore[reportArgumentType]
        openai_base_url="https://example.invalid/v1",
        langfuse_enabled=False,
    )
    rag_builder = MagicMock(return_value=MagicMock(model_id="rag"))
    coordinator_builder = MagicMock()
    analysis_builder = MagicMock()
    with (
        patch("agentic_eval_toyapp.app.OpenAICompatibleChatClient") as raw_client,
        patch("agentic_eval_toyapp.app._build_rag_agent", rag_builder),
        patch(
            "agentic_eval_toyapp.app.build_openai_coordinator_provider",
            coordinator_builder,
        ),
        patch(
            "agentic_eval_toyapp.app.OpenAIAnalysisGenerationProvider",
            analysis_builder,
        ),
        patch("agentic_eval_toyapp.app.RagSpecialistAdapter"),
        patch("agentic_eval_toyapp.app.ReportAnalysisAgent"),
        patch("agentic_eval_toyapp.app.CoordinatorAgent"),
    ):
        _build_coordinator_agent(settings, MagicMock())

    shared = coordinator_builder.call_args.kwargs["client"]
    assert isinstance(shared, RateLimitedCompletionClient)
    assert rag_builder.call_args.kwargs["completion_client"] is shared
    assert analysis_builder.call_args.kwargs["client"] is shared
    raw_client.assert_called_once()
