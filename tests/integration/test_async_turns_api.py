"""HTTP-level proof that an in-flight turn never stalls the event loop."""

import asyncio
from collections.abc import Sequence

import pytest
from httpx2 import ASGITransport, AsyncClient

from agentic_eval_toyapp.app import create_app
from agentic_eval_toyapp.config import AppSettings
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


class _GatedAgent:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.gate = asyncio.Event()

    async def run(
        self,
        user_message: str,
        *,
        history: Sequence[ConversationMessage] = (),
        session_id: str | None = None,
        reports: Sequence[StoredReport] = (),
    ) -> AgentTurnResult:
        del history, session_id, reports
        self.entered.set()
        await self.gate.wait()
        question = f"Clarify: {user_message}"
        return AgentTurnResult(
            status=AgentTurnStatus.CLARIFICATION_REQUIRED,
            message=question,
            termination_reason=TerminationReason.CLARIFICATION_REQUIRED,
            model_calls=1,
            searches=0,
            model_actions=(ClarifyAction(question=question),),
        )


@pytest.mark.anyio
async def test_upload_and_reads_proceed_while_a_turn_is_in_flight() -> None:
    """Regression: the old threading lock made upload_report freeze the loop."""
    agent = _GatedAgent()
    application = create_app(
        AppSettings(),
        session_store=InMemorySessionStore(),
        agent=agent,
    )
    async with AsyncClient(
        transport=ASGITransport(app=application), base_url="http://testserver"
    ) as client:
        created = await client.post("/api/v1/sessions")
        session_id = created.json()["session_id"]
        other_id = (await client.post("/api/v1/sessions")).json()["session_id"]

        held_turn = asyncio.create_task(
            client.post(
                f"/api/v1/sessions/{session_id}/messages",
                json={"message": "held turn"},
            )
        )
        await asyncio.wait_for(agent.entered.wait(), _TIMEOUT_SECONDS)

        # Each request would hang forever if the loop were blocked on the turn.
        uploaded = await asyncio.wait_for(
            client.post(
                f"/api/v1/sessions/{session_id}/files",
                files={"file": ("report.md", b"# Report\n\nBody.", "text/markdown")},
            ),
            _TIMEOUT_SECONDS,
        )
        listed = await asyncio.wait_for(
            client.get(f"/api/v1/sessions/{session_id}/files"), _TIMEOUT_SECONDS
        )
        snapshot = await asyncio.wait_for(
            client.get(f"/api/v1/sessions/{session_id}"), _TIMEOUT_SECONDS
        )
        other = await asyncio.wait_for(
            client.get(f"/api/v1/sessions/{other_id}"), _TIMEOUT_SECONDS
        )

        assert uploaded.status_code == 201
        assert listed.status_code == 200
        assert len(listed.json()["files"]) == 1
        assert snapshot.json()["messages"] == []
        assert other.status_code == 200
        assert not held_turn.done()

        agent.gate.set()
        answered = await asyncio.wait_for(held_turn, _TIMEOUT_SECONDS)

    assert answered.status_code == 200
    assert answered.json()["result"]["message"] == "Clarify: held turn"
