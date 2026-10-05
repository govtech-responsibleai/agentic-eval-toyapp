"""Integration tests for the local browser client and static assets."""

from collections.abc import Sequence
from typing import cast

from httpx2 import Client, Response
from starlette.testclient import TestClient

from agentic_eval_toyapp.app import create_app
from agentic_eval_toyapp.config import AppSettings
from agentic_eval_toyapp.models import AgentTurnResult, ConversationMessage


class _FrontendAgent:
    def run(
        self,
        user_message: str,
        *,
        history: Sequence[ConversationMessage] = (),
        session_id: str | None = None,
    ) -> AgentTurnResult:
        del session_id
        raise AssertionError("frontend asset tests must not execute an agent turn")


def _build_client(*, app_name: str = "Frontend Test Target") -> Client:
    application = create_app(
        AppSettings(app_name=app_name),
        agent=_FrontendAgent(),
    )
    return cast(Client, TestClient(application))


def test_root_page_loads_with_accessible_controls() -> None:
    """The browser shell should expose the intended accessible controls."""
    response: Response = _build_client().get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Frontend Test Target" in response.text
    assert 'id="chat-form"' in response.text
    assert 'id="new-session"' in response.text
    assert 'for="message-input"' in response.text
    assert 'role="log"' in response.text
    assert 'role="status"' in response.text
    assert 'id="execution-details"' in response.text
    assert (
        '<details id="execution-details" class="execution-details" open>'
        in response.text
    )
    assert "Execution details" in response.text
    assert "Ask about Northstar policies" in response.text
    assert "Outcome reason" in response.text
    assert "Model calls" in response.text
    assert "Final model action" in response.text
    assert 'id="controller-events"' in response.text
    assert 'id="validation-findings"' in response.text
    assert 'id="model-actions"' in response.text
    assert "Provider calls" not in response.text
    assert "Enter sends." not in response.text
    assert 'data-api-base="/api/v1"' in response.text
    assert "/static/app.js" in response.text
    assert "/static/style.css" in response.text


def test_root_page_escapes_configuration_and_sets_browser_guards() -> None:
    """Configuration text must not become executable markup."""
    response: Response = _build_client(
        app_name="Target <script>alert('unsafe')</script>"
    ).get("/")

    assert "<script>alert" not in response.text
    assert "&lt;script&gt;" in response.text
    assert "default-src 'self'" in response.headers["content-security-policy"]
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "https://" not in response.text


def test_javascript_uses_the_versioned_api_and_safe_dom_rendering() -> None:
    """The browser must use the shared API and avoid HTML injection sinks."""
    response: Response = _build_client().get("/static/app.js")

    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]
    assert 'apiRequest("/sessions"' in response.text
    assert "/sessions/${sessionId}/messages" in response.text
    assert "textContent" in response.text
    assert "replaceChildren" in response.text
    assert "renderCitations" in response.text
    assert 'setAttribute("aria-controls"' in response.text
    assert 'window.matchMedia("(prefers-reduced-motion: reduce)")' in response.text
    assert "HTMLDetailsElement" in response.text
    assert "executionDetails.open = true" in response.text
    assert "!event.shiftKey && !event.isComposing" in response.text
    assert 'return "Needs clarification"' in response.text
    assert 'return "Answered"' in response.text
    assert "validation_findings" in response.text
    assert "controller_events" in response.text
    assert "model_actions" in response.text
    assert "formatModelAction" in response.text
    assert '"No retrieval was performed."' in response.text
    assert "catch (error)" in response.text
    assert "workingIndicator.remove()" in response.text
    assert "setBusy(false)" in response.text
    assert "messageInput.focus()" in response.text
    assert "Answered without evidence" not in response.text
    assert "event.ctrlKey" not in response.text
    assert "event.metaKey" not in response.text
    assert "innerHTML" not in response.text
    assert "eval(" not in response.text


def test_stylesheet_is_local_and_contains_responsive_layout() -> None:
    """The demo UI should not depend on a CDN or fixed desktop viewport."""
    response: Response = _build_client().get("/static/style.css")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/css")
    assert "@media (max-width: 900px)" in response.text
    assert "@media (max-width: 560px)" in response.text
    assert "@media (prefers-reduced-motion: reduce)" in response.text
    assert "@media (prefers-reduced-transparency: reduce)" in response.text
    assert "@media (prefers-contrast: more)" in response.text
    assert "backdrop-filter" in response.text
    assert ".message-user .message-body p" in response.text
    assert "white-space: pre-wrap" in response.text
    assert "http://" not in response.text
    assert "https://" not in response.text
