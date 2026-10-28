"""Integration tests for the local browser client and static assets."""

import hashlib
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import cast
from uuid import uuid4

import pytest
from httpx2 import Client, Response
from starlette.testclient import TestClient

from agentic_eval_toyapp.app import create_app
from agentic_eval_toyapp.config import AppSettings
from agentic_eval_toyapp.models import AgentTurnResult, ConversationMessage
from agentic_eval_toyapp.reports import StoredReport


class _FrontendAgent:
    async def run(
        self,
        user_message: str,
        *,
        history: Sequence[ConversationMessage] = (),
        session_id: str | None = None,
        reports: Sequence[StoredReport] = (),
    ) -> AgentTurnResult:
        del session_id, reports
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
    assert "Ask about policies or a report" in response.text
    assert 'id="files-panel"' in response.text
    assert response.text.index('id="files-panel"') < response.text.index(
        '<form id="chat-form"'
    )
    assert 'id="report-upload"' in response.text
    assert 'id="upload-report-button"' in response.text
    assert 'id="uploaded-files"' in response.text
    assert 'id="upload-status"' in response.text
    assert 'id="execution-steps"' in response.text
    assert 'id="report-references"' in response.text
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
    assert (
        response.text.index("/static/vendor/markdown-it.umd.min.js")
        < response.text.index("/static/vendor/purify.min.js")
        < response.text.index("/static/app.js")
    )
    assert '<div class="details-scroll">' in response.text


def test_uuid_session_path_loads_the_same_browser_shell() -> None:
    """A shareable session URL should bootstrap the existing browser client."""
    client = _build_client()
    root_response: Response = client.get("/")
    session_response: Response = client.get(f"/session/{uuid4()}")

    assert session_response.status_code == 200
    assert session_response.text == root_response.text
    assert (
        session_response.headers["content-security-policy"]
        == root_response.headers["content-security-policy"]
    )
    assert session_response.headers["x-content-type-options"] == "nosniff"


@pytest.mark.parametrize(
    "path",
    [
        "/not-a-session-id",
        "/session/not-a-session-id",
        "/00000000-0000-4000-8000-000000000000",
    ],
)
def test_unknown_browser_path_does_not_load_the_shell(path: str) -> None:
    """The named session route must not mask unknown application paths."""
    response: Response = _build_client().get(path)

    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")


def test_compact_file_panel_and_composer_upload_controls() -> None:
    """Upload stays separate from sending, with discoverable file instructions."""
    html: str = _build_client().get("/").text

    file_panel = html.split('<section id="files-panel"', 1)[1].split("</section>", 1)[0]
    composer = html.split('<form id="chat-form"', 1)[1].split("</form>", 1)[0]
    assert "Session file(s)" in file_panel
    assert 'id="file-help-tooltip"' in file_panel
    assert "Select a file and start messaging to analyse it." in file_panel
    assert "Deselect a file to exclude it from your messages." in file_panel
    assert "Only one file can be analysed at a time." in file_panel
    tooltip_section = file_panel.split('id="file-help-tooltip"', 1)[1]
    help_tooltip = tooltip_section.split("</div>", 1)[0]
    assert help_tooltip.count("<li>") == 3
    assert "<br>" not in help_tooltip
    assert 'id="uploaded-files"' in file_panel
    assert 'id="report-upload"' not in file_panel
    assert 'aria-describedby="file-help-tooltip"' in file_panel
    assert composer.index('id="upload-report-button"') < composer.index(
        'id="send-message"'
    )
    assert 'id="upload-control" class="tooltip-anchor" hidden' in composer
    assert (
        'id="report-upload" type="file" accept=".txt,.md,.pdf" hidden disabled'
        in composer
    )
    assert 'aria-describedby="upload-tooltip"' in composer
    assert "Choose a .txt/.md/.pdf file to analyse" in composer
    assert 'id="message-input"' in composer
    assert 'rows="2"' in composer


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
    assert response.headers["cache-control"] == "no-store"
    assert "https://" not in response.text


def test_browser_assets_use_one_content_derived_cache_version() -> None:
    """The HTML shell must not reuse stale application JavaScript or CSS."""
    html: str = _build_client().get("/").text

    stylesheet_version = html.split("/static/style.css?v=", 1)[1].split('"', 1)[0]
    javascript_version = html.split("/static/app.js?v=", 1)[1].split('"', 1)[0]

    assert stylesheet_version == javascript_version
    assert len(javascript_version) == 12
    assert all(character in "0123456789abcdef" for character in javascript_version)


def test_javascript_uses_the_versioned_api_and_safe_dom_rendering() -> None:
    """The browser must use the shared API and avoid HTML injection sinks."""
    response: Response = _build_client().get("/static/app.js")

    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]
    assert 'apiRequest("/sessions"' in response.text
    assert "class ApiRequestError extends Error" in response.text
    assert "this.status = status" in response.text
    assert "function sessionIdFromPath()" in response.text
    assert "window.location.pathname.match(" in response.text
    assert "const path = `/session/${encodeURIComponent(id)}`" in response.text
    assert "function updateSessionUrl(id, replace = false)" in response.text
    assert 'const method = replace ? "replaceState" : "pushState"' in response.text
    assert "window.history[method]({ sessionId: id }" in response.text
    assert "apiRequest(`/sessions/${encodeURIComponent(resumeId)}`)" in response.text
    assert "error.status !== 404" in response.text
    assert "is no longer available. Started a new session instead." in response.text
    assert (
        'createSession(sessionIdFromPath(), window.location.pathname === "/")'
        in response.text
    )
    assert 'window.addEventListener("popstate", () => window.location.reload())' in (
        response.text
    )
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
    assert "new FormData()" in response.text
    assert 'body.append("file", file)' in response.text
    assert (
        "body: JSON.stringify({ message, file_ids: selectedFileId ? [selectedFileId] : [] })"
        in response.text
    )
    assert (
        "selectedFileId = selectedFileId === file.file_id ? null : file.file_id"
        in response.text
    )
    assert "selected for messages." not in response.text
    assert "deleted from this session." not in response.text
    assert 'method: "DELETE"' in response.text
    assert "apiRequest(`/sessions/${sessionId}/files`)" in response.text
    assert (
        'uploadButton.addEventListener("click", () => reportUpload.click())'
        in response.text
    )
    assert "fileOperationInProgress || filesPanel.hidden" in response.text
    assert "uploadControl.hidden = filesPanel.hidden" in response.text
    assert (
        'appendMessage(message.role, message.content, "", "", false)' in response.text
    )
    assert "function resetSessionScroll()" in response.text
    assert "transcript.scrollTop = 0" in response.text
    assert "detailsScroll.scrollTop = 0" in response.text
    assert "uploadedFiles.scrollTop = 0" in response.text
    assert 'window.scrollTo({ top: 0, left: 0, behavior: "instant" })' in response.text
    assert 'window.history.scrollRestoration = "manual"' in response.text
    assert 'window.addEventListener("pageshow", resetSessionScroll)' in response.text
    assert 'sessionFiles.push(metadata);\n    setUploadStatus("");' in response.text
    assert "uploaded. Select it to use it in a message." not in response.text
    assert (
        'setStatus("Session ready. Send a message when you are ready.")'
        not in response.text
    )
    assert "result.execution_steps || []" in response.text
    assert "result.report_references || []" in response.text
    assert "window.markdownit({ html: false" in response.text
    assert "window.DOMPurify.sanitize" in response.text
    assert "RETURN_DOM_FRAGMENT: true" in response.text
    assert "ALLOWED_TAGS: markdownTags" in response.text
    assert "container.append(fragment)" in response.text
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
    assert ".files-panel[hidden]" in response.text
    assert ".uploaded-files" in response.text
    assert "max-height: 7rem" in response.text
    assert ".tooltip-anchor:focus-within .tooltip-bubble" in response.text
    assert ".tooltip-anchor[hidden]" in response.text
    assert ".upload-button" in response.text
    assert ".uploaded-file.is-selected" in response.text
    assert ".message-user .message-body p" in response.text
    assert "white-space: pre-wrap" in response.text
    assert "height: calc(100dvh - 2.5rem)" in response.text
    assert ".details-scroll" in response.text
    assert "scrollbar-width: thin" in response.text
    assert ".message-body blockquote" in response.text
    assert ".message-body pre" in response.text
    assert "http://" not in response.text
    assert "https://" not in response.text


@pytest.mark.parametrize(
    ("filename", "sha256"),
    [
        (
            "markdown-it.umd.min.js",
            "391f2aa638e75e6c7859d0a5c5650ef0956722a45998f17e05b83bc655778e1e",
        ),
        (
            "purify.min.js",
            "cd37898f9782adf54820c413191b5691803f0f137c799d6edcfd38523b4c2c31",
        ),
    ],
)
def test_pinned_markdown_assets_are_served_locally(filename: str, sha256: str) -> None:
    """Offline browser bundles should be byte-identical to reviewed releases."""
    response: Response = _build_client().get(f"/static/vendor/{filename}")

    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]
    assert hashlib.sha256(response.content).hexdigest() == sha256


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is not installed")
def test_pinned_markdown_renderer_formats_text_without_raw_html() -> None:
    """The shipped parser must format emphasis while escaping untrusted HTML."""
    bundle = (
        Path(__file__).parents[2]
        / "src/agentic_eval_toyapp/static/vendor/markdown-it.umd.min.js"
    )
    script = """
const MarkdownIt = require(process.argv[1]);
const markdown = MarkdownIt({ html: false, linkify: true, breaks: false });
process.stdout.write(markdown.render('**Bold**\\n\\n- item\\n\\n<script>alert(1)</script>\\n\\n[bad](javascript:alert(1))'));
"""

    output = subprocess.run(
        ["node", "-e", script, str(bundle)],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    ).stdout

    assert "<strong>Bold</strong>" in output
    assert "<li>item</li>" in output
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in output
    assert "<script>" not in output
    assert 'href="javascript:' not in output
