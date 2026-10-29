"use strict";

const app = document.querySelector("#app");
const apiBase = app.dataset.apiBase;
const transcript = document.querySelector("#transcript");
const detailsScroll = document.querySelector(".details-scroll");
const form = document.querySelector("#chat-form");
const messageInput = document.querySelector("#message-input");
const sendButton = document.querySelector("#send-message");
const newSessionButton = document.querySelector("#new-session");
const demoButton = document.querySelector("#demo-session");
const demoSessionId = app.dataset.demoSessionId || null;
const exportButton = document.querySelector("#export-session");
const sessionIdElement = document.querySelector("#session-id");
const copySessionIdButton = document.querySelector("#copy-session-id");
const appStatus = document.querySelector("#app-status");
const detailsEmpty = document.querySelector("#details-empty");
const turnDetails = document.querySelector("#turn-details");
const turnStatus = document.querySelector("#turn-status");
const executionDetails = document.querySelector("#execution-details");
const terminationReason = document.querySelector("#termination-reason");
const modelCalls = document.querySelector("#model-calls");
const searches = document.querySelector("#searches");
const modelAction = document.querySelector("#model-action");
const citations = document.querySelector("#citations");
const sources = document.querySelector("#sources");
const actions = document.querySelector("#actions");
const modelActions = document.querySelector("#model-actions");
const controllerEvents = document.querySelector("#controller-events");
const validationFindings = document.querySelector("#validation-findings");
const executionSteps = document.querySelector("#execution-steps");
const reportReferences = document.querySelector("#report-references");
const filesPanel = document.querySelector("#files-panel");
const uploadControl = document.querySelector("#upload-control");
const uploadButton = document.querySelector("#upload-report-button");
const reportUpload = document.querySelector("#report-upload");
const uploadedFiles = document.querySelector("#uploaded-files");
const uploadStatus = document.querySelector("#upload-status");
const markdownRenderer = typeof window.markdownit === "function"
  ? window.markdownit({ html: false, linkify: true, breaks: false })
  : null;

const markdownTags = [
  "a", "blockquote", "br", "code", "del", "em", "h1", "h2", "h3", "h4", "h5", "h6",
  "hr", "li", "ol", "p", "pre", "strong", "table", "tbody", "td", "th", "thead", "tr", "ul",
];

let sessionId = null;
let requestInProgress = false;
let fileOperationInProgress = false;
let selectedFileId = null;
let sessionFiles = [];

class ApiRequestError extends Error {
  constructor(message, status) {
    super(message);
    this.name = "ApiRequestError";
    this.status = status;
  }
}

function preferredScrollBehavior() {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches
    ? "auto"
    : "smooth";
}

function resetSessionScroll() {
  transcript.scrollTop = 0;
  detailsScroll.scrollTop = 0;
  uploadedFiles.scrollTop = 0;
  window.scrollTo({ top: 0, left: 0, behavior: "instant" });
}

function setStatus(message, isError = false) {
  appStatus.textContent = message;
  appStatus.classList.toggle("app-status-error", isError);
}

function setBusy(isBusy) {
  requestInProgress = isBusy;
  updateControls();
  sendButton.textContent = isBusy ? "Working…" : "Send message";
}

function updateControls() {
  const unavailable = sessionId === null || requestInProgress;
  messageInput.disabled = unavailable;
  sendButton.disabled = unavailable || fileOperationInProgress || !messageInput.value.trim();
  newSessionButton.disabled = requestInProgress || fileOperationInProgress;
  demoButton.disabled = !demoSessionId || requestInProgress || fileOperationInProgress;
  exportButton.disabled = sessionId === null || requestInProgress || fileOperationInProgress;
  const uploadUnavailable = unavailable || fileOperationInProgress || filesPanel.hidden;
  uploadControl.hidden = filesPanel.hidden;
  uploadButton.disabled = uploadUnavailable;
  reportUpload.disabled = uploadUnavailable;
  for (const button of uploadedFiles.querySelectorAll("button")) {
    button.disabled = unavailable || fileOperationInProgress;
  }
  copySessionIdButton.disabled = sessionId === null;
  app.setAttribute("aria-busy", String(requestInProgress || fileOperationInProgress));
}

async function apiRequest(path, options = {}) {
  const response = await fetch(`${apiBase}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
  });
  let payload = {};
  try {
    payload = await response.json();
  } catch (_error) {
    throw new Error(`The server returned an unreadable response (${response.status}).`);
  }
  if (!response.ok) {
    throw new ApiRequestError(
      payload.message || `Request failed (${response.status}).`,
      response.status,
    );
  }
  return payload;
}

async function fileRequest(path, options) {
  const response = await fetch(`${apiBase}${path}`, options);
  if (response.ok) {
    return response.status === 204 ? null : response.json();
  }
  let payload = {};
  try {
    payload = await response.json();
  } catch (_error) {
    // Keep the HTTP status when an upstream response is not JSON.
  }
  throw new Error(payload.message || `Request failed (${response.status}).`);
}

function createTextElement(tagName, text, className = "") {
  const element = document.createElement(tagName);
  element.textContent = text;
  if (className) {
    element.className = className;
  }
  return element;
}

function setUploadStatus(message, isError = false) {
  uploadStatus.textContent = message;
  uploadStatus.classList.toggle("upload-status-error", isError);
}

function renderFiles() {
  uploadedFiles.replaceChildren();
  if (sessionFiles.length === 0) {
    uploadedFiles.append(createTextElement("li", "No reports uploaded yet.", "files-empty"));
  }
  for (const file of sessionFiles) {
    const item = document.createElement("li");
    item.className = `uploaded-file${selectedFileId === file.file_id ? " is-selected" : ""}`;
    const description = document.createElement("div");
    description.className = "uploaded-file-description";
    description.append(createTextElement("strong", file.filename));
    description.append(createTextElement("span", `${Math.ceil(file.byte_size / 1024)} KB · ${file.media_type}`));
    const controls = document.createElement("div");
    controls.className = "uploaded-file-controls";
    const select = createTextElement(
      "button",
      selectedFileId === file.file_id ? "Deselect" : "Select",
      "file-action",
    );
    select.type = "button";
    select.setAttribute("aria-label", `${select.textContent} ${file.filename} for messages`);
    select.setAttribute("aria-pressed", String(selectedFileId === file.file_id));
    select.addEventListener("click", () => {
      selectedFileId = selectedFileId === file.file_id ? null : file.file_id;
      setUploadStatus("");
      renderFiles();
    });
    const remove = createTextElement("button", "Delete", "file-action file-delete");
    remove.type = "button";
    remove.setAttribute("aria-label", `Permanently delete ${file.filename} from this session`);
    remove.addEventListener("click", () => deleteFile(file));
    controls.append(select, remove);
    item.append(description, controls);
    uploadedFiles.append(item);
  }
  updateControls();
}

async function loadFiles() {
  const response = await apiRequest(`/sessions/${sessionId}/files`);
  sessionFiles = response.files;
  if (selectedFileId && !sessionFiles.some((file) => file.file_id === selectedFileId)) {
    selectedFileId = null;
  }
  renderFiles();
}

async function uploadFile() {
  const file = reportUpload.files?.[0];
  if (!file || sessionId === null || fileOperationInProgress) {
    return;
  }
  fileOperationInProgress = true;
  updateControls();
  setUploadStatus(`Uploading ${file.name}…`);
  const body = new FormData();
  body.append("file", file);
  try {
    const metadata = await fileRequest(`/sessions/${sessionId}/files`, {
      method: "POST",
      body,
    });
    sessionFiles.push(metadata);
    setUploadStatus("");
    renderFiles();
  } catch (error) {
    setUploadStatus(error.message, true);
  } finally {
    reportUpload.value = "";
    fileOperationInProgress = false;
    updateControls();
  }
}

async function deleteFile(file) {
  if (sessionId === null || fileOperationInProgress || requestInProgress) {
    return;
  }
  if (!window.confirm(`Delete ${file.filename} from this session? Past messages will remain.`)) {
    return;
  }
  fileOperationInProgress = true;
  updateControls();
  try {
    await fileRequest(`/sessions/${sessionId}/files/${encodeURIComponent(file.file_id)}`, {
      method: "DELETE",
    });
    sessionFiles = sessionFiles.filter((item) => item.file_id !== file.file_id);
    if (selectedFileId === file.file_id) {
      selectedFileId = null;
    }
    setUploadStatus("");
    renderFiles();
  } catch (error) {
    await loadFiles().catch(() => {});
    setUploadStatus(error.message, true);
  } finally {
    fileOperationInProgress = false;
    updateControls();
  }
}

function renderWelcome() {
  const item = document.createElement("li");
  item.className = "conversation-welcome";

  const mark = createTextElement("span", "N", "welcome-mark");
  mark.setAttribute("aria-hidden", "true");
  item.append(mark);

  const copy = document.createElement("div");
  copy.append(createTextElement("h3", "Ask about policies or a report"));
  copy.append(
    createTextElement(
      "p",
      "Ask a policy question, or upload a report and select it when you want the assistant to use it.",
    ),
  );
  item.append(copy);
  transcript.append(item);
}

function removeWelcome() {
  transcript.querySelector(".conversation-welcome")?.remove();
}

function scrollToElement(element) {
  element.scrollIntoView({ behavior: preferredScrollBehavior(), block: "nearest" });
}

function renderResponseText(container, text) {
  if (markdownRenderer === null || typeof window.DOMPurify?.sanitize !== "function") {
    container.classList.add("message-body-plain");
    container.append(createTextElement("p", text));
    return;
  }

  const fragment = window.DOMPurify.sanitize(markdownRenderer.render(text), {
    RETURN_DOM_FRAGMENT: true,
    ALLOWED_TAGS: markdownTags,
    ALLOWED_ATTR: ["href", "title"],
  });
  for (const link of fragment.querySelectorAll("a[href]")) {
    const href = link.getAttribute("href");
    try {
      const url = new URL(href, window.location.href);
      if (!["http:", "https:", "mailto:"].includes(url.protocol)) {
        link.removeAttribute("href");
        continue;
      }
      link.setAttribute("target", "_blank");
      link.setAttribute("rel", "noopener noreferrer");
    } catch (_error) {
      link.removeAttribute("href");
    }
  }
  container.append(fragment);
}

function appendMessage(role, text, status = "", filename = "", shouldScroll = true) {
  removeWelcome();
  const item = document.createElement("li");
  item.className = `message message-${role}`;

  const heading = createTextElement(
    "p",
    role === "user" ? "You" : "Toy app",
    "message-role",
  );
  item.append(heading);
  if (status) {
    item.append(createTextElement("span", status.replaceAll("_", " "), "message-status"));
  }

  const body = document.createElement("div");
  body.className = "message-body";
  if (role === "assistant") {
    renderResponseText(body, text);
  } else {
    body.append(createTextElement("p", text));
    if (filename) {
      body.append(createTextElement("span", `Attached: ${filename}`, "message-attachment"));
    }
  }
  item.append(body);
  transcript.append(item);
  if (shouldScroll) {
    scrollToElement(item);
  }
}

function appendWorkingIndicator() {
  const item = document.createElement("li");
  item.className = "message message-assistant message-working";
  item.setAttribute("aria-hidden", "true");

  const dots = document.createElement("div");
  dots.className = "working-dots";
  for (let index = 0; index < 3; index += 1) {
    dots.append(document.createElement("span"));
  }
  item.append(dots);
  transcript.append(item);
  scrollToElement(item);
  return item;
}

function renderList(container, values, emptyLabel) {
  container.replaceChildren();
  if (values.length === 0) {
    container.append(createTextElement("li", emptyLabel, "muted-item"));
    return;
  }
  for (const value of values) {
    container.append(createTextElement("li", value));
  }
}

function formatModelAction(action) {
  if (action.kind === "search") {
    return `search: ${action.query} (top_k=${action.top_k})`;
  }
  if (action.kind === "clarify") {
    return `clarify: ${action.question}`;
  }
  const citationCount = action.citations.length;
  return `answer: ${citationCount} ${citationCount === 1 ? "citation" : "citations"}`;
}

function resultStatusLabel(result) {
  if (result.status === "answered") {
    return "Answered";
  }
  if (result.status === "clarification_required") {
    return "Needs clarification";
  }
  return "Failed";
}

function resultStatusClass(result) {
  return `status-${result.status}`;
}

function renderSources(retrievedChunks, searchCount) {
  sources.replaceChildren();
  const sourceIds = new Map();
  if (retrievedChunks.length === 0) {
    const emptyMessage =
      searchCount === 0
        ? "No retrieval was performed."
        : "No source passed the retrieval threshold.";
    sources.append(createTextElement("p", emptyMessage, "muted-item"));
    return sourceIds;
  }

  for (const [index, chunk] of retrievedChunks.entries()) {
    const details = document.createElement("details");
    details.className = "source-item";
    details.id = `retrieved-source-${index + 1}`;
    sourceIds.set(chunk.chunk_id, details.id);
    const score = Number(chunk.similarity_score).toFixed(3);
    const summary = createTextElement(
      "summary",
      `${chunk.title} · ${chunk.chunk_id} · score ${score}`,
    );
    details.append(summary);
    details.append(createTextElement("p", chunk.text));
    details.append(
      createTextElement(
        "p",
        `Topic: ${chunk.metadata.topic} · Effective: ${chunk.metadata.effective_date}`,
        "source-metadata",
      ),
    );
    sources.append(details);
  }
  return sourceIds;
}

function renderCitations(values, sourceIds) {
  citations.replaceChildren();
  if (values.length === 0) {
    citations.append(createTextElement("li", "No citations returned.", "muted-item"));
    return;
  }

  for (const value of values) {
    const item = document.createElement("li");
    const sourceId = sourceIds.get(value);
    if (sourceId) {
      const button = createTextElement("button", value, "citation-chip");
      button.type = "button";
      button.setAttribute("aria-controls", sourceId);
      button.addEventListener("click", () => {
        const source = document.getElementById(sourceId);
        if (!(source instanceof HTMLDetailsElement)) {
          return;
        }
        for (const candidate of sources.querySelectorAll(".source-highlight")) {
          candidate.classList.remove("source-highlight");
        }
        source.open = true;
        source.classList.add("source-highlight");
        scrollToElement(source);
      });
      item.append(button);
    } else {
      item.append(createTextElement("span", value, "muted-item"));
    }
    citations.append(item);
  }
}

function renderTurnDetails(result) {
  detailsEmpty.hidden = true;
  turnDetails.hidden = false;
  turnStatus.textContent = resultStatusLabel(result);
  turnStatus.className = `status-badge ${resultStatusClass(result)}`;
  terminationReason.textContent = result.termination_reason.replaceAll("_", " ");
  modelCalls.textContent = String(result.model_calls);
  searches.textContent = String(result.searches);
  modelAction.textContent = result.model_actions.at(-1)?.kind ?? "None";

  const sourceIds = renderSources(result.retrieved_chunks, result.searches);
  renderCitations(result.citations, sourceIds);
  renderList(
    reportReferences,
    (result.report_references || []).map((reference) => {
      const filename = sessionFiles.find((file) => file.file_id === reference.file_id)?.filename
        ?? reference.file_id;
      const locator = reference.locator.kind === "page"
        ? `page ${reference.locator.page_number}`
        : `section ${reference.locator.section}`;
      return `${filename} · ${locator}: ${reference.excerpt}`;
    }),
    "No report references returned.",
  );
  renderList(
    actions,
    result.action_summaries.map(
      (summary) => `${summary.action.replaceAll("_", " ")}: ${summary.detail}`,
    ),
    "No public action summary returned.",
  );
  renderList(
    executionSteps,
    (result.execution_steps || []).map((step) => {
      const duration = step.duration_ms === null ? "" : ` · ${Math.round(step.duration_ms)} ms`;
      return `${step.agent} · ${step.action.replaceAll("_", " ")} · ${step.status}${duration}: ${step.summary}`;
    }),
    "No agent timeline returned.",
  );
  renderList(
    modelActions,
    result.model_actions.map(formatModelAction),
    "No valid structured model action was returned.",
  );
  renderList(
    controllerEvents,
    result.controller_events.map(
      (event) => `${event.kind.replaceAll("_", " ")}: ${event.detail}`,
    ),
    "No controller intervention was required.",
  );
  renderList(
    validationFindings,
    result.validation_findings.map((finding) => {
      const citationSuffix =
        finding.citation_ids.length === 0
          ? ""
          : ` (${finding.citation_ids.join(", ")})`;
      return `${finding.code.replaceAll("_", " ")}: ${finding.detail}${citationSuffix}`;
    }),
    "No deterministic citation inconsistency was detected.",
  );
  executionDetails.open = true;
}

function resetDetails() {
  detailsEmpty.hidden = false;
  turnDetails.hidden = true;
  turnStatus.textContent = "Waiting";
  turnStatus.className = "status-badge status-idle";
  executionDetails.open = true;
  terminationReason.textContent = "—";
  modelCalls.textContent = "—";
  searches.textContent = "—";
  modelAction.textContent = "—";
  citations.replaceChildren();
  sources.replaceChildren();
  reportReferences.replaceChildren();
  actions.replaceChildren();
  executionSteps.replaceChildren();
  modelActions.replaceChildren();
  controllerEvents.replaceChildren();
  validationFindings.replaceChildren();
}

function sessionIdFromPath() {
  const match = window.location.pathname.match(
    /^\/session\/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\/?$/i,
  );
  return match === null ? null : match[1].toLowerCase();
}

function updateSessionUrl(id, replace = false) {
  const path = `/session/${encodeURIComponent(id)}`;
  if (window.location.pathname === path) {
    return;
  }
  const method = replace ? "replaceState" : "pushState";
  window.history[method]({ sessionId: id }, "", path);
}

async function createSession(resumeId = null, replaceUrl = false) {
  setBusy(true);
  setStatus(resumeId === null ? "Creating a new session…" : "Loading session…");
  sessionId = null;
  selectedFileId = null;
  sessionFiles = [];
  filesPanel.hidden = true;
  renderFiles();
  setUploadStatus("");
  sessionIdElement.textContent = "Creating session…";
  transcript.replaceChildren();
  renderWelcome();
  resetDetails();
  resetSessionScroll();
  try {
    let session = null;
    let fallbackMessage = "";
    if (resumeId !== null) {
      try {
        session = await apiRequest(`/sessions/${encodeURIComponent(resumeId)}`);
      } catch (error) {
        if (!(error instanceof ApiRequestError) || error.status !== 404) {
          throw error;
        }
        fallbackMessage = `Session ${resumeId} is no longer available. Started a new session instead.`;
      }
    }
    if (session === null) {
      session = await apiRequest("/sessions", { method: "POST" });
    }
    sessionId = session.session_id;
    updateSessionUrl(sessionId, replaceUrl || fallbackMessage !== "");
    sessionIdElement.textContent = sessionId;
    for (const message of session.messages) {
      appendMessage(message.role, message.content, "", "", false);
    }
    try {
      const capabilities = await apiRequest("/capabilities");
      filesPanel.hidden = !capabilities.actions.includes("route_analysis");
      if (!filesPanel.hidden) {
        await loadFiles();
      }
    } catch (error) {
      filesPanel.hidden = true;
      setStatus(`Session ready, but file controls are unavailable: ${error.message}`, true);
    }
    if (!appStatus.classList.contains("app-status-error")) {
      setStatus(fallbackMessage, fallbackMessage !== "");
    }
  } catch (error) {
    setStatus(error.message, true);
    sessionIdElement.textContent = "Unavailable";
  } finally {
    setBusy(false);
    resetSessionScroll();
  }
}

async function submitMessage(event) {
  event.preventDefault();
  const message = messageInput.value.trim();
  if (!message || sessionId === null || requestInProgress || fileOperationInProgress) {
    return;
  }

  const attachedFile = sessionFiles.find((file) => file.file_id === selectedFileId);
  appendMessage("user", message, "", attachedFile?.filename ?? "");
  messageInput.value = "";
  setBusy(true);
  setStatus("Reviewing your message and preparing a response…");
  const workingIndicator = appendWorkingIndicator();
  try {
    const response = await apiRequest(`/sessions/${sessionId}/messages`, {
      method: "POST",
      body: JSON.stringify({ message, file_ids: selectedFileId ? [selectedFileId] : [] }),
    });
    workingIndicator.remove();
    appendMessage("assistant", response.result.message, resultStatusLabel(response.result));
    renderTurnDetails(response.result);
    setStatus("Turn complete.");
  } catch (error) {
    workingIndicator.remove();
    appendMessage("assistant", error.message, "request_error");
    setStatus(error.message, true);
    messageInput.value = message;
    if (selectedFileId) {
      await loadFiles().catch(() => {});
    }
  } finally {
    setBusy(false);
    messageInput.focus();
  }
}

async function copySessionId() {
  if (sessionId === null) {
    return;
  }
  try {
    await navigator.clipboard.writeText(sessionId);
    copySessionIdButton.textContent = "Copied";
    copySessionIdButton.classList.add("copy-button-copied");
  } catch (_error) {
    copySessionIdButton.textContent = "Failed";
    copySessionIdButton.classList.add("copy-button-failed");
  }
  setTimeout(() => {
    copySessionIdButton.textContent = "Copy";
    copySessionIdButton.classList.remove("copy-button-copied", "copy-button-failed");
  }, 2000);
}

function exportSession() {
  if (sessionId === null || requestInProgress || fileOperationInProgress) {
    return;
  }
  window.location.href = `${apiBase}/sessions/${encodeURIComponent(sessionId)}/export`;
}

form.addEventListener("submit", submitMessage);
newSessionButton.addEventListener("click", () => createSession());
copySessionIdButton.addEventListener("click", copySessionId);
demoButton.addEventListener("click", () => {
  if (demoSessionId) {
    createSession(demoSessionId);
  }
});
exportButton.addEventListener("click", exportSession);
uploadButton.addEventListener("click", () => reportUpload.click());
reportUpload.addEventListener("change", uploadFile);
messageInput.addEventListener("input", updateControls);
messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    form.requestSubmit();
  }
});

if ("scrollRestoration" in window.history) {
  window.history.scrollRestoration = "manual";
}
window.addEventListener("pageshow", resetSessionScroll);
window.addEventListener("popstate", () => window.location.reload());
createSession(sessionIdFromPath(), window.location.pathname === "/");
