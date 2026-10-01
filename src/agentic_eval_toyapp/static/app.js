"use strict";

const app = document.querySelector("#app");
const apiBase = app.dataset.apiBase;
const transcript = document.querySelector("#transcript");
const form = document.querySelector("#chat-form");
const messageInput = document.querySelector("#message-input");
const sendButton = document.querySelector("#send-message");
const newSessionButton = document.querySelector("#new-session");
const sessionIdElement = document.querySelector("#session-id");
const appStatus = document.querySelector("#app-status");
const detailsEmpty = document.querySelector("#details-empty");
const turnDetails = document.querySelector("#turn-details");
const turnStatus = document.querySelector("#turn-status");
const executionDetails = document.querySelector("#execution-details");
const terminationReason = document.querySelector("#termination-reason");
const modelCalls = document.querySelector("#model-calls");
const searches = document.querySelector("#searches");
const citations = document.querySelector("#citations");
const sources = document.querySelector("#sources");
const actions = document.querySelector("#actions");

let sessionId = null;
let requestInProgress = false;

function preferredScrollBehavior() {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches
    ? "auto"
    : "smooth";
}

function setStatus(message, isError = false) {
  appStatus.textContent = message;
  appStatus.classList.toggle("app-status-error", isError);
}

function setBusy(isBusy) {
  requestInProgress = isBusy;
  messageInput.disabled = isBusy || sessionId === null;
  sendButton.disabled = isBusy || sessionId === null;
  newSessionButton.disabled = isBusy;
  sendButton.textContent = isBusy ? "Working…" : "Send question";
  app.setAttribute("aria-busy", String(isBusy));
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
    throw new Error(payload.message || `Request failed (${response.status}).`);
  }
  return payload;
}

function createTextElement(tagName, text, className = "") {
  const element = document.createElement(tagName);
  element.textContent = text;
  if (className) {
    element.className = className;
  }
  return element;
}

function renderWelcome() {
  const item = document.createElement("li");
  item.className = "conversation-welcome";

  const mark = createTextElement("span", "N", "welcome-mark");
  mark.setAttribute("aria-hidden", "true");
  item.append(mark);

  const copy = document.createElement("div");
  copy.append(createTextElement("h3", "Ask about Northstar policies"));
  copy.append(
    createTextElement(
      "p",
      "Try annual leave, flexible work, learning allowances, or another topic from the synthetic employee handbook.",
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
  const lines = text.split("\n");
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) {
      continue;
    }
    if (trimmed.startsWith("### ")) {
      container.append(createTextElement("h4", trimmed.slice(4)));
    } else if (trimmed.startsWith("## ")) {
      container.append(createTextElement("h3", trimmed.slice(3)));
    } else if (trimmed.startsWith("- ")) {
      container.append(createTextElement("p", `• ${trimmed.slice(2)}`));
    } else {
      container.append(createTextElement("p", trimmed));
    }
  }
}

function appendMessage(role, text, status = "") {
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
  }
  item.append(body);
  transcript.append(item);
  scrollToElement(item);
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

function resultStatusLabel(result) {
  if (result.status === "answered") {
    return "Answered";
  }
  if (result.status === "clarification_required") {
    return "Needs clarification";
  }
  if (result.termination_reason === "empty_retrieval") {
    return "No relevant evidence";
  }
  return "Failed";
}

function resultStatusClass(result) {
  if (result.termination_reason === "empty_retrieval") {
    return "status-no-evidence";
  }
  return `status-${result.status}`;
}

function renderSources(retrievedChunks) {
  sources.replaceChildren();
  const sourceIds = new Map();
  if (retrievedChunks.length === 0) {
    sources.append(createTextElement("p", "No source passed the retrieval threshold.", "muted-item"));
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

  const sourceIds = renderSources(result.retrieved_chunks);
  renderCitations(result.citations, sourceIds);
  renderList(
    actions,
    result.action_summaries.map(
      (summary) => `${summary.action.replaceAll("_", " ")}: ${summary.detail}`,
    ),
    "No public action summary returned.",
  );
  executionDetails.open = false;
}

function resetDetails() {
  detailsEmpty.hidden = false;
  turnDetails.hidden = true;
  turnStatus.textContent = "Waiting";
  turnStatus.className = "status-badge status-idle";
  executionDetails.open = false;
  terminationReason.textContent = "—";
  modelCalls.textContent = "—";
  searches.textContent = "—";
  citations.replaceChildren();
  sources.replaceChildren();
  actions.replaceChildren();
}

async function createSession() {
  setBusy(true);
  setStatus("Creating a new session…");
  sessionId = null;
  sessionIdElement.textContent = "Creating session…";
  transcript.replaceChildren();
  renderWelcome();
  resetDetails();
  try {
    const session = await apiRequest("/sessions", { method: "POST" });
    sessionId = session.session_id;
    sessionIdElement.textContent = sessionId;
    setStatus("Session ready. Ask a question about the policy corpus.");
  } catch (error) {
    setStatus(error.message, true);
    sessionIdElement.textContent = "Unavailable";
  } finally {
    setBusy(false);
    if (sessionId !== null) {
      messageInput.focus();
    }
  }
}

async function submitMessage(event) {
  event.preventDefault();
  const message = messageInput.value.trim();
  if (!message || sessionId === null || requestInProgress) {
    return;
  }

  appendMessage("user", message);
  messageInput.value = "";
  setBusy(true);
  setStatus("Searching the corpus and preparing a response…");
  const workingIndicator = appendWorkingIndicator();
  try {
    const response = await apiRequest(`/sessions/${sessionId}/messages`, {
      method: "POST",
      body: JSON.stringify({ message }),
    });
    workingIndicator.remove();
    appendMessage("assistant", response.result.message, resultStatusLabel(response.result));
    renderTurnDetails(response.result);
    setStatus("Turn complete.");
  } catch (error) {
    workingIndicator.remove();
    appendMessage("assistant", error.message, "request_error");
    setStatus(error.message, true);
  } finally {
    setBusy(false);
    messageInput.focus();
  }
}

form.addEventListener("submit", submitMessage);
newSessionButton.addEventListener("click", createSession);
messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    form.requestSubmit();
  }
});

createSession();
