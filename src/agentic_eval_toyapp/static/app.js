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
const terminationReason = document.querySelector("#termination-reason");
const modelCalls = document.querySelector("#model-calls");
const searches = document.querySelector("#searches");
const citations = document.querySelector("#citations");
const sources = document.querySelector("#sources");
const actions = document.querySelector("#actions");

let sessionId = null;
let requestInProgress = false;

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
  item.scrollIntoView({ behavior: "smooth", block: "end" });
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

function renderSources(retrievedChunks) {
  sources.replaceChildren();
  if (retrievedChunks.length === 0) {
    sources.append(createTextElement("p", "No source passed the retrieval threshold.", "muted-item"));
    return;
  }

  for (const chunk of retrievedChunks) {
    const details = document.createElement("details");
    details.className = "source-item";
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
}

function renderTurnDetails(result) {
  detailsEmpty.hidden = true;
  turnDetails.hidden = false;
  turnStatus.textContent = resultStatusLabel(result);
  turnStatus.className = `status-badge status-${result.status}`;
  terminationReason.textContent = result.termination_reason.replaceAll("_", " ");
  modelCalls.textContent = String(result.model_calls);
  searches.textContent = String(result.searches);

  renderList(citations, result.citations, "No citations returned.");
  renderSources(result.retrieved_chunks);
  renderList(
    actions,
    result.action_summaries.map(
      (summary) => `${summary.action.replaceAll("_", " ")}: ${summary.detail}`,
    ),
    "No public action summary returned.",
  );
}

function resetDetails() {
  detailsEmpty.hidden = false;
  turnDetails.hidden = true;
  turnStatus.textContent = "Waiting";
  turnStatus.className = "status-badge status-idle";
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
  try {
    const response = await apiRequest(`/sessions/${sessionId}/messages`, {
      method: "POST",
      body: JSON.stringify({ message }),
    });
    appendMessage("assistant", response.result.message, resultStatusLabel(response.result));
    renderTurnDetails(response.result);
    setStatus("Turn complete.");
  } catch (error) {
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
