# Capabilities and browser usage

## Capabilities

The application currently provides:

- a default coordinator that can answer, clarify, or select one typed
  specialist route from combined text and authorized report input;
- a standalone bounded RAG mode with `search`, `clarify`, and `answer` actions;
- a report analysis specialist that summarizes uploaded text, Markdown, and
  text-based PDFs with validated page or section references;
- semantic retrieval using a pinned SentenceTransformer and in-memory FAISS;
- live OpenAI-compatible generation or an explicit deterministic provider;
- bounded multi-turn sessions and reports with selectable in-memory or Redis
  storage;
- session-scoped upload and validation for UTF-8 text, Markdown, and text-based
  PDF reports;
- browser and programmatic API access;
- optional Langfuse tracing for coordinator and standalone RAG turns,
  correlated by application session ID;
- controlled traces for the `harmful_content_refusal` metric;
- a non-root Docker image containing the pinned embedding model.

For component boundaries, data flows, and technical design, see
[architecture.md](../architecture.md).

## Using the browser

The page creates a session automatically and puts its UUID in the browser URL
as `/session/{session_id}`.
Reloading or opening that URL restores the retained public conversation.
**New session** starts an independent conversation and updates the URL. If a
shared session is unavailable, the page states that it is unavailable and
creates a replacement. With Redis selected, session URLs remain usable across
application restarts while the Redis data remains available. Their uploaded
reports also remain available. With the default in-memory stores, both remain
process-local.
Enter submits a message and Shift+Enter inserts a newline. Refreshing or
starting a new session returns the page and panel scroll positions to the top.
Assistant replies render sanitized Markdown; your messages remain literal
text.

In the default coordinator mode, use the paperclip beside **Send message** to
upload a `.txt`, `.md`, or text-based `.pdf` report. The **Session file(s)** panel lists
uploaded reports; its `?` icon explains selection. Uploading stores the original
bytes, metadata, and extracted text but does not contact an agent or send a
message. Select one stored report to
include it in messages. The selection remains active across turns until you
deselect, delete, select another file, or start a new session. After a page
reload, choose the file again from the restored list. The list and selected
styling show file state without success notices; upload progress and errors
remain visible. Delete removes stored content from this session, but not
answers or traces already produced from it. The file panel and paperclip are
hidden in standalone policy-only RAG mode.

The default upload limit is 5 MiB per file, with separate limits of 50 PDF
pages and 100,000 extracted characters. These values are configurable in
`.env`; only one file can be selected for a browser message at a time.

For each completed turn, the right panel shows:

- citations and retrieved candidate chunks;
- report references and the public agent/action timeline in coordinator mode;
- similarity scores and source metadata;
- public action summaries and structured model actions;
- controller interventions and citation-validation findings;
- termination reason and model/search counters.

The execution panel is open by default and can be collapsed.
