# API

Create a session:

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/sessions \
  | uv run python -m json.tool
```

Copy the returned `session_id`, then submit a message:

```bash
curl -s -X POST \
  http://127.0.0.1:8000/api/v1/sessions/<session_id>/messages \
  -H 'Content-Type: application/json' \
  -d '{"message":"How many annual leave days do employees receive?"}' \
  | uv run python -m json.tool
```

Upload a report without sending a chat message:

```bash
curl -s -X POST \
  http://127.0.0.1:8000/api/v1/sessions/<session_id>/files \
  -F 'file=@./report.md;type=text/markdown' \
  | uv run python -m json.tool
```

The response contains a session-scoped `file_id`, filename, media type, byte
and extracted-text sizes, and a SHA-256 checksum. List retained file metadata
without invoking an agent:

```bash
curl -s http://127.0.0.1:8000/api/v1/sessions/<session_id>/files \
  | uv run python -m json.tool
```

This lists the session's uploaded files without invoking an agent. To remove
one stored file, including its extracted text:

```bash
curl -i -X DELETE \
  http://127.0.0.1:8000/api/v1/sessions/<session_id>/files/<file_id>
```

A successful deletion returns HTTP 204. To analyse a retained file, submit
its ID with a message:

```bash
curl -s -X POST \
  http://127.0.0.1:8000/api/v1/sessions/<session_id>/messages \
  -H 'Content-Type: application/json' \
  -d '{"message":"Summarise the main findings and limitations.","file_ids":["<file_id>"]}' \
  | uv run python -m json.tool
```

Upload and message submission are separate operations. An upload alone does
not invoke the agent, add conversation history, generate an assistant message,
or export an agent-turn trace. Standalone RAG mode does not consume uploaded
content. In the default coordinator mode, authorized extracted content is
supplied to the coordinator only when the user submits the message with its
file ID. The
coordinator may derive a focused policy task from that content and route it to
the RAG specialist, but uploaded text never becomes policy evidence or enters
FAISS. A report analysis route uses the selected report content and returns
`report_references` separately from policy citations. An upload without a
message is not analysed.

Retrieve the retained public conversation:

```bash
curl -s http://127.0.0.1:8000/api/v1/sessions/<session_id> \
  | uv run python -m json.tool
```

METIS or another client can use these same routes to generate normal live
traces programmatically. One application session may contain multiple user
turns. The default ten-turn setting limits retained history; it does not stop
the user from submitting additional turns.

## API routes

| Method and path | Purpose |
|---|---|
| `GET /` | Serve the browser chat. |
| `GET /session/{session_id}` | Serve the browser chat shell for a UUID session URL. |
| `GET /health` | Report process liveness and application version. |
| `GET /ready` | Report initialised chat dependencies. |
| `GET /api/v1/capabilities` | Advertise the available target features and actions. |
| `POST /api/v1/sessions` | Create an empty conversation session in the configured store. |
| `POST /api/v1/sessions/{session_id}/files` | Validate and retain one session-scoped report without invoking an agent. |
| `GET /api/v1/sessions/{session_id}/files` | List safe metadata for that session's retained reports. |
| `DELETE /api/v1/sessions/{session_id}/files/{file_id}` | Remove one owned report without changing past turns. |
| `POST /api/v1/sessions/{session_id}/messages` | Execute and retain one agent turn; the client waits for the reply. Returns `503` `server_busy` with `Retry-After: 10` when the admission cap is reached. |
| `GET /api/v1/sessions/{session_id}` | Return the retained public conversation history. |
