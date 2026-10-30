# Configuration

Settings live in a local `.env` file. Create it with `cp .env.example .env`; see [.env.example](../.env.example) for every setting.

Create a local environment file:

```bash
cp .env.example .env
```

For the default live model provider, replace these placeholders in `.env`:

```dotenv
OPENAI_API_KEY=your-platform-ai-api-key
OPENAI_BASE_URL=https://your-platform-ai-gateway/v1
```

The configured endpoint must support OpenAI-compatible chat completions and
strict structured JSON output. Credentials and `.env` are excluded from Git.

## Langfuse tracing

The environment template configures Langfuse for the default coordinator.
Replace the Langfuse placeholders to export turns; standalone RAG mode is
also available by setting `TOYAPP_AGENT_MODE=rag`:

```dotenv
TOYAPP_LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=your-langfuse-public-key
LANGFUSE_SECRET_KEY=your-langfuse-secret-key
LANGFUSE_BASE_URL=https://your-langfuse-host
```

The application session UUID is also used as the Langfuse session ID. Use the
service base URL, including a reverse-proxy prefix such as `/langfuse`; do not
append the OTLP API path. Keep `TOYAPP_LANGFUSE_FLUSH_ON_TURN=true` for evaluation
runs so the SDK attempts export before the HTTP turn completes. A returned
flush is not proof of successful ingestion. If the per-turn flush itself fails,
the message route returns 503 with code `trace_export_failed` and the turn is not
saved to the session history, so a failed export is distinguishable from an
application error and the trace should be treated as incomplete.

`TOYAPP_LANGFUSE_FLUSH_ON_TURN=false` is a supported deferred mode for load
tests. No flush is made per turn, so the HTTP response no longer waits for
Langfuse. The SDK exports in the background, and one final flush runs when the
application shuts down. Any run that reads traces right after a turn, including
METIS evaluations and trace cases, must keep the default `true`: in deferred
mode a trace may not exist yet when you fetch it, and a crash or kill signal
can lose whatever has not been exported. There is no interval or hybrid
flushing; the SDK's own batching sets the export cadence.

Set `TOYAPP_LANGFUSE_ENABLED=false`
for a run that should not export traces. After changing `.env`, fully restart
`make run` or recreate the Docker container; an already running process keeps
its initial Langfuse client and URL.
Uploaded report text is not exported by default; source IDs and locators are.
For synthetic report-grounding evaluations only, set
`TOYAPP_LANGFUSE_INCLUDE_REPORT_TEXT=true` to export complete extracted text.

Run `make test-langfuse-live` after changing the Langfuse endpoint. It uses
scripted actions, not paid model calls, and verifies whole v4 observation
trees by session ID. If logs show `Failed to export spans batch code: 500`,
the OTLP ingress rejected the trace; a successful read-API request or
`flush_attempted` log does not make that trace complete. Check the Langfuse
ingress/server logs for the configured `.../api/public/otel/v1/traces` route
before using the session as METIS input.

See [.env.example](../.env.example) for model, retrieval, agent, session, report
upload, and service settings.

## Session and report persistence

`make run` uses in-memory conversation and report stores by default. To retain
conversation history, original uploaded files, report metadata, and extracted
text outside the application process, provide a Redis service and configure:

```dotenv
TOYAPP_SESSION_STORE_BACKEND=redis
TOYAPP_REDIS_URL=redis://127.0.0.1:6379/0
```

Use `rediss://` when the Redis connection crosses an untrusted network, and do
not commit credentials embedded in the URL. `TOYAPP_SESSION_MAX_TURNS` and
`TOYAPP_UPLOAD_MAX_REPORTS_PER_SESSION` remain the per-session retention bounds
for both backends. Redis availability is included in `/ready`; requests that
need an unavailable store return safe `503` storage errors.

Docker Compose configures the app to use its Redis service automatically. The
Redis service enables append-only persistence in the named `redis-data` volume,
so `make docker-down` followed by `make docker-up` retains conversations and
reports. Running `docker compose down --volumes` intentionally removes that
local data. Redis contains full original uploads in persistent mode; use only
synthetic, non-sensitive reports unless the deployment's access control,
encryption, retention, and backup policy is appropriate for the data.
