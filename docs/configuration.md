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
text outside the application process, provide a Redis 7 or later service and
configure:

```dotenv
TOYAPP_SESSION_STORE_BACKEND=redis
TOYAPP_REDIS_URL=redis://127.0.0.1:6379/0
```

Use a `rediss://` URL to connect over TLS, which is required whenever the
connection crosses an untrusted network, and do not commit credentials embedded
in the URL. `TOYAPP_SESSION_MAX_TURNS` and
`TOYAPP_UPLOAD_MAX_REPORTS_PER_SESSION` remain the per-session retention bounds
for both backends. Redis availability is included in `/ready`; requests that
need an unavailable store return safe `503` storage errors.

The Redis backend works with any Redis provider, memory limit, eviction policy,
and number of app instances sharing one Redis. It needs a single Redis node:
Redis Cluster is not supported, because eviction runs as one Lua script over
keys whose names it derives.

### Redis settings

| Variable | Default | Purpose |
|---|---|---|
| `TOYAPP_SESSION_IDLE_TTL_SECONDS` | `2592000` (30 days) | Idle expiry for every key a session owns. Any request on the session renews all of them. |
| `TOYAPP_REDIS_MEMORY_HIGH_WATER` | `0.75` | Fraction of Redis `maxmemory` the app may fill before it evicts idle sessions. |
| `TOYAPP_REDIS_MEMORY_BUDGET_BYTES` | unset | Fallback budget in bytes. Required when `maxmemory` is `0` or unreadable. |
| `TOYAPP_REPORT_LOAD_CONCURRENCY` | `4` | Report-content loads one app instance runs at once, which bounds its temporary memory use. |
| `TOYAPP_REDIS_KEY_PREFIX` | `agentic-eval-toyapp` | Namespace for every key the app writes. |

### Memory budget and eviction

Before creating a session, appending a turn, or storing a report, the app
compares Redis `used_memory` plus the incoming size with its budget. When the
write would not fit, it evicts the longest-idle sessions first, completely and
atomically, until there is room. When nothing can be evicted, the request
returns `503 server_busy` with a `Retry-After` header; this means the service
is at capacity, not that it is down. A turn is checked before the agent runs,
so a refused turn makes no model call and leaves no trace.

The app reads the limit from `INFO memory` only and never uses `CONFIG`, so it
works where a provider blocks administrative commands. The budget is:

- `maxmemory × TOYAPP_REDIS_MEMORY_HIGH_WATER` when `maxmemory` is above zero;
- otherwise `TOYAPP_REDIS_MEMORY_BUDGET_BYTES`;
- the smaller of the two when both are available.

The headroom below `maxmemory` covers Redis's own overhead, memory a background
save temporarily duplicates, and writes already in progress. Set
`TOYAPP_REDIS_MEMORY_BUDGET_BYTES` whenever Redis reports `maxmemory` as `0`
(unlimited) or the value cannot be read. Without either limit, `/ready` reports
the `redis_memory_budget` check as failed and writes return `503` rather than
running with no bound.

Seeded demo sessions never expire and are never evicted. A session with a turn
or report change in progress, and the session making the current request, are
also never evicted.

### Redis eviction policies

The app does not depend on Redis's own eviction policy, but each family
interacts with it differently:

- `noeviction`: Redis rejects writes once full, so the app's own eviction keeps
  memory below the budget. This is the safest policy for the app.
- `volatile-*`: every key the app writes for a session carries an idle TTL, so
  Redis may also remove them; the app's eviction is still what keeps writes
  succeeding.
- `allkeys-*`: Redis may remove any key under memory pressure, including part
  of a session. The app treats a missing session key as a removed session
  (`404 session_not_found`, and the session's leftover report keys are
  deleted) and a missing report as a deleted report (`404 report_not_found`
  when attached, and dropped from listings and exports). No request returns
  `500` for a missing key.

In the browser, a `session_not_found` response during sending, uploading, file
listing, or deletion starts a new session and explains why. An unsent message
is kept in the input box.

### Deploying a new version

Stored data is versioned and not migrated. When deploying a version that
changes the stored layout, flush the Redis database or change
`TOYAPP_REDIS_KEY_PREFIX`; data under the old layout is otherwise ignored but
still occupies memory until it expires or is flushed.

### Operational logs

Each session removed from Redis is logged on one line with fixed fields and no
stored content:

```text
toyapp_storage event=evict session_id=… reason=memory_budget bytes_freed=… reports_removed=…
toyapp_storage event=expire session_id=… reason=expired bytes_freed=0 reports_removed=…
toyapp_storage event=orphan_cleanup session_id=… reason=session_missing bytes_freed=0 reports_removed=…
toyapp_storage event=report_pruned session_id=… reason=content_missing bytes_freed=0 reports_removed=…
toyapp_storage event=capacity_refused reason=… used_bytes=… budget_bytes=… incoming_bytes=…
```

### Docker Compose

Docker Compose configures the app to use its Redis service automatically. That
Redis sets no `maxmemory`, so Compose passes
`TOYAPP_REDIS_MEMORY_BUDGET_BYTES` with a 512 MiB default; override it in
`.env` to suit the host. The Redis service enables append-only persistence in
the named `redis-data` volume, so `make docker-down` followed by
`make docker-up` retains conversations and reports. Running
`docker compose down --volumes` intentionally removes that local data. Redis
contains full original uploads in persistent mode; use only synthetic,
non-sensitive reports unless the deployment's access control, encryption,
retention, and backup policy is appropriate for the data.
