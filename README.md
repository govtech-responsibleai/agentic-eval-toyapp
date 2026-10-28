# Agentic Eval Toy App

A lightweight multi-agent application for demonstrating and regression-testing
METIS agent evaluation metrics.

> [!WARNING]
> **This is an experimental evaluation target, not a production-ready
> application.** Use synthetic or non-sensitive data only. The project has no
> availability, durability, or support SLA and must not be used to make or
> automate operational decisions.

The app answers questions about a synthetic employee-policy corpus. It exposes
the same workflow through a browser and a versioned FastAPI service, and can
export coordinator and standalone RAG turns to Langfuse for evaluation retrieval
and scoring.
Repository tooling can also generate controlled control/defect trace pairs with
known expected metric outcomes.

## Table of Contents

- [Capabilities](#capabilities)
- [Project status and support](#project-status-and-support)
- [Requirements](#requirements)
- [Install](#install)
- [Configure](#configure)
- [Run the application](#run-the-application)
- [Use the browser](#use-the-browser)
- [Use the API](#use-the-api)
- [Generate controlled traces](#generate-controlled-traces)
- [Verify the application](#verify-the-application)
- [Repository structure](#repository-structure)
- [Current runtime constraints](#current-runtime-constraints)
- [Development](#development)
- [Security](#security)
- [License](#license)

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
[architecture.md](architecture.md).

## Project status and support

The repository is maintained as a research and evaluation demonstration with
a **best-effort community support tier**. It supports Redis-backed conversation
sessions and uploaded reports, answers each agent turn on the request, and
prioritises controlled evaluation behaviour over production scalability.

- For reproducible bugs and feature requests, open a
  [GitHub issue](https://github.com/govtech-responsibleai/agentic-eval-toyapp/issues).
- For contribution setup and review expectations, see
  [CONTRIBUTING.md](CONTRIBUTING.md).
- Do not report suspected vulnerabilities in a public issue. Follow
  [SECURITY.md](SECURITY.md) instead.

## Requirements

- Python 3.11 or later
- [uv](https://docs.astral.sh/uv/)
- Make
- Redis when persistent sessions are enabled outside Docker Compose

Alternatively, Docker with Compose can build and run the packaged service
without installing Python or uv on the host.

Live operation additionally requires:

- an API key and base URL for an OpenAI-compatible model provider;
- Langfuse credentials when trace export is enabled;
- network access for the first local embedding-model download if it is not
  cached, or during the Docker image build.

## Install

Clone the repository, enter it, and install the locked dependencies:

```bash
make install
```

Run the deterministic offline checks:

```bash
make ci
```

## Configure

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

### Langfuse tracing

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

See [.env.example](.env.example) for model, retrieval, agent, session, report
upload, and service settings.

### Session and report persistence

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

### Backpressure and model-call limits

Overload is refused before a turn starts, and model calls inside an admitted
turn wait for a slot but are never rejected. Three environment variables set
the limits; the defaults suit a key with 200 requests per minute (RPM) and 2M
tokens per minute (TPM) that only this app uses.

| Variable | Default | Effect |
|---|---|---|
| `TOYAPP_MAX_PENDING_TURNS` | `128` | Turns admitted at once. The next message request gets `503` with code `server_busy` and `Retry-After: 10`. It runs no agent code, creates no trace and stores no turn. |
| `TOYAPP_LLM_MAX_REQUESTS_PER_MINUTE` | `180` | Model-call starts are spaced evenly at this rate across the whole process. |
| `TOYAPP_LLM_MAX_CONCURRENT_CALLS` | `16` | Most model calls in flight at once; a guard against slow calls. |

Each value must be at least 1. Time spent waiting for a slot does not count
against `TOYAPP_GENERATION_TIMEOUT_SECONDS`, and the OpenAI client keeps
`max_retries=0` so a real provider `429` stays visible as `rate_limited`. There
is no token limiter. `server_busy` and `agent_unavailable` are both `503`;
tell them apart by the `code` field.

A turn makes several model calls: a policy question about three (coordinator,
RAG search step, RAG answer step) and a report analysis two.

| Traffic | Binding limit | Turns per minute at 200 RPM / 2M TPM |
|---|---|---|
| Policy question | requests | about 60 |
| Report analysis, small file | requests | up to 90 |
| Report analysis, PDF at the size cap | tokens | about 24 |

Token figures are estimates at 4 characters per token, not measurements.

#### Tuning

Use this when the key's limits change.

1. **Request rate.** Set `TOYAPP_LLM_MAX_REQUESTS_PER_MINUTE` to
   `floor(0.9 x key RPM)`. The margin covers sliding-window counting and clock
   differences. If another tool shares the key, subtract its usage from the RPM
   first. Do not go above 90%: the cost of a `429` is a failed turn.
2. **Token check.** Confirm `turns per minute x tokens per turn <= 0.9 x key
   TPM`, taking tokens per turn from Langfuse generation usage or the provider
   dashboard. If it fails for your traffic (typically large PDFs), lower the
   request rate until it holds.
3. **Concurrency guard.** Set `TOYAPP_LLM_MAX_CONCURRENT_CALLS` to about
   `1.5 x (request rate per second) x (mean call latency in seconds)`. This is
   Little's law with 50% headroom. Get the latency from Langfuse generation
   durations, or time a few turns serially and divide by `model_calls` in each
   response. Report analysis carries a much larger prompt and can run tens of
   seconds, so a 30-second call needs ten times the slots of a 3-second call;
   its latency has not been measured. Too low shows as a low call rate with the
   provider idle; too high is harmless while pacing is correct.
4. **Admission cap.** Set `TOYAPP_MAX_PENDING_TURNS` to about
   `(request rate per minute / calls per turn) x acceptable wait in minutes`.
   The cap bounds queue time; it does not add capacity.

Worked examples, assuming 3-second calls, 3 calls per policy turn and a
2-minute acceptable wait (replace with measured values):

| Key RPM | Request rate | Concurrency guard | Policy turns per minute | Admission cap |
|---|---|---|---|---|
| 200 | 180 | 16 | 60 | 128 |
| 500 | 450 | 36 | 150 | 300 |
| 1000 | 900 | 68 | 300 | 600 |

At about one policy turn per second, 100 simultaneous turns take on the order
of 100 seconds to drain. The default cap accepts a 100-request burst but its
last request waits about that long, so clients need longer timeouts or a
smaller cap.

#### Verify after changing anything

Run a controlled burst (for example 100 concurrent turns on distinct sessions)
against the live provider. It spends real quota, so it is never part of
`make ci`. Then read the signals, changing one setting at a time:

| Signal | Meaning | Action |
|---|---|---|
| Logs show a provider failure with code `rate_limited` | The provider refused a call | Lower the request rate; check for other users of the key; recheck the token budget |
| Responses with `server_busy` | Admission cap reached | Expected beyond capacity; raise the cap only if clients can wait longer |
| Slow turns, no failures, dashboard well below limits | Settings too conservative | Raise the request rate toward 90% of RPM, then the concurrency guard |
| Call rate far below the request rate setting | Concurrency guard too low | Raise it per step 3 |

#### Limits that do not move with the key

- Retrieval stays single-slot and handles roughly 190 searches per second.
- The admission count, pacing clock and semaphore live in one process. Keep a
  single worker; with several, each enforces its own limits and the provider
  sees their sum.
- A provider may cap simultaneous requests separately from RPM and TPM. If so,
  keep the concurrency guard at or below that limit.
- A client that disconnects mid-turn still holds its admission slot until the
  turn finishes.

## Run the application

Start the default live-model coordinator application:

```bash
make run
```

In both RAG and coordinator modes, startup validates configuration, loads the
pinned embedding model, embeds the corpus, and builds the FAISS index before
the service accepts requests. The first local run downloads the embedding model
to the Hugging Face cache if it is not already present. `/ready` becomes
available after the selected top-level agent and its required dependencies are
initialised.

To run without a live generation request or model-provider credentials, select
the deterministic provider explicitly:

```bash
TOYAPP_GENERATION_PROVIDER=deterministic make run
```

This still uses the real pinned embedding model and FAISS retriever.

To run the standalone RAG agent, select it explicitly:

```bash
TOYAPP_AGENT_MODE=rag \
TOYAPP_GENERATION_PROVIDER=deterministic \
TOYAPP_LANGFUSE_ENABLED=false \
make run
```

Coordinator mode accepts the same session and message API. It can answer or
clarify directly and records public coordinator and RAG actions in
`execution_steps`. A `route_rag` decision delegates one focused policy task to
the existing FAISS-backed RAG agent; a `route_analysis` decision sends selected
report content to the analysis specialist. The specialist receives the full
extracted report in one model call, using a separate configurable 16,384-token
output allowance by default. The result's `report_references` identify exact
report pages or sections.
When Langfuse is enabled, coordinator turns export a nested trace. The
coordinator exposes a browser file panel for uploaded reports.

Open these local endpoints:

```text
Browser chat:      http://127.0.0.1:8000/
Health:            http://127.0.0.1:8000/health
Readiness:         http://127.0.0.1:8000/ready
Capabilities:      http://127.0.0.1:8000/api/v1/capabilities
OpenAPI interface: http://127.0.0.1:8000/docs
```

### Run with Docker

The image installs the locked runtime dependencies and downloads the exact
embedding-model revision during the build. Runtime model loading is local-only,
so container startup does not contact Hugging Face.

After creating `.env`, build and start the default live-model service:

```bash
make docker-up
```

For a local run without Platform AI credentials or model-call cost:

```bash
TOYAPP_GENERATION_PROVIDER=deterministic \
TOYAPP_LANGFUSE_ENABLED=false \
make docker-up
```

The same endpoints are available at `http://127.0.0.1:8000`. Compose also starts
Redis for persistent conversation and report storage. Inspect or stop the
services with:

```bash
make docker-logs
make docker-down
```

The startup log states the selected top-level agent, generation provider, and
whether Langfuse is configured. `/ready` and `/api/v1/capabilities` report
whether trace export is operational for that agent mode.

### Deploy to Airbase

The committed `airbase.json` targets
`rai-poc/metis-agentic-eval-toyapp` on port `8000` using a `b.small` instance.
The runtime image accepts Airbase's injected `PORT`, runs as UID/GID 999, and
keeps credentials outside the image. Airbase deploys this one application image;
it does not start the Redis service from `compose.yaml`. The Airbase environment
therefore selects the in-memory session and report backend explicitly.

Create a gitignored `.env.staging` for the first infrastructure test:

```dotenv
TOYAPP_AGENT_MODE=coordinator
TOYAPP_GENERATION_PROVIDER=deterministic
TOYAPP_LANGFUSE_ENABLED=false
TOYAPP_SESSION_STORE_BACKEND=memory
```

Build an immutable AMD64 image, run the complete deterministic container smoke
test, and deploy that same local image to the named staging environment:

```bash
IMAGE_TAG="metis-agentic-eval-toyapp:$(git rev-parse --short HEAD)"
make docker-build \
  DOCKER_PLATFORM=linux/amd64 \
  DOCKER_IMAGE="$IMAGE_TAG"
make docker-smoke-run DOCKER_IMAGE="$IMAGE_TAG"
airbase deploy staging --image "$IMAGE_TAG" --yes
```

The staging URL is
`https://staging--metis-agentic-eval-toyapp.app.tc1.airbase.sg`. Verify
`/health`, `/ready`, and `/api/v1/capabilities` before opening the browser.
Application logs are available in the Airbase web console; the CLI has no logs
or deployment-status command.

Run the issue's concurrency test after the staging canary passes and while
Langfuse remains disabled, so the result measures Airbase and the application
without external trace-export latency:

```bash
make load-test \
  LOAD_TEST_URL=https://staging--metis-agentic-eval-toyapp.app.tc1.airbase.sg \
  LOAD_TEST_USERS=60
```

Each worker starts at the same barrier, creates an isolated session, submits one
representative policy question, and reports failures, throughput, and latency
percentiles. The command exits non-zero if any flow fails.

After the load test passes, enable trace export in `.env.staging` and redeploy
the same image:

```dotenv
TOYAPP_LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=replace-with-staging-public-key
LANGFUSE_SECRET_KEY=replace-with-staging-secret-key
LANGFUSE_BASE_URL=https://replace-with-langfuse-service-base-url
TOYAPP_LANGFUSE_ENVIRONMENT=staging
TOYAPP_LANGFUSE_FLUSH_ON_TURN=true
```

Submit one deterministic chat turn and verify its application session UUID in
Langfuse. A `langfuse_tracing` capability of `true` means tracing is configured;
the trace lookup is the proof that export succeeded.

For the unnamed production environment, put the intended production values in
the already-ignored `.env` and deploy the exact tested image without an
environment argument:

```bash
airbase deploy --image "$IMAGE_TAG" --yes
```

Do not use `airbase deploy production`: that creates a separate named
`production--...` environment. The application does not implement
authentication or rate limiting, so keep the deterministic provider on an
unrestricted public deployment unless access control and model cost have been
explicitly accepted.

See [Airbase deployment evidence](docs/airbase-deployment.md) for the verified
environments, load-test results, Langfuse checks, and runtime limitations.

### Operational error logs

Failed provider calls, invalid model actions, upload validation, specialist
delegation, retrieval, and trace-export failures produce classified events in
the `make run` terminal or `make docker-logs` output. Search for
`toyapp_failure`. A message attempt has a `session_id` and a separate internal
`turn_id` so multiple failures in one session can be distinguished. Uploads
have a session ID but no turn ID because upload does not invoke the agent.
Events include the agent, operation, safe error code, and relevant model,
attempt, output budget, duration, or HTTP status. Expected upload rejections
and grounding findings are warnings; failures are errors. These diagnostics
are not shown in the browser or written to a separate file. Raw exception
messages, prompts, report contents, and credentials are intentionally omitted.
For a provider response that could not be used, the same event also records
`finish_reason`, `input_tokens`, `output_tokens`, `reasoning_tokens`, and
`total_tokens` when supplied by the gateway; missing values appear as `-`.
`max_tokens` is the configured output cap, not the number used. A
`finish_reason=length` reports that generation stopped at a limit; token
counts are provider-reported usage, not a local estimate. Transport failures
may have no response or usage fields. The configured 90-second timeout applies
to each live model call, not the entire coordinator-led turn; successful calls
do not emit usage in these operational error logs.

The first image build needs network access for locked Python packages, the base
images, and the pinned embedding model. Platform AI and Langfuse remain external
services; their credentials are passed from `.env` only when the container
starts and are not copied into the image.

The image uses the CPU-only PyTorch wheel source and does not include CUDA
libraries. In a managed network, `make docker-build` automatically forwards the
certificate bundle named by the host's `SSL_CERT_FILE` through a BuildKit
secret. Override it explicitly when necessary:

```bash
make docker-build DOCKER_CA_CERT=/absolute/path/to/combined-ca-bundle.pem
```

TLS verification remains enabled. The selected public CA bundle is copied into
the runtime image so the unprivileged service can read it while only root can
modify it. Supply public CA certificates only; do not include private keys.

## Use the browser

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

## Use the API

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

### API routes

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

## Generate controlled traces

Controlled generation requires the live OpenAI-compatible provider and enabled
Langfuse configuration. A live command can incur provider cost.

List supported metrics and variants:

```bash
make trace-cases-list
```

Generate every available metric and variant:

```bash
make trace-case
```

Apply optional filters:

```bash
make trace-case METRIC=harmful_content_refusal
make trace-case VARIANT=control
make trace-case METRIC=harmful_content_refusal VARIANT=control
make trace-case METRIC=harmful_content_refusal VARIANT=defect
```

With no `METRIC`, all metrics are selected. With no `VARIANT`, both control and
defect variants are selected. Generated local run information is written under
the gitignored `trace_cases/runs/` directory using the Langfuse-correlated
session UUID as the filename.

See [Controlled trace generation](architecture.md#10-controlled-trace-generation)
for the technical flow and
[`harmful_content_refusal` documentation](trace_cases/metrics/harmful_content_refusal/README.md)
for the metric-specific case contract.

## Verify the application

| Command | Purpose | Network or external service |
|---|---|---|
| `make ci` | Run the lock check, lint, format check, strict type check, and all offline tests. | No |
| `make test` | Run all offline tests without the static checks. | No |
| `make test-unit` | Run unit tests only. | No |
| `make test-live` | Exercise natural-language retrieval through standalone RAG and the default coordinator API with the pinned embedding model. | May download the embedding model |
| `make test-model-live` | Verify live provider connectivity and structured-action support. | Live model request |
| `make test-langfuse-live` | Export scripted RAG and multi-turn coordinator traces, then verify complete v4 observations by session ID. | Live Langfuse requests |
| `make docker-build` | Build the non-root runtime image and cache the pinned embedding model in it. | Base images, packages, and embedding model during build |
| `make docker-up` | Build and start the Compose service using runtime values from `.env`. | Depends on selected provider and tracing settings |
| `make docker-down` | Stop and remove the Compose service. | No |
| `make docker-logs` | Follow logs from the Compose service. | No |
| `make docker-smoke` | Build the image and validate readiness plus one deterministic real-retrieval turn. | Build-time downloads only |
| `make load-test LOAD_TEST_URL=https://...` | Run isolated session/message flows for 60 simultaneous users by default. | Calls the selected deployment |
| `make check` | Run Ruff and strict Pyright checks. | No |
| `make fix` | Apply safe Ruff fixes and formatting. | No |
| `make lock-check` | Confirm `pyproject.toml` and `uv.lock` agree. | No |
| `make clean` | Remove generated Python and test caches. | No |

The offline CI path uses deterministic embeddings, providers, and trace fakes.
Live tests are excluded unless their target is selected explicitly.

## Repository structure

```text
agentic-eval-toyapp/
├── .dockerignore                             # Secrets, tests, caches, and local artefacts excluded from image builds
├── .env.example                              # Non-secret local configuration template
├── .github/
│   ├── CODEOWNERS                            # Default maintainer review ownership
│   ├── dependabot.yml                        # Automated dependency update configuration
│   └── workflows/
│       └── ci.yml                            # GitHub Actions offline CI workflow
├── .gitignore                                # Secrets, caches, builds, models, and run exclusions
├── .python-version                           # Python version selected by uv
├── AGENTS.md                                 # Repository conventions for coding agents
├── airbase.json                              # Airbase project, port, and instance configuration
├── CONTRIBUTING.md                           # Contributor setup and development workflow
├── Dockerfile                                # Multi-stage non-root runtime image with cached embeddings
├── LICENSE                                   # MIT licence terms
├── Makefile                                  # Stable install, run, verification, and trace commands
├── README.md                                 # Project introduction and user guide
├── SECURITY.md                               # Private vulnerability disclosure process
├── architecture.md                           # Technical component architecture and data flows
├── compose.yaml                              # Local application and persistent Redis services
├── docs/
│   ├── airbase-deployment.md                 # Verified deployment and load-test evidence
│   └── github-workflow.md                    # Internal and external collaboration playbook
├── pyproject.toml                            # Package metadata, dependencies, and tool configuration
├── uv.lock                                   # Exact direct and transitive dependency lock
├── scripts/
│   ├── __init__.py                           # Build-tool package marker
│   ├── cache_embedding_model.py              # Image-build download and model-dimension validation
│   ├── docker_smoke.py                       # Container readiness and deterministic RAG smoke test
│   └── load_test.py                          # Concurrent deployed session/message verification
├── trace_cases/
│   ├── __init__.py                           # Isolated trace-generation harness package
│   ├── __main__.py                           # `python -m trace_cases` entry point
│   ├── catalog.py                            # Case discovery and cross-file integrity validation
│   ├── cli.py                                # Metric/variant listing and generation CLI
│   ├── controlled_client.py                  # Live answer-substitution completion boundary
│   ├── models.py                             # Local case, label, capture, and manifest contracts
│   ├── runner.py                             # Normal API execution and private manifest writing
│   ├── runs/                                 # Gitignored generated manifests, created as needed
│   └── metrics/
│       └── harmful_content_refusal/
│           ├── README.md                     # Metric contract, injection design, and verification
│           ├── cases.json                    # Matched control/defect inputs and effective answers
│           └── ground_truth.json             # Expected outcomes retained outside Langfuse
├── src/
│   └── agentic_eval_toyapp/
│       ├── __init__.py                       # Package version
│       ├── agent.py                          # Bounded RAG loop and coordinator specialist adapter
│       ├── analysis.py                       # Report specialist, model adapters, and reference validation
│       ├── analysis_models.py                # Typed report-analysis requests and model actions
│       ├── app.py                            # FastAPI factory, dependency wiring, and routes
│       ├── config.py                         # Validated environment-backed settings
│       ├── coordinator.py                    # Bounded coordinator decisions and specialist dispatch
│       ├── diagnostics.py                    # Classified operational logs and per-turn correlation
│       ├── embeddings.py                     # Embedding interfaces, validation, and providers
│       ├── models.py                         # API, retrieval, agent, and session contracts
│       ├── multi_agent_tracing.py            # Coordinator-owned nested specialist trace context
│       ├── orchestration_models.py           # Coordinator actions, attachments, and handoff contracts
│       ├── providers.py                      # Live, deterministic, and scripted generation adapters
│       ├── prompts/
│       │   ├── analysis.md                   # Report grounding and findings instructions
│       │   ├── coordinator.md                # Coordinator routing and trust-boundary instructions
│       │   └── system.md                     # RAG role and action instructions
│       ├── retrieval.py                      # Corpus loading and exact FAISS search
│       ├── report_sources.py                 # Complete report text with stable page and section locators
│       ├── reports.py                        # Bounded in-memory and Redis report storage
│       ├── resources/
│       │   └── documents.json                # Synthetic Northstar policy corpus
│       ├── sessions.py                       # Event-loop-safe in-memory and Redis conversation stores
│       ├── static/
│       │   ├── app.js                        # Browser API client and sanitized Markdown rendering
│       │   ├── style.css                     # Responsive panel layout and browser styling
│       │   └── vendor/
│       │       ├── README.md                 # Pinned browser bundle versions and checksums
│       │       ├── LICENSE.dompurify         # DOMPurify Apache license text
│       │       ├── LICENSE.markdown-it       # markdown-it MIT license text
│       │       ├── markdown-it.umd.min.js    # Local Markdown parser for assistant replies
│       │       └── purify.min.js             # Local HTML sanitizer for rendered replies
│       ├── tracing.py                        # No-op and Langfuse standalone turn recorders
│       └── templates/
│           └── index.html                    # Accessible server-rendered chat page
└── tests/
    ├── fixtures/
    │   └── test_report.md                    # Synthetic report for analysis verification
    ├── integration/
    │   ├── test_agent_retrieval.py           # Offline agent-to-FAISS composition test
    │   ├── test_api.py                       # Session/message API contract and failure tests
    │   ├── test_api_live.py                  # Opt-in pinned-model API smoke test
    │   ├── test_container_contract.py        # Static image, context, and Compose safety contracts
    │   ├── test_diagnostics_api.py            # API-to-provider error correlation and redaction
    │   ├── test_frontend.py                  # Browser page, asset, and rendering-safety tests
    │   ├── test_health_api.py                # Health, readiness, capability, and OpenAPI tests
    │   ├── test_langfuse_live.py             # Opt-in Langfuse export and v4 retrieval test
    │   ├── test_model_live.py                # Opt-in live-provider structured-action test
    │   └── test_retrieval_live.py            # Opt-in pinned-model semantic retrieval tests
    ├── trace_cases/
    │   ├── test_catalog.py                   # Controlled-pair and ground-truth integrity tests
    │   ├── test_cli.py                       # Metric/variant selection-plan tests
    │   ├── test_controlled_client.py         # Substitution, pass-through, and trace-isolation tests
    │   ├── test_makefile.py                  # Optional Make-filter behavior tests
    │   └── test_runner.py                    # API reuse and private-manifest tests
    └── unit/
        ├── test_agent.py                     # Agent actions, bounds, repair, and findings tests
        ├── test_analysis.py                  # Report decisions, grounding, limits, and provider tests
        ├── test_config.py                    # Defaults, environment aliases, and secret validation tests
        ├── test_coordinator.py               # Coordinator routing, limits, providers, and handoff tests
        ├── test_diagnostics.py               # Safe error classification, context reset, and redaction
        ├── test_build_scripts.py             # Container smoke validation and build-input edge cases
        ├── test_embeddings.py                # Embedding validation and normalization tests
        ├── test_models.py                    # Pydantic contract and consistency tests
        ├── test_orchestration_models.py      # Coordinator action and handoff schema tests
        ├── test_package.py                   # Package/distribution version consistency test
        ├── test_providers.py                 # Live and deterministic provider boundary tests
        ├── test_report_sources.py            # Complete report source and locator edge cases
        ├── test_retrieval.py                 # Corpus integrity and deterministic FAISS tests
        ├── test_reports.py                   # Report extraction, limits, hashing, and isolation tests
        ├── test_redis_reports.py             # Offline Redis report persistence and integrity tests
        ├── test_redis_sessions.py            # Offline Redis persistence and failure-boundary tests
        ├── test_sessions.py                  # Session isolation, history, and truncation tests
        ├── test_multi_agent_tracing.py       # Coordinator trace hierarchy, privacy, and usage tests
        └── test_tracing.py                   # Offline standalone trace shape and session-correlation tests
```

The local `design.md`, `.env`, embedding caches, FAISS artefacts, and generated
`trace_cases/runs/` manifests are intentionally excluded from source control.

## Current runtime constraints

- Conversation sessions and complete report records are persistent and shared
  across app workers when Redis is selected; the default source-based `make
  run` configuration remains process-local.
- Report storage is bounded per session. In Redis mode it includes original
  upload bytes, metadata, extracted text, and PDF page boundaries.
- Text-based PDFs are supported, but scanned PDFs and OCR are not.
- Standalone RAG mode does not consume report contents.
- The default coordinator can answer, clarify, delegate policy tasks to RAG, and
  analyse selected reports. When enabled, its turn and selected route are
  exported as one nested Langfuse trace.
- Clients wait for each turn's reply; there is no streaming, cancellation, or background job API.
- The runtime image contains the application only; controlled trace generation
  remains source-repository tooling.
- METIS's current legacy Langfuse extractor still requires migration for
  Langfuse v4 events-only deployments; the toy app's v4 export and retrieval
  path is independently verified.

See [Current limitations](architecture.md#current-limitations) for technical
details.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full contributor guide and
[docs/github-workflow.md](docs/github-workflow.md) for the collaboration
playbook.

```bash
make check          # lint + format-check + typecheck (non-mutating)
make fix            # auto-fix lint issues and format code
make ci             # full deterministic offline CI suite
make clean          # remove generated and cached files
```

## Security

Do not report security vulnerabilities in public issues. See
[SECURITY.md](SECURITY.md) for how to report a vulnerability.

## License

Agentic Eval Toy App is released under the [MIT License](LICENSE).
