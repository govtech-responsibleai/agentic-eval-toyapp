# Deploying to Airbase

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

## Session memory on b.small

Airbase runs one uvicorn process with the in-memory backend, so every session
and uploaded report lives in that process. The app keeps their estimated size
under `TOYAPP_SESSION_MEMORY_BUDGET_BYTES` by evicting the longest-idle
sessions; see [In-memory backend limits](configuration.md#in-memory-backend-limits).
The budget is only part of the process's memory:

```text
2048 MiB  instance memory (assumed for b.small; confirm below)
 ~700     baseline: Python, torch, embedding model, FAISS index, FastAPI  (to be measured)
 ~500     headroom: PDF parsing, report loading per turn, concurrent model calls,
          Python object overhead the estimate does not count
 ~768     session budget (default)
```

These figures are provisional until measured on the instance. To confirm them:

1. Read the container's memory limit from `/sys/fs/cgroup/memory.max` in an
   Airbase console session, if one is available. This is the limit at which the
   container is killed; `max` means no cgroup limit, and the instance memory
   applies.
2. Deploy and read the `toyapp_memory event=startup` line from the
   application logs. Its `rss_bytes` is the baseline with no sessions stored,
   before the first turn and PDF upload allocate the model's and parser's
   working memory.
3. Run a load test at the intended budget: many sessions uploading PDFs until
   the budget is full and eviction starts, with turns running at the same
   time. Every eviction pass logs a `toyapp_memory event=evict` line. Record
   the highest `peak_rss_bytes`; it includes the baseline, the stored
   sessions, the overhead the estimate omits, and temporary parsing memory.
   The process should stay up, memory should level off, and evicted sessions
   should return `404 session_not_found`.
4. Keep the budget if that peak stays below about 90% of the memory limit.
   Otherwise lower the budget by the excess and repeat the load test.

Memory levels off rather than falling after eviction, because CPython keeps
freed small objects for reuse; large report buffers are returned to the
operating system.

Recommended values for a one-day workshop, until the measurements above
confirm or replace them:

```dotenv
TOYAPP_SESSION_STORE_BACKEND=memory
# Provisional: confirm against the measured baseline and peak.
TOYAPP_SESSION_MEMORY_BUDGET_BYTES=805306368
# The default 30 days never applies within a day; it clears old sessions
# from a long-running instance between events.
TOYAPP_SESSION_IDLE_TTL_SECONDS=2592000
```

To check eviction on staging without generating 768 MiB of uploads, deploy
once with a small budget, such as `TOYAPP_SESSION_MEMORY_BUDGET_BYTES=20971520`
(20 MiB), upload a few reports from several sessions, and confirm the
`toyapp_storage event=evict` lines and the browser's new-session notice.
Restore the workshop value before the event.

If the measured headroom is too small and there is no time to change the
budget, lower `TOYAPP_UPLOAD_MAX_BYTES` and
`TOYAPP_UPLOAD_MAX_REPORTS_PER_SESSION`, which shrink both the largest session
and the temporary memory a single upload needs, or deploy on a larger instance
than `b.small`.

See [Airbase deployment evidence](airbase-deployment.md) for the verified
environments, load-test results, Langfuse checks, and runtime limitations.
