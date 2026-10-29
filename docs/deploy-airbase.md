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

See [Airbase deployment evidence](airbase-deployment.md) for the verified
environments, load-test results, Langfuse checks, and runtime limitations.
