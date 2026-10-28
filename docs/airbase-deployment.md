# Airbase deployment evidence

Issue: `govtech-responsibleai/agentic-eval-toyapp#23`

Verification date: 28 October 2026

## Deployment

- Airbase project: `rai-poc/metis-agentic-eval-toyapp`
- Tested image tag: `metis-agentic-eval-toyapp:airbase-local`
- Tested local image ID: `sha256:fe1447238372c7a09c72d23c500466f0690739492b58226782007503e3c1c783`
- Image platform: `linux/amd64`
- Runtime identity: UID 999, GID 999
- Port: 8000, with runtime `PORT` override support
- Instance type: `b.small`
- Session and report backend: in-memory
- Staging URL: <https://staging--metis-agentic-eval-toyapp.app.tc1.airbase.sg>
- Production URL: <https://metis-agentic-eval-toyapp.app.tc1.airbase.sg>

Both environments passed `/health`, `/ready`, `/api/v1/capabilities`, browser
asset, shareable-session, live policy-turn, and session-reload checks. The
production readiness response reported configured Langfuse tracing. CSS and
JavaScript returned HTTP 200 over HTTPS using scheme-independent `/static/...`
URLs.

The live model and Langfuse integration were verified in staging before
production deployment. A production canary was retrieved in Langfuse by its
application session ID with the expected coordinator-owned trace and nested RAG
observations.

## Load tests

Each load-test worker started at a common barrier, created an isolated session,
and submitted one annual-leave policy question through the synchronous API.

### Staging

The first deterministic 60-user run completed 55 flows and received five
Airbase gateway HTTP 502 responses. The app remained healthy and retained an
existing in-memory session, so there was no observed container restart. An
immediate repeat completed all 60 flows successfully. The initial failed run is
retained as evidence of transient gateway behaviour rather than omitted.

Live-model and Langfuse-enabled staging checks were subsequently completed
before production rollout.

### Production

The live-model and Langfuse-enabled production run completed successfully:

| Measurement | Result |
|---|---:|
| Concurrent users | 60 |
| Successful flows | 60 |
| Failed flows | 0 |
| Total elapsed time | 22.914 s |
| Throughput | 2.619 users/s |
| Minimum latency | 8.199 s |
| p50 latency | 14.533 s |
| p95 latency | 20.188 s |
| p99 latency | 22.901 s |
| Maximum latency | 22.901 s |

Langfuse was checked for session-correlated production traces after the run,
and Airbase logs were checked for trace-export failures.

## Coverage and limitations

The verification covered Airbase routing, application startup, concurrent
session creation, live structured model calls, semantic retrieval, response
serialization, browser assets, session URLs, and Langfuse export. It did not
establish a performance or availability service-level objective.

Airbase deploys only the application image, not the Redis service in
`compose.yaml`. Sessions, uploaded files, metadata, and extracted report text
therefore use process memory and are lost when the instance restarts or is
replaced. The deployment has no application authentication or rate limiting;
live-provider credentials on a publicly reachable environment can incur cost.
Provider quota, model availability, Airbase gateway behaviour, and Langfuse
ingestion remain external dependencies.
