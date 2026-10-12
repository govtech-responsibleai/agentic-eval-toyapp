# Agentic Eval Toy App

A lightweight RAG application for demonstrating and regression-testing METIS
agent evaluation metrics.

The app answers questions about a synthetic employee-policy corpus. It exposes
the same workflow through a browser and a versioned FastAPI service, and can
export each agent turn to Langfuse for evaluation retrieval and scoring.
Repository tooling can also generate controlled control/defect trace pairs with
known expected metric outcomes.

The application currently provides:

- a single bounded agent with `search`, `clarify`, and `answer` actions;
- semantic retrieval using a pinned SentenceTransformer and in-memory FAISS;
- live OpenAI-compatible generation or an explicit deterministic provider;
- process-local multi-turn sessions;
- browser and programmatic API access;
- optional Langfuse tracing correlated by application session ID;
- controlled traces for the `harmful_content_refusal` metric.

For component boundaries, data flows, and technical design, see
[architecture.md](architecture.md).

## Requirements

- Python 3.11 or later
- [uv](https://docs.astral.sh/uv/)
- Make

Live operation additionally requires:

- an API key and base URL for an OpenAI-compatible model provider;
- Langfuse credentials when trace export is enabled;
- network access for the first embedding-model download if it is not cached.

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

### Optional Langfuse tracing

To export normal browser and API turns, add the Langfuse connection values and
enable tracing:

```dotenv
TOYAPP_LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=your-langfuse-public-key
LANGFUSE_SECRET_KEY=your-langfuse-secret-key
LANGFUSE_BASE_URL=https://your-langfuse-host
```

The application session UUID is also used as the Langfuse session ID. Keep
`TOYAPP_LANGFUSE_FLUSH_ON_TURN=true` for evaluation runs so observations are
sent before the HTTP turn completes.

See [.env.example](.env.example) for model, retrieval, agent, session, and
service settings.

## Run the application

Start the default live-model application:

```bash
make run
```

Startup validates configuration, loads the pinned embedding model, embeds the
corpus, and builds the FAISS index before the service accepts requests. The
first local run downloads the embedding model to the Hugging Face cache if it
is not already present. `/ready` becomes available after startup completes.

To run without a live generation request or model-provider credentials, select
the deterministic provider explicitly:

```bash
TOYAPP_GENERATION_PROVIDER=deterministic make run
```

This still uses the real pinned embedding model and FAISS retriever.

Open these local endpoints:

```text
Browser chat:      http://127.0.0.1:8000/
Health:            http://127.0.0.1:8000/health
Readiness:         http://127.0.0.1:8000/ready
Capabilities:      http://127.0.0.1:8000/api/v1/capabilities
OpenAPI interface: http://127.0.0.1:8000/docs
```

## Use the browser

The page creates a session automatically. Enter submits a message and
Shift+Enter inserts a newline. **New session** starts an independent
conversation.

For each completed turn, the right panel shows:

- citations and retrieved candidate chunks;
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
| `GET /health` | Report process liveness and application version. |
| `GET /ready` | Report initialised chat dependencies. |
| `GET /api/v1/capabilities` | Advertise the available target features and actions. |
| `POST /api/v1/sessions` | Create an empty process-local conversation session. |
| `POST /api/v1/sessions/{session_id}/messages` | Execute and retain one synchronous agent turn. |
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
| `make test-live` | Exercise natural-language retrieval and the default API with the pinned embedding model. | May download the embedding model |
| `make test-model-live` | Verify live provider connectivity and structured-action support. | Live model request |
| `make test-langfuse-live` | Export an API turn and retrieve its observations by session ID. | Live Langfuse request |
| `make check` | Run Ruff and strict Pyright checks. | No |
| `make fix` | Apply safe Ruff fixes and formatting. | No |
| `make lock-check` | Confirm `pyproject.toml` and `uv.lock` agree. | No |
| `make clean` | Remove generated Python and test caches. | No |

The offline CI path uses deterministic embeddings, providers, and trace fakes.
Live tests are excluded unless their target is selected explicitly.

## Repository structure

```text
agentic-eval-toyapp/
├── .env.example                              # Non-secret local configuration template
├── .github/
│   └── workflows/
│       └── ci.yml                            # GitHub Actions offline CI workflow
├── .gitignore                                # Secrets, caches, builds, models, and run exclusions
├── .python-version                           # Python version selected by uv
├── AGENTS.md                                 # Repository conventions for coding agents
├── CONTRIBUTING.md                           # Contributor setup and development workflow
├── Makefile                                  # Stable install, run, verification, and trace commands
├── README.md                                 # Project introduction and user guide
├── architecture.md                           # Technical component architecture and data flows
├── pyproject.toml                            # Package metadata, dependencies, and tool configuration
├── uv.lock                                   # Exact direct and transitive dependency lock
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
│       ├── agent.py                          # Bounded structured-action RAG loop
│       ├── app.py                            # FastAPI factory, dependency wiring, and routes
│       ├── config.py                         # Validated environment-backed settings
│       ├── embeddings.py                     # Embedding interfaces, validation, and providers
│       ├── models.py                         # API, retrieval, agent, and session contracts
│       ├── providers.py                      # Live, deterministic, and scripted generation adapters
│       ├── prompts/
│       │   └── system.md                     # Version-controlled agent role and action instructions
│       ├── retrieval.py                      # Corpus loading and exact FAISS search
│       ├── resources/
│       │   └── documents.json                # Synthetic Northstar policy corpus
│       ├── sessions.py                       # Thread-safe bounded in-memory conversation state
│       ├── tracing.py                        # No-op and Langfuse turn recorders
│       ├── static/
│       │   ├── app.js                        # Browser API client and safe response rendering
│       │   └── style.css                     # Responsive local browser styling
│       └── templates/
│           └── index.html                    # Accessible server-rendered chat page
└── tests/
    ├── integration/
    │   ├── test_agent_retrieval.py           # Offline agent-to-FAISS composition test
    │   ├── test_api.py                       # Session/message API contract and failure tests
    │   ├── test_api_live.py                  # Opt-in pinned-model API smoke test
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
        ├── test_config.py                    # Defaults, environment aliases, and secret validation tests
        ├── test_embeddings.py                # Embedding validation and normalization tests
        ├── test_models.py                    # Pydantic contract and consistency tests
        ├── test_package.py                   # Package/distribution version consistency test
        ├── test_providers.py                 # Live and deterministic provider boundary tests
        ├── test_retrieval.py                 # Corpus integrity and deterministic FAISS tests
        ├── test_sessions.py                  # Session isolation, history, and truncation tests
        └── test_tracing.py                   # Offline trace shape and session-correlation tests
```

The local `design.md`, `.env`, embedding caches, FAISS artefacts, and generated
`trace_cases/runs/` manifests are intentionally excluded from source control.

## Current runtime constraints

- Sessions are stored in one process and are lost on restart.
- Agent turns are synchronous; there is no cancellation or background job API.
- The repository does not yet contain Docker packaging.
- METIS's current legacy Langfuse extractor still requires migration for
  Langfuse v4 events-only deployments; the toy app's v4 export and retrieval
  path is independently verified.

See [Current limitations](architecture.md#current-limitations) for technical
details.
