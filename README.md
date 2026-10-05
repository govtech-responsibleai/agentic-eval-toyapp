# Agentic Eval Toy App

A lightweight, single-agent RAG application for validating METIS metrics with controlled functional cases.

## MVP summary

The completed MVP will:

- answer questions over a fixed synthetic policy corpus;
- use SentenceTransformer embeddings and an in-memory FAISS index;
- expose the same workflow through FastAPI and a browser chat;
- record agent, model, and retrieval activity in Langfuse;
- return session identifiers that METIS can use to retrieve traces;
- provide paired control/defect cases for scorer regression tests;
- run locally or in Docker with a uv and Makefile workflow.

The MVP uses one bounded RAG agent. It does not use a leader or specialist agents.

## Development setup

Requirements:

- Python 3.11 or later
- [uv](https://docs.astral.sh/uv/)
- Make

Install and run the offline checks:

```bash
make install
make ci
```

Useful commands:

| Command | Purpose |
|---|---|
| `make run` | Start the local FastAPI development server. |
| `make lock-check` | Verify that `pyproject.toml` and `uv.lock` agree. |
| `make check` | Run linting, formatting checks, and strict type checking. |
| `make fix` | Apply safe lint fixes and formatting. |
| `make test` | Run all offline tests. |
| `make test-unit` | Run unit tests only. |
| `make test-live` | Run opt-in checks against the pinned embedding model. |
| `make test-model-live` | Make a real structured request to the configured model provider. |
| `make test-langfuse-live` | Export and retrieve one synthetic API turn using configured Langfuse. |
| `make clean` | Remove generated Python and test caches. |

Copy `.env.example` to `.env`, then replace the `OPENAI_API_KEY` and
`OPENAI_BASE_URL` placeholders with credentials for an OpenAI-compatible
provider such as Platform AI. Start the model-backed application with:

```bash
make run
```

To run the same browser and API without credentials or model charges, select
the deterministic provider explicitly for that process:

```bash
TOYAPP_GENERATION_PROVIDER=deterministic make run
```

Langfuse tracing is independently opt-in. Configure its three connection
values in `.env`, set `TOYAPP_LANGFUSE_ENABLED=true`, and leave
`TOYAPP_LANGFUSE_FLUSH_ON_TURN=true` for evaluation runs. The app session UUID
then becomes the Langfuse session ID used by METIS.

Then inspect:

```text
http://127.0.0.1:8000/
http://127.0.0.1:8000/health
http://127.0.0.1:8000/ready
http://127.0.0.1:8000/api/v1/capabilities
http://127.0.0.1:8000/docs
```

Application startup validates the selected generation provider, loads the
pinned embedding model, embeds the corpus, and builds the FAISS index before
the server accepts requests. If the embedding model is not already cached,
startup first downloads it from Hugging Face. Missing live credentials fail
startup rather than silently selecting deterministic behavior. `/ready`
reports the completed startup state; it does not make a billable provider call.

Create a session:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/sessions
```

Copy the returned `session_id`, then submit and inspect a turn:

```bash
curl -X POST \
  http://127.0.0.1:8000/api/v1/sessions/<session_id>/messages \
  -H 'Content-Type: application/json' \
  -d '{"message":"How many annual leave days do employees receive?"}'

curl http://127.0.0.1:8000/api/v1/sessions/<session_id>
```

The default live provider makes OpenAI-compatible structured-output requests.
On every new user turn, the model may search, ask for clarification, or answer
directly. The system prompt identifies the app as a Northstar employee-policy
assistant and lists all eight supported policy areas, so conversational and
capability questions do not require retrieval. Questions about policy contents
should use semantic search, but tool choice remains model behavior for METIS to
evaluate rather than a controller-enforced outcome.

A search action includes a model-selected `top_k` from 1 through 8. Retrieved
chunks are candidates: the model analyses their relevance and cites only the
subset used. Empty and irrelevant retrieval results are passed back to the
model, which may answer without citations. The deterministic provider follows
the same typed action loop but performs no external model calls.

The controller validates API and action schemas, enforces tool/model-call
limits, rejects duplicate search execution, and allows one structured-action
repair. It does not replace a completed answer because retrieval was skipped or
citations were imperfect. Answers preserve all structured `model_actions` in
order and expose
non-blocking `validation_findings` for duplicate citations, citations that were
not retrieved, and disagreement between inline and structured citations.
`failed` is reserved for technical or unrecoverable protocol outcomes.

The browser page creates a session automatically. It displays the answer,
citations, retrieved chunks and similarity scores, public action summaries,
controller events, citation-validation findings, ordered structured model
actions and the final model action,
termination reason, model-call count, and search count returned by the same
API used by METIS. Execution details are open by default and remain
collapsible. Use **New session** to clear the browser transcript and create a
separate backend session. Enter sends a message; Shift+Enter inserts a new
line.

| Method and path | Purpose |
|---|---|
| `GET /` | Serve the browser chat client. |
| `GET /health` | Report process liveness after application startup. |
| `GET /ready` | Report the chat dependencies initialised during startup. |
| `GET /api/v1/capabilities` | Advertise supported target features and actions. |
| `POST /api/v1/sessions` | Create an empty process-local conversation session. |
| `POST /api/v1/sessions/{session_id}/messages` | Run and retain one synchronous RAG turn. |
| `GET /api/v1/sessions/{session_id}` | Return the retained public conversation history. |

## Repository structure

```text
agentic-eval-toyapp/
├── .env.example                              # Non-secret environment-variable template
├── .github/
│   └── workflows/
│       └── ci.yml                            # GitHub Actions offline CI workflow
├── .gitignore                                # Local secrets, caches, models, and build exclusions
├── .python-version                           # Python version selected by uv
├── AGENTS.md                                 # Repository conventions for coding agents
├── CONTRIBUTING.md                           # Contributor setup and development workflow
├── Makefile                                  # Standard install, check, test, and run commands
├── README.md                                 # Project setup, architecture, and file guide
├── pyproject.toml                            # Package metadata, dependencies, and tool configuration
├── uv.lock                                   # Exact direct and transitive dependency lock
├── src/
│   └── agentic_eval_toyapp/
│       ├── __init__.py                       # Package version
│       ├── agent.py                          # Bounded RAG agent loop
│       ├── app.py                            # FastAPI wiring, readiness, session, and message endpoints
│       ├── config.py                         # Validated service, provider, retrieval, agent, and session settings
│       ├── embeddings.py                     # Embedding interface, validation, and providers
│       ├── models.py                         # API, retrieval, agent, session, and provenance schemas
│       ├── providers.py                      # Scripted, extractive, and OpenAI-compatible providers
│       ├── prompts/
│       │   └── system.md                     # Versioned grounding and action instructions
│       ├── retrieval.py                      # Corpus loading and exact FAISS search
│       ├── resources/
│       │   └── documents.json                # Synthetic Northstar policy corpus
│       ├── sessions.py                       # Thread-safe bounded in-memory conversation state
│       ├── tracing.py                        # No-op and Langfuse turn-observation implementations
│       ├── static/
│       │   ├── app.js                        # Safe browser API client and result rendering
│       │   └── style.css                     # Responsive local browser styling
│       └── templates/
│           └── index.html                    # Accessible server-rendered chat page
└── tests/
    ├── integration/
    │   ├── test_agent_retrieval.py           # Offline agent-to-FAISS integration test
    │   ├── test_api.py                       # Offline session and chat API contract tests
    │   ├── test_api_live.py                  # Opt-in default semantic API smoke test
    │   ├── test_frontend.py                  # Browser page, asset, and rendering-safety tests
    │   ├── test_health_api.py                # FastAPI health and capability contract tests
    │   ├── test_langfuse_live.py             # Opt-in Langfuse export and retrieval test
    │   ├── test_model_live.py                # Opt-in live-provider connection and schema test
    │   └── test_retrieval_live.py            # Opt-in pinned-model semantic retrieval tests
    └── unit/
        ├── test_config.py                    # Configuration validation tests
        ├── test_embeddings.py                # Embedding validation and normalisation tests
        ├── test_agent.py                     # Agent flow, grounding, and limit tests
        ├── test_models.py                    # Pydantic contract and immutability tests
        ├── test_package.py                   # Package/distribution version consistency test
        ├── test_providers.py                 # Deterministic and live provider boundary tests
        ├── test_retrieval.py                 # Corpus integrity and deterministic FAISS tests
        ├── test_sessions.py                  # Session isolation, history, and truncation tests
        └── test_tracing.py                   # Offline trace shape and session-correlation tests
```

The local `design.md` is intentionally excluded because it is not committed.

## Additional MVP files

The full MVP architecture also includes these files:

```text
agentic-eval-toyapp/
├── .dockerignore                              # Docker build-context exclusions
├── Dockerfile                                 # Reproducible application container image
├── LICENSE                                    # Repository licence
├── compose.yaml                               # Local container configuration
├── src/agentic_eval_toyapp/
│   ├── scenarios.py                           # Controlled functional-case selection and mutations
│   └── resources/
│       └── cases.yaml                         # Opaque controlled-case runtime inputs
└── tests/
    ├── fixtures/
    │   ├── ground_truth.yaml                  # Evaluation-only expected labels
    │   └── model_actions/                     # Deterministic generation fixtures
    └── unit/
        └── test_scenarios.py                  # Control/defect isolation tests
```

## Retrieval design

Document chunks will be encoded with a pinned SentenceTransformer model and L2-normalised. They will be stored in an in-memory FAISS `IndexFlatIP` index. With normalised vectors, inner-product ranking is equivalent to cosine-similarity ranking.

The current model pin is:

```text
model: sentence-transformers/all-MiniLM-L6-v2
revision: 1110a243fdf4706b3f48f1d95db1a4f5529b4d41
dimension: 384
```

The validated `northstar-policies-v1` corpus contains 16 chunks from eight synthetic documents. Its canonical SHA-256 checksum is:

```text
fd79e97d578089ebef4b5c3bf7ada9ee4e5545bf72c66fbb8fb1abd2fa8682ee
```

The corpus and FAISS index are read-only during a run. Results are filtered by
the configured cosine threshold, ordered by descending score, and use chunk ID
as a deterministic tie-break. The model can request between one and eight
results per search; the configured `TOYAPP_RETRIEVAL_TOP_K` value is the
default used by the deterministic provider.

Offline CI uses fixed test embeddings and does not download the model. To run
the six natural-language retrieval checks and the default API smoke test
against the pinned model:

```bash
make test-live
```

The first live run downloads the model to the local Hugging Face cache. The retriever is a Python component designed for use by the agent and chat API.

To verify Platform AI credentials, connectivity, and structured-output support
with one real request:

```bash
make test-model-live
```

This test is excluded from offline CI because it requires credentials, network
access, and may incur provider cost. It validates the action schema rather than
exact response wording.

## Intended workflow

```text
Browser or METIS
       ↓
FastAPI session/message API
       ↓
Single bounded RAG agent
       ├── ask a clarification question
       ├── search the FAISS knowledge base
       └── return a cited answer or an observable zero-evidence response
       ↓
Langfuse session trace
       ↓
METIS trace retrieval and scoring
```
