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
| `make test-live` | Run opt-in checks that may download a model or call a future external service. |
| `make clean` | Remove generated Python and test caches. |

Start the current API:

```bash
make run
```

Then inspect:

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/ready
http://127.0.0.1:8000/api/v1/capabilities
http://127.0.0.1:8000/docs
```

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
│       ├── app.py                            # FastAPI application and service endpoints
│       ├── config.py                         # Validated service and retrieval settings
│       ├── embeddings.py                     # Embedding interface, validation, and providers
│       ├── models.py                         # API, corpus, retrieval, and provenance schemas
│       ├── retrieval.py                      # Corpus loading and exact FAISS search
│       └── resources/
│           └── documents.json                # Synthetic Northstar policy corpus
└── tests/
    ├── integration/
    │   ├── test_health_api.py                # FastAPI health and capability contract tests
    │   └── test_retrieval_live.py            # Opt-in pinned-model semantic retrieval tests
    └── unit/
        ├── test_config.py                    # Configuration validation tests
        ├── test_embeddings.py                # Embedding validation and normalisation tests
        ├── test_models.py                    # Pydantic contract and immutability tests
        ├── test_package.py                   # Package/distribution version consistency test
        └── test_retrieval.py                 # Corpus integrity and deterministic FAISS tests
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
│   ├── agent.py                               # Bounded RAG agent loop
│   ├── providers.py                           # Real and deterministic generation providers
│   ├── scenarios.py                           # Controlled functional-case selection and mutations
│   ├── sessions.py                            # Bounded in-memory conversation state
│   ├── tracing.py                             # Langfuse tracing abstraction and implementation
│   ├── prompts/
│   │   └── system.md                          # Versioned agent instructions
│   ├── resources/
│   │   └── cases.yaml                         # Opaque controlled-case runtime inputs
│   ├── static/
│   │   ├── app.js                             # Browser API client and chat interactions
│   │   └── style.css                          # Browser interface styling
│   └── templates/
│       └── index.html                         # Server-rendered chat page
└── tests/
    ├── fixtures/
    │   ├── ground_truth.yaml                  # Evaluation-only expected labels
    │   └── model_actions/                     # Deterministic generation fixtures
    ├── integration/
    │   ├── test_api.py                        # End-to-end fake-provider API tests
    │   └── test_langfuse_live.py              # Opt-in METIS trace-retrieval test
    └── unit/
        ├── test_agent.py                      # Agent actions, limits, and citation tests
        ├── test_scenarios.py                  # Control/defect isolation tests
        └── test_tracing.py                    # Trace shape, metadata, and redaction tests
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

The corpus and FAISS index are read-only during a run. Results are filtered by the configured cosine threshold, ordered by descending score, and use chunk ID as a deterministic tie-break.

Offline CI uses fixed test embeddings and does not download the model. To run the six natural-language retrieval checks against the pinned model:

```bash
make test-live
```

The first live run downloads the model to the local Hugging Face cache. The retriever is a Python component designed for use by the agent and chat API.

## Intended workflow

```text
Browser or METIS
       ↓
FastAPI session/message API
       ↓
Single bounded RAG agent
       ├── ask a clarification question
       ├── search the FAISS knowledge base
       └── return a cited Markdown answer
       ↓
Langfuse session trace
       ↓
METIS trace retrieval and scoring
```
