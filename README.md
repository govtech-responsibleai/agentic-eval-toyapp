# Agentic Eval Toy App

A lightweight, single-agent RAG application for validating METIS metrics with controlled functional cases.

> **Status:** Stage 1 is complete. Stage 2 typed configuration and service-introspection API are implemented and awaiting manual verification. The RAG agent, FAISS retrieval, frontend, scenarios, and tracing are planned but not implemented.

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

## Current structure

```text
agentic-eval-toyapp/
├── .env.example
├── .github/
│   └── workflows/
│       └── ci.yml
├── .gitignore
├── .python-version
├── AGENTS.md
├── CONTRIBUTING.md
├── Makefile
├── README.md
├── pyproject.toml
├── uv.lock
├── src/
│   └── agentic_eval_toyapp/
│       ├── __init__.py
│       ├── app.py
│       ├── config.py
│       └── models.py
└── tests/
    ├── integration/
    │   └── test_health_api.py
    └── unit/
        ├── test_config.py
        ├── test_models.py
        └── test_package.py
```

The local `design.md` is intentionally excluded because it is not committed.

### Current file responsibilities

| File | Purpose |
|---|---|
| `pyproject.toml` | Defines package metadata, Hatchling builds, uv constraints, dev dependencies, and Ruff/Pyright/pytest configuration. |
| `uv.lock` | Locks exact package and development dependency versions. |
| `Makefile` | Provides the standard local and CI command interface, including lockfile validation. |
| `.python-version` | Selects Python 3.11 for uv and local development. |
| `.env.example` | Documents current non-secret service configuration. |
| `.gitignore` | Excludes secrets, environments, caches, generated data, and the local design file. |
| `.github/workflows/ci.yml` | Runs the offline `make ci` suite on future pushes and pull requests. |
| `AGENTS.md` | Records repository-specific implementation, testing, reporting, and Git conventions. |
| `CONTRIBUTING.md` | Explains setup, branching, code standards, tests, commits, and review expectations. |
| `README.md` | Describes the current scaffold, planned app, files, and commands. |
| `src/agentic_eval_toyapp/__init__.py` | Declares the installable package and static version. |
| `src/agentic_eval_toyapp/config.py` | Loads immutable, validated `TOYAPP_` service settings. |
| `src/agentic_eval_toyapp/models.py` | Defines strict health, readiness, capability, and error contracts. |
| `src/agentic_eval_toyapp/app.py` | Constructs FastAPI and exposes the initial service-introspection endpoints. |
| `tests/unit/test_package.py` | Verifies that package and distribution versions remain aligned. |
| `tests/unit/test_config.py` | Tests defaults, environment-variable overrides, and invalid service settings. |
| `tests/unit/test_models.py` | Tests strict API contracts and feature defaults. |
| `tests/integration/test_health_api.py` | Tests health, readiness, capability, and OpenAPI responses. |

## Planned MVP structure

The following files will be added in later, separately verified stages:

```text
agentic-eval-toyapp/
├── .dockerignore
├── Dockerfile
├── LICENSE
├── compose.yaml
├── src/agentic_eval_toyapp/
│   ├── agent.py
│   ├── embeddings.py
│   ├── providers.py
│   ├── retrieval.py
│   ├── scenarios.py
│   ├── sessions.py
│   ├── tracing.py
│   ├── prompts/
│   │   └── system.md
│   ├── resources/
│   │   ├── cases.yaml
│   │   └── documents.json
│   ├── static/
│   │   ├── app.js
│   │   └── style.css
│   └── templates/
│       └── index.html
└── tests/
    ├── fixtures/
    │   ├── ground_truth.yaml
    │   └── model_actions/
    ├── integration/
    │   ├── test_api.py
    │   └── test_langfuse_live.py
    └── unit/
        ├── test_agent.py
        ├── test_embeddings.py
        ├── test_retrieval.py
        ├── test_scenarios.py
        └── test_tracing.py
```

### Planned application files

| File | Purpose |
|---|---|
| `embeddings.py` | Provide the pinned SentenceTransformer embedder and deterministic fake. |
| `retrieval.py` | Validate the corpus, build FAISS `IndexFlatIP`, and return ranked chunks. |
| `providers.py` | Define the generation-model interface and real/fake implementations. |
| `agent.py` | Run the bounded `search`, `clarify`, or `answer` loop. |
| `sessions.py` | Store bounded in-memory conversation history and session metadata. |
| `scenarios.py` | Load controlled functional profiles and apply deliberate mutations. |
| `tracing.py` | Create Langfuse session/turn observations and test adapters. |
| `prompts/system.md` | Hold versioned grounding, citation, clarification, and constraint instructions. |
| `resources/documents.json` | Hold immutable synthetic Northstar Services policy chunks. |
| `resources/cases.yaml` | Hold opaque functional case IDs and public inputs. |
| `static/app.js` | Call the REST API and update the chat interface. |
| `static/style.css` | Style the minimal browser interface. |
| `templates/index.html` | Define the server-rendered chat page. |

### Planned deployment files

| File | Purpose |
|---|---|
| `.dockerignore` | Exclude unnecessary and sensitive files from Docker builds. |
| `Dockerfile` | Build the API, UI, and pinned embedding model into one image. |
| `compose.yaml` | Run the toy app locally with environment and health configuration. |
| `LICENSE` | State the repository licence after it is confirmed. |

### Planned test files

| File | Purpose |
|---|---|
| `tests/unit/test_embeddings.py` | Test embedding validation, normalisation, dimensions, and the fake embedder. |
| `tests/unit/test_retrieval.py` | Test FAISS ranking, thresholds, mapping, ties, and malformed corpus cases. |
| `tests/unit/test_agent.py` | Test action validation, bounded looping, citations, clarification, and termination. |
| `tests/unit/test_scenarios.py` | Test control/defect isolation and case-manifest integrity. |
| `tests/unit/test_tracing.py` | Test trace structure, metadata, redaction, and failure handling. |
| `tests/integration/test_api.py` | Test fake-provider sessions through FastAPI. |
| `tests/integration/test_langfuse_live.py` | Prove that METIS can retrieve an emitted Langfuse session trace. |
| `tests/fixtures/model_actions/` | Store deterministic fake-model action sequences. |
| `tests/fixtures/ground_truth.yaml` | Store evaluation-only labels that are never shown to the agent. |

## Retrieval design

Document chunks will be encoded with a pinned SentenceTransformer model and L2-normalised. They will be stored in an in-memory FAISS `IndexFlatIP` index. With normalised vectors, inner-product ranking is equivalent to cosine-similarity ranking.

The corpus and FAISS index will be read-only during a run. This keeps benchmark cases reproducible while allowing natural user phrasing.

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
