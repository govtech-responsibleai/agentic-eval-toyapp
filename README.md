# Agentic Eval Toy App

A lightweight, single-agent RAG application for validating METIS metrics with controlled functional cases.

> **Status:** design phase. The application files below describe the planned MVP structure; implementation has not started.

## MVP summary

The app will:

- answer questions over a fixed synthetic policy corpus;
- use SentenceTransformer embeddings and an in-memory FAISS index for semantic retrieval;
- expose the same workflow through a FastAPI API and browser chat;
- record agent, model, and retrieval activity in Langfuse;
- return session identifiers that METIS can use to retrieve traces;
- provide paired control/defect cases for scorer regression tests;
- run locally or in Docker using a uv and Makefile workflow.

The MVP uses one bounded RAG agent. It does not use a leader or specialist agents.

## Planned project structure

```text
agentic-eval-toyapp/
├── .dockerignore
├── .env.example
├── .github/
│   └── workflows/
│       └── ci.yml
├── .gitignore
├── .python-version
├── AGENTS.md
├── CONTRIBUTING.md
├── Dockerfile
├── LICENSE
├── Makefile
├── README.md
├── compose.yaml
├── pyproject.toml
├── uv.lock
├── src/
│   └── agentic_eval_toyapp/
│       ├── __init__.py
│       ├── agent.py
│       ├── app.py
│       ├── config.py
│       ├── embeddings.py
│       ├── models.py
│       ├── providers.py
│       ├── retrieval.py
│       ├── scenarios.py
│       ├── sessions.py
│       ├── tracing.py
│       ├── prompts/
│       │   └── system.md
│       ├── resources/
│       │   ├── cases.yaml
│       │   └── documents.json
│       ├── static/
│       │   ├── app.js
│       │   └── style.css
│       └── templates/
│           └── index.html
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

## File responsibilities

### Project files

| File | Purpose |
|---|---|
| `pyproject.toml` | Defines the Python package, runtime dependencies, dev dependencies, and tool configuration. |
| `uv.lock` | Locks exact dependency versions for reproducible installation. |
| `Makefile` | Provides standard install, run, check, test, CI, and Docker commands. |
| `.python-version` | Declares the local Python version. |
| `.env.example` | Documents required model, Langfuse, service, and retrieval settings without secrets. |
| `.gitignore` | Excludes secrets, local environments, caches, artifacts, and the local design document. |
| `.dockerignore` | Excludes unnecessary and sensitive files from the Docker build context. |
| `Dockerfile` | Builds the API, web UI, and pinned embedding model into one runtime image. |
| `compose.yaml` | Runs the toy app locally with environment and health-check configuration. |
| `README.md` | Explains the app, its structure, and common commands. |
| `CONTRIBUTING.md` | Defines contribution, testing, and review expectations. |
| `AGENTS.md` | Provides repository-specific guidance for coding agents. |
| `LICENSE` | States the repository licence. |
| `.github/workflows/ci.yml` | Runs static checks, offline tests, and the Docker smoke test. |

### Application files

| File | Purpose |
|---|---|
| `src/agentic_eval_toyapp/__init__.py` | Declares the application package and version. |
| `app.py` | Constructs FastAPI, exposes health/session/message routes, and serves the browser UI. |
| `config.py` | Loads and validates typed environment configuration. |
| `models.py` | Defines Pydantic API contracts and internal action/result schemas. |
| `embeddings.py` | Provides the pinned SentenceTransformer embedder and deterministic fake test embedder. |
| `retrieval.py` | Validates the corpus, builds the FAISS `IndexFlatIP` index, and returns ranked chunks. |
| `providers.py` | Defines the model-provider interface plus real and fake implementations. |
| `agent.py` | Runs the bounded `search`, `clarify`, or `answer` loop for one turn. |
| `sessions.py` | Stores bounded in-memory conversation history and session metadata. |
| `scenarios.py` | Loads controlled functional profiles and applies one deliberate mutation at a named seam. |
| `tracing.py` | Creates Langfuse session/turn observations and test tracing adapters. |
| `prompts/system.md` | Contains the versioned grounding, citation, clarification, and constraint instructions. |
| `resources/documents.json` | Contains the fixed synthetic Northstar Services policy chunks. |
| `resources/cases.yaml` | Contains opaque functional case IDs and their public inputs. |
| `static/app.js` | Calls the REST API and updates the chat interface. |
| `static/style.css` | Styles the minimal browser interface. |
| `templates/index.html` | Defines the server-rendered chat page. |

### Test files

| File | Purpose |
|---|---|
| `tests/unit/test_embeddings.py` | Tests embedding validation, normalisation, dimensions, and the fake embedder. |
| `tests/unit/test_retrieval.py` | Tests FAISS ranking, thresholds, mapping, ties, and malformed corpus cases. |
| `tests/unit/test_agent.py` | Tests action validation, bounded looping, citations, clarification, and termination. |
| `tests/unit/test_scenarios.py` | Tests control/defect isolation and case-manifest integrity. |
| `tests/unit/test_tracing.py` | Tests trace structure, metadata, redaction, and failure handling. |
| `tests/integration/test_api.py` | Tests complete fake-provider sessions through the FastAPI API. |
| `tests/integration/test_langfuse_live.py` | Opt-in test proving that METIS can retrieve the emitted Langfuse session trace. |
| `tests/fixtures/model_actions/` | Stores deterministic fake-model action sequences. |
| `tests/fixtures/ground_truth.yaml` | Stores evaluation-only expected labels and rationales; it is never shown to the agent. |

## Retrieval design

Document chunks are encoded with a pinned SentenceTransformer model and L2-normalised. They are stored in an in-memory FAISS `IndexFlatIP` index. Because both query and document vectors are normalised, inner-product ranking is equivalent to cosine-similarity ranking.

The corpus and FAISS index are read-only during a run. This keeps benchmark cases reproducible while still allowing users to phrase questions naturally.

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

Implementation will begin only after the design and mock corpus are approved.

