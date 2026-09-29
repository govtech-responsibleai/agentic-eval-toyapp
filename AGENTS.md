# Agentic Eval Toy App — Codebase Guide

## Purpose

This repository contains a small, controlled RAG application used as a target for METIS evaluations. It prioritises reproducibility, explicit functional ground truth, trace completeness, and a narrow capability surface.

## Technology

- Python 3.11 or later
- uv for dependency and environment management
- Hatchling as the build backend
- Pydantic v2 for shared contracts when schemas are introduced
- FastAPI for the future API and browser service
- SentenceTransformers and FAISS for future semantic retrieval
- Langfuse for future tracing
- Ruff, Pyright, and pytest for development checks

## Commands

```bash
make install
make check
make fix
make test
make ci
```

Run `make ci` before requesting review.

## Implementation workflow

Before implementing a stage:

1. explain the relevant data, retrieval, metric, or transformation logic;
2. provide a concise plan listing files, interfaces, dependencies, schemas, and tests;
3. keep the work within the current unchecked stage in `design.md`.

After implementing a stage, report:

1. the outcome and technical or statistical logic;
2. every changed file and its purpose;
3. dependency, configuration, API, and schema effects;
4. tests, edge cases, commands, and observed results;
5. exact manual verification steps;
6. limitations, deferred work, and open decisions.

Do not mark a stage complete in `design.md` until the user explicitly confirms successful manual verification.

## Code conventions

- Use a `src/` package layout and absolute imports from `agentic_eval_toyapp`.
- Order imports as standard library, third-party packages, then local modules.
- Type all public interfaces and keep Pyright in strict mode.
- Use Pydantic models for API and shared domain contracts.
- Keep external providers behind narrow interfaces with deterministic fakes for tests.
- Keep the document corpus immutable during evaluation runs.
- Do not expose ground-truth labels or mutation descriptions to the model or traces.
- Do not record private chain-of-thought.
- Use structured operational logging rather than `print()` in application code.
- Add edge-case tests for every data transformation, retrieval function, and metric-related behaviour.

## Testing boundaries

- Offline CI must not require model, embedding, Langfuse, or other network access.
- Unit tests use deterministic fakes and fixtures.
- Live provider and Langfuse tests must be explicitly selected and skipped by default.
- Do not assert exact wording from a stochastic live model.
- Scenario tests must verify that a control/defect pair differs at exactly one named seam.

## Git conventions

Use Conventional Commits with a concise scope, for example:

```text
chore(toyapp): scaffold development workflow
feat(retrieval): add FAISS semantic search
test(scenarios): cover grounding control pair
```

Keep unrelated changes in separate commits. Never commit `.env`, credentials, model caches, generated FAISS indexes, or the local `design.md` file.
