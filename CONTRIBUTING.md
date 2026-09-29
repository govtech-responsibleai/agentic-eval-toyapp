# Contributing to Agentic Eval Toy App

## Prerequisites

- Python 3.11 or later
- [uv](https://docs.astral.sh/uv/)
- Make

## Setup

```bash
git clone <repository-url>
cd agentic-eval-toyapp
make install
make ci
```

The remote repository will be added later. Until then, use the same commands in the local repository.

## Development workflow

Create a focused branch from `main` using one of these prefixes:

| Prefix | Use |
|---|---|
| `feat/` | New functionality |
| `fix/` | Bug fixes |
| `docs/` | Documentation only |
| `refactor/` | Behaviour-preserving restructuring |
| `test/` | Test-only changes |
| `chore/` | Tooling and maintenance |

Before review, run:

```bash
make check
make test
```

`make check` is non-mutating. Use `make fix` to apply safe Ruff fixes and formatting.

## Code standards

- Keep imports ordered: standard library, third-party packages, local modules.
- Type public functions, classes, methods, and shared data structures.
- Use Pydantic v2 models for API and domain schemas.
- Isolate model, embedding, and tracing services behind interfaces.
- Keep offline tests deterministic and independent of external credentials.
- Add tests for empty values, malformed inputs, boundary values, shape mismatches, NaN/non-finite data, and relevant failure paths.
- Preserve ground-truth isolation: expected labels must not enter model prompts or target traces.

## Tests

```bash
make test
make test-unit
uv run pytest tests/unit/test_package.py
```

Live integration tests will be opt-in when they are introduced. They must not run as part of offline CI.

## Commit messages

Use Conventional Commits:

```text
<type>(<scope>): <imperative subject>
```

Examples:

```text
chore(toyapp): scaffold development workflow
feat(api): add session creation endpoint
fix(retrieval): reject non-finite embeddings
```

## Pull requests

- Keep the change focused on one concern.
- Describe behaviour and configuration changes.
- Include automated and manual verification instructions.
- Confirm `make ci` passes.
- Do not include secrets, `.env`, model caches, generated indexes, or the local `design.md`.
