# Contributing to Agentic Eval Toy App

Thank you for contributing. This guide covers both GovTech team members and
external contributors. The application is an experimental evaluation target,
so changes should preserve deterministic offline verification, synthetic data,
and the narrow capability boundaries documented in `architecture.md`.

## Before you start

- Search existing issues and pull requests before proposing duplicate work.
- Open an issue before making a substantial behavioural or architectural
  change so its scope can be agreed.
- Use public issues only for non-sensitive bugs and feature requests.
- Report suspected vulnerabilities through [SECURITY.md](SECURITY.md), never a
  public issue.

## Prerequisites

- Python 3.11 or later
- [uv](https://docs.astral.sh/uv/)
- Make

## Setup

```bash
git clone https://github.com/govtech-responsibleai/agentic-eval-toyapp.git
cd agentic-eval-toyapp
make install
make ci
```

## Development workflow

### GovTech team members

Create a focused branch from the latest `main`, push it to the upstream
repository, and open a pull request targeting `main`.

### External contributors

Fork the repository, create a focused branch from the latest upstream `main`,
push it to your fork, and open a pull request from the fork back to upstream
`main`.

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
make ci
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

Live provider, embedding, and Langfuse tests are opt-in and remain excluded
from offline CI. Pull requests must not require external credentials or network
services for their required checks.

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
- Link the issue that defines or motivates the change.
- Describe behaviour and configuration changes.
- Include automated and manual verification instructions.
- Confirm `make ci` passes.
- Do not include secrets, `.env`, model caches, generated indexes, or the local `design.md`.
- Expect maintainer review; contributors must not merge until required review
  and CI checks pass.

Maintainers may use squash merging for a focused change. Delete the source
branch after merge when it is no longer needed.

## Getting help

- Usage and setup questions: read [README.md](README.md) and `make help` first.
- Architecture questions: see [architecture.md](architecture.md).
- Reproducible bugs or proposed enhancements: open a GitHub issue.
- Vulnerabilities: follow [SECURITY.md](SECURITY.md).
