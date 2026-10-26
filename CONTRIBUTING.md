# Contributing to Agentic Eval Toy App

Thank you for contributing to Agentic Eval Toy App. This guide covers
expectations and best practices whether you are on the core team or
contributing externally.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Repository Orientation](#repository-orientation)
- [Development Workflow](#development-workflow)
- [Code Standards](#code-standards)
- [Testing](#testing)
- [Pull Requests](#pull-requests)
- [Getting Help](#getting-help)

## Prerequisites

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) (package and environment manager)
- Make
- [direnv](https://direnv.net/) (optional — loads `.env` and activates the
  virtual environment)
- Docker with Compose (optional — required for container verification)

## Getting Started

```bash
git clone https://github.com/govtech-responsibleai/agentic-eval-toyapp.git
cd agentic-eval-toyapp
make install    # install the package and development tools
make check      # lint, format-check and type-check
```

If you use direnv, review `.envrc` and run `direnv allow`. Otherwise, prefix
commands with `uv run` or activate `.venv` manually. Copy `.env.example` to
`.env` only when local runtime configuration is needed; never commit `.env`.

## Repository Orientation

```text
agentic-eval-toyapp/
├── src/agentic_eval_toyapp/   # FastAPI service, agents and browser assets
├── tests/                      # Unit, integration and opt-in live tests
├── trace_cases/                # Controlled evaluation trace generation
├── scripts/                    # Docker build and smoke-test helpers
├── architecture.md             # Component contracts and data flows
├── Dockerfile                  # Non-root runtime image
└── pyproject.toml              # Package, dependency and tool configuration
```

The default coordinator can answer or clarify directly and can delegate to a
policy RAG specialist or a selected-report analysis specialist. Sessions and
reports are bounded and process-local. See [architecture.md](architecture.md)
for the full specification and trust boundaries.

## Development Workflow

For the step-by-step GitHub collaboration playbook, see
[docs/github-workflow.md](docs/github-workflow.md).

### Branching

Branch from the latest `main` with a descriptive prefix:

| Prefix | Use |
|--------|-----|
| `feat/` | New functionality |
| `fix/` | Bug fixes |
| `docs/` | Documentation only |
| `refactor/` | Behaviour-preserving restructuring |
| `test/` | Adding or improving tests |
| `chore/` | Tooling and maintenance |

Examples: `feat/session-resumption`, `fix/static-asset-urls`.

### Running Checks

```bash
make check          # lint + format-check + typecheck (non-mutating)
make fix            # auto-fix lint and formatting
make ci             # lock check + static checks + offline tests
```

Run `make ci` before requesting review. Required CI must remain deterministic,
network-free and independent of model, embedding and tracing services.

### Commit Messages

Use Conventional Commits in this format:

```text
{type}({scope}): {subject}

{body}

{footer}
```

Rules:

- `{type}` is one of `feat`, `fix`, `docs`, `style`, `refactor`, `test` or
  `chore`.
- `{scope}` identifies the affected area, such as `toyapp`, `api`, `web`,
  `retrieval` or `trace-cases`.
- Write the subject in imperative present tense with no trailing full stop.
- Explain motivation and behaviour changes in the body when they are not
  obvious.

Examples:

```text
feat(api): add session creation endpoint

fix(web): use relative static asset URLs

chore(toyapp): prepare open-source governance
```

Avoid generic messages such as `fix bug` or `update code`.

## Code Standards

| Concern | Tool | Notes |
|---------|------|-------|
| Linting | Ruff | Rules are configured in `pyproject.toml` |
| Formatting | Ruff | Use the repository formatter |
| Type checking | Pyright | Strict mode; public interfaces are typed |
| Data models | Pydantic v2 | API and shared domain contracts |
| Web service | FastAPI | Versioned API and browser routes |

### Conventions

- Use the `src/` package layout and absolute package imports.
- Keep external model, embedding and tracing providers behind narrow
  interfaces.
- Use deterministic fakes and fixtures in offline tests.
- Keep the synthetic corpus immutable during evaluation runs.
- Never expose ground-truth labels or mutation descriptions to the target model
  or scorer-facing traces.
- Never record private chain-of-thought.
- Use structured operational logging and redact credentials, prompts and report
  contents from failures.

## Testing

```bash
make ci
make test
make test-unit
make docker-smoke    # requires Docker and build-time network access
```

### Where Tests Go

| Type | Location | Scope |
|------|----------|-------|
| Unit | `tests/unit/` | One transformation, boundary or component |
| Integration | `tests/integration/` | API, browser, retrieval and container contracts |
| Trace cases | `tests/trace_cases/` | Controlled trace-generation behaviour |
| Fixtures | `tests/fixtures/` | Shared synthetic input data |

### Guidelines

- Unit tests must not call external model, embedding, dataset or tracing
  services.
- New functionality should include tests; bug fixes should include a regression
  test.
- Add edge cases for malformed inputs, empty values, limits and provider
  failures.
- Live provider and Langfuse tests must be explicitly selected and are skipped
  by default.
- Do not assert exact prose from a stochastic live model.

## Pull Requests

### Before Opening

- [ ] `make ci` passes locally
- [ ] New behaviour has tests
- [ ] Manual verification steps are included where applicable
- [ ] No unrelated changes are bundled
- [ ] No secrets, `.env`, model caches, generated indexes, local design notes
      or trace-run artefacts are included

### What Reviewers Look For

- Preserved deterministic and ground-truth isolation boundaries
- Typed public interfaces and validated shared models
- Safe handling of uploaded content, credentials and operational errors
- Focused scope and proportionate test coverage
- Documented configuration, API or schema changes

### Merging

Pull requests require code-owner review and passing CI. Squash-merge is
preferred for focused branches; merge commits are acceptable when meaningful
intermediate history must be retained.

## Getting Help

- Application setup and usage -> [README.md](README.md)
- Architecture and trust boundaries -> [architecture.md](architecture.md)
- Build commands -> `make help`
- Bugs and feature requests -> GitHub issues
- Vulnerabilities -> [SECURITY.md](SECURITY.md)

If something in this guide is unclear or outdated, open an issue or fix it
directly — documentation contributions are contributions too.
