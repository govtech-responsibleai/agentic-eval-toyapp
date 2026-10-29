# Agentic Eval Toy App

A lightweight multi-agent application for demonstrating and regression-testing
METIS agent evaluation metrics.

> [!WARNING]
> **This is an experimental evaluation target, not a production-ready
> application.** Use synthetic or non-sensitive data only. The project has no
> availability, durability, or support SLA and must not be used to make or
> automate operational decisions.

## What it does

The app answers questions about a synthetic employee-policy corpus and analyses
uploaded reports. A coordinator decides whether to answer directly, ask for
clarification, or hand the task to a specialist. Every turn can be exported to
Langfuse so METIS can retrieve and score it, and repository tooling can generate
control/defect trace pairs with known expected metric outcomes.

```text
            Browser chat  ·  REST API (/api/v1)
                          │
                          ▼
                   ┌─────────────┐
                   │ Coordinator │── answer / clarify directly
                   └──────┬──────┘
          route_rag │             │ route_analysis
                    ▼             ▼
          ┌──────────────┐  ┌─────────────────┐
          │  Policy RAG  │  │ Report analysis │
          │ (embeddings  │  │ (.txt .md .pdf) │
          │  + FAISS)    │  └────────┬────────┘
          └──────┬───────┘           │
                 ▼                   ▼
         policy citations     report_references
                          │
                          ▼
          Langfuse traces  ──▶  METIS evaluations
```

## Quick start

You need Python 3.11+, [uv](https://docs.astral.sh/uv/), and Make (or Docker
with Compose; see [Running the application](docs/running.md)).

```bash
make install
TOYAPP_GENERATION_PROVIDER=deterministic TOYAPP_LANGFUSE_ENABLED=false make run
```

Then open <http://127.0.0.1:8000/>. This uses a deterministic model stand-in, so
it needs no API keys and costs nothing. The first run downloads the embedding
model. For a live model and Langfuse tracing, see
[Configuration](docs/configuration.md).

## Documentation

| Guide | Contents |
|---|---|
| [Capabilities and browser usage](docs/usage.md) | Full feature list, chat page, report upload |
| [API](docs/api.md) | Session, upload, and message routes with `curl` examples |
| [Configuration](docs/configuration.md) | `.env`, live model, Langfuse, Redis persistence |
| [Running the application](docs/running.md) | Local and Docker runs, endpoints, error logs |
| [Model-call limits](docs/rate-limits.md) | Backpressure settings and tuning |
| [Deploying to Airbase](docs/deploy-airbase.md) | Staging and production procedure |
| [Controlled trace generation](docs/trace-generation.md) | Control/defect trace pairs |
| [Repository structure](docs/repository-structure.md) | File-by-file map |
| [Architecture](architecture.md) | Components, data flows, limitations |
| [Contributing](CONTRIBUTING.md) | Setup, `make` targets, workflow |

## Support, security, and license

Support is best-effort. Report reproducible bugs and feature requests as
[GitHub issues](https://github.com/govtech-responsibleai/agentic-eval-toyapp/issues).
Do not report vulnerabilities publicly; follow [SECURITY.md](SECURITY.md).
Released under the [MIT License](LICENSE).
