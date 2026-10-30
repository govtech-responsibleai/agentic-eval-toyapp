# Repository structure

The structure below is a snapshot and can drift; the source tree is authoritative.
The local `design.md`, `.env`, embedding caches, FAISS artefacts, and generated
`trace_cases/runs/` manifests are intentionally excluded from source control.

```text
agentic-eval-toyapp/
├── .dockerignore                             # Secrets, tests, caches, and local artefacts excluded from image builds
├── .env.example                              # Non-secret local configuration template
├── .github/
│   ├── CODEOWNERS                            # Default maintainer review ownership
│   ├── dependabot.yml                        # Automated dependency update configuration
│   └── workflows/
│       └── ci.yml                            # GitHub Actions offline CI workflow
├── .gitignore                                # Secrets, caches, builds, models, and run exclusions
├── .python-version                           # Python version selected by uv
├── AGENTS.md                                 # Repository conventions for coding agents
├── airbase.json                              # Airbase project, port, and instance configuration
├── CONTRIBUTING.md                           # Contributor setup and development workflow
├── Dockerfile                                # Multi-stage non-root runtime image with cached embeddings
├── LICENSE                                   # MIT licence terms
├── Makefile                                  # Stable install, run, verification, and trace commands
├── README.md                                 # Project overview and quick start
├── SECURITY.md                               # Private vulnerability disclosure process
├── architecture.md                           # Technical component architecture and data flows
├── compose.yaml                              # Local application and persistent Redis services
├── docs/
│   ├── airbase-deployment.md                 # Verified deployment and load-test evidence
│   ├── api.md                                # Session, upload, and message API guide
│   ├── configuration.md                      # Environment, Langfuse, and Redis settings
│   ├── deploy-airbase.md                     # Airbase staging and production procedure
│   ├── github-workflow.md                    # Internal and external collaboration playbook
│   ├── rate-limits.md                        # Backpressure settings and tuning
│   ├── repository-structure.md               # This file
│   ├── running.md                            # Local and Docker runs, endpoints, error logs
│   ├── trace-generation.md                   # Controlled trace generation guide
│   └── usage.md                              # Capabilities and browser usage
├── pyproject.toml                            # Package metadata, dependencies, and tool configuration
├── uv.lock                                   # Exact direct and transitive dependency lock
├── scripts/
│   ├── __init__.py                           # Build-tool package marker
│   ├── cache_embedding_model.py              # Image-build download and model-dimension validation
│   ├── docker_smoke.py                       # Container readiness and deterministic RAG smoke test
│   └── load_test.py                          # Concurrent deployed session/message verification
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
│       ├── agent.py                          # Bounded RAG loop and coordinator specialist adapter
│       ├── analysis.py                       # Report specialist, model adapters, and reference validation
│       ├── analysis_models.py                # Typed report-analysis requests and model actions
│       ├── app.py                            # FastAPI factory, dependency wiring, and routes
│       ├── config.py                         # Validated environment-backed settings
│       ├── coordinator.py                    # Bounded coordinator decisions and specialist dispatch
│       ├── diagnostics.py                    # Classified operational logs and per-turn correlation
│       ├── embeddings.py                     # Embedding interfaces, validation, and providers
│       ├── models.py                         # API, retrieval, agent, and session contracts
│       ├── multi_agent_tracing.py            # Coordinator-owned nested specialist trace context
│       ├── orchestration_models.py           # Coordinator actions, attachments, and handoff contracts
│       ├── providers.py                      # Live, deterministic, and scripted generation adapters
│       ├── prompts/
│       │   ├── analysis.md                   # Report grounding and findings instructions
│       │   ├── coordinator.md                # Coordinator routing and trust-boundary instructions
│       │   └── system.md                     # RAG role and action instructions
│       ├── retrieval.py                      # Corpus loading and exact FAISS search
│       ├── report_sources.py                 # Complete report text with stable page and section locators
│       ├── reports.py                        # Bounded in-memory and Redis report storage
│       ├── resources/
│       │   └── documents.json                # Synthetic Northstar policy corpus
│       ├── sessions.py                       # Event-loop-safe in-memory and Redis conversation stores
│       ├── static/
│       │   ├── app.js                        # Browser API client and sanitized Markdown rendering
│       │   ├── style.css                     # Responsive panel layout and browser styling
│       │   └── vendor/
│       │       ├── README.md                 # Pinned browser bundle versions and checksums
│       │       ├── LICENSE.dompurify         # DOMPurify Apache license text
│       │       ├── LICENSE.markdown-it       # markdown-it MIT license text
│       │       ├── markdown-it.umd.min.js    # Local Markdown parser for assistant replies
│       │       └── purify.min.js             # Local HTML sanitizer for rendered replies
│       ├── tracing.py                        # No-op and Langfuse standalone turn recorders
│       └── templates/
│           └── index.html                    # Accessible server-rendered chat page
└── tests/
    ├── fixtures/
    │   └── test_report.md                    # Synthetic report for analysis verification
    ├── integration/
    │   ├── test_agent_retrieval.py           # Offline agent-to-FAISS composition test
    │   ├── test_api.py                       # Session/message API contract and failure tests
    │   ├── test_api_live.py                  # Opt-in pinned-model API smoke test
    │   ├── test_container_contract.py        # Static image, context, and Compose safety contracts
    │   ├── test_diagnostics_api.py            # API-to-provider error correlation and redaction
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
        ├── test_analysis.py                  # Report decisions, grounding, limits, and provider tests
        ├── test_config.py                    # Defaults, environment aliases, and secret validation tests
        ├── test_coordinator.py               # Coordinator routing, limits, providers, and handoff tests
        ├── test_diagnostics.py               # Safe error classification, context reset, and redaction
        ├── test_build_scripts.py             # Container smoke validation and build-input edge cases
        ├── test_embeddings.py                # Embedding validation and normalization tests
        ├── test_models.py                    # Pydantic contract and consistency tests
        ├── test_orchestration_models.py      # Coordinator action and handoff schema tests
        ├── test_package.py                   # Package/distribution version consistency test
        ├── test_providers.py                 # Live and deterministic provider boundary tests
        ├── test_report_sources.py            # Complete report source and locator edge cases
        ├── test_retrieval.py                 # Corpus integrity and deterministic FAISS tests
        ├── test_reports.py                   # Report extraction, limits, hashing, and isolation tests
        ├── test_redis_reports.py             # Offline Redis report persistence and integrity tests
        ├── test_redis_sessions.py            # Offline Redis persistence and failure-boundary tests
        ├── test_sessions.py                  # Session isolation, history, and truncation tests
        ├── test_multi_agent_tracing.py       # Coordinator trace hierarchy, privacy, and usage tests
        └── test_tracing.py                   # Offline standalone trace shape and session-correlation tests
```
