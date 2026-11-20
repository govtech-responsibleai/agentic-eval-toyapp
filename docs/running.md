# Running the application

## Requirements

- Python 3.11 or later
- [uv](https://docs.astral.sh/uv/)
- Make
- Redis when persistent sessions are enabled outside Docker Compose

Alternatively, Docker with Compose can build and run the packaged service
without installing Python or uv on the host.

Live operation additionally requires:

- an API key and base URL for an OpenAI-compatible model provider;
- Langfuse credentials when trace export is enabled;
- network access for the first local embedding-model download if it is not
  cached, or during the Docker image build.

Clone the repository and run `make install`, then `make ci` to confirm the deterministic offline checks pass.

## Run locally

Start the default live-model coordinator application:

```bash
make run
```

In both RAG and coordinator modes, startup validates configuration, loads the
pinned embedding model, embeds the corpus, and builds the FAISS index before
the service accepts requests. The first local run downloads the embedding model
to the Hugging Face cache if it is not already present. `/ready` becomes
available after the selected top-level agent and its required dependencies are
initialised.

To run without a live generation request or model-provider credentials, select
the deterministic provider explicitly:

```bash
TOYAPP_GENERATION_PROVIDER=deterministic make run
```

This still uses the real pinned embedding model and FAISS retriever.

To run the standalone RAG agent, select it explicitly:

```bash
TOYAPP_AGENT_MODE=rag \
TOYAPP_GENERATION_PROVIDER=deterministic \
TOYAPP_LANGFUSE_ENABLED=false \
make run
```

Coordinator mode accepts the same session and message API. It can answer or
clarify directly and records public coordinator and RAG actions in
`execution_steps`. A `route_rag` decision delegates one focused policy task to
the existing FAISS-backed RAG agent; a `route_analysis` decision sends selected
report content to the analysis specialist. The specialist receives the full
extracted report in one model call, using a separate configurable 16,384-token
output allowance by default. The result's `report_references` identify exact
report pages or sections.
When Langfuse is enabled, coordinator turns export a nested trace. The
coordinator exposes a browser file panel for uploaded reports.

Open these local endpoints:

```text
Browser chat:      http://127.0.0.1:8000/
Health:            http://127.0.0.1:8000/health
Readiness:         http://127.0.0.1:8000/ready
Capabilities:      http://127.0.0.1:8000/api/v1/capabilities
OpenAPI interface: http://127.0.0.1:8000/docs
```

## Run with Docker

The image installs the locked runtime dependencies and downloads the exact
embedding-model revision during the build. Runtime model loading is local-only,
so container startup does not contact Hugging Face.

After creating `.env`, build and start the default live-model service:

```bash
make docker-up
```

For a local run without Platform AI credentials or model-call cost:

```bash
TOYAPP_GENERATION_PROVIDER=deterministic \
TOYAPP_LANGFUSE_ENABLED=false \
make docker-up
```

The same endpoints are available at `http://127.0.0.1:8000`. Compose also starts
Redis for persistent conversation and report storage. Inspect or stop the
services with:

```bash
make docker-logs
make docker-down
```

The startup log states the selected top-level agent, generation provider, and
whether Langfuse is configured. `/ready` and `/api/v1/capabilities` report
whether trace export is operational for that agent mode.

The first image build needs network access for locked Python packages, the base
images, and the pinned embedding model. Platform AI and Langfuse remain external
services; their credentials are passed from `.env` only when the container
starts and are not copied into the image.

The image uses the CPU-only PyTorch wheel source and does not include CUDA
libraries.

### Behind a TLS-intercepting proxy

A proxy that intercepts TLS, such as Cloudflare Gateway, makes the build fail
with `CERTIFICATE_VERIFY_FAILED` while it downloads packages or the embedding
model. Live model calls from the running container fail the same way. Both
targets read a CA bundle from `CA_BUNDLE`, which defaults to the host's
`SSL_CERT_FILE`. If your shell exports `SSL_CERT_FILE`, you don't need to pass
anything:

```bash
make docker-build   # the builder trusts CA_BUNDLE through a BuildKit secret
make docker-up      # compose.ca.yaml mounts CA_BUNDLE read-only at /run/toyapp-ca.pem
```

To check that the default was picked up, look for the bundle path in the dry
run. If this prints nothing, `SSL_CERT_FILE` isn't exported in this shell:

```bash
make -n docker-build | grep -o 'src="[^"]*"'
```

To use another bundle, or to run the container with only the trust that Airbase
has, set `CA_BUNDLE` yourself:

```bash
make docker-up CA_BUNDLE=/absolute/path/to/combined-ca-bundle.pem
make docker-up CA_BUNDLE=    # no proxy CA for the build or the run
```

The bundle must be a combined one: the public root certificates plus the
proxy's root CA. It replaces the trust store wherever it is used, so a file
that holds only the proxy's CA breaks every connection that the proxy doesn't
intercept. Supply public CA certificates only; do not include private keys.

The bundle is never in the image. The builder stage uses it and is not shipped,
and the runtime image trusts only the Debian system bundle at
`/etc/ssl/certs/ca-certificates.crt`. An image built behind the proxy therefore
carries no proxy CA to Airbase. TLS verification stays on throughout.

## Operational error logs

Failed provider calls, invalid model actions, upload validation, specialist
delegation, retrieval, and trace-export failures produce classified events in
the `make run` terminal or `make docker-logs` output. Search for
`toyapp_failure`. A message attempt has a `session_id` and a separate internal
`turn_id` so multiple failures in one session can be distinguished. Uploads
have a session ID but no turn ID because upload does not invoke the agent.
Events include the agent, operation, safe error code, and relevant model,
attempt, output budget, duration, or HTTP status. Expected upload rejections
and grounding findings are warnings; failures are errors. These diagnostics
are not shown in the browser or written to a separate file. Raw exception
messages, prompts, report contents, and credentials are intentionally omitted.
For a provider response that could not be used, the same event also records
`finish_reason`, `input_tokens`, `output_tokens`, `reasoning_tokens`, and
`total_tokens` when supplied by the gateway; missing values appear as `-`.
`max_tokens` is the configured output cap, not the number used. A
`finish_reason=length` reports that generation stopped at a limit; token
counts are provider-reported usage, not a local estimate. Transport failures
may have no response or usage fields. The configured 90-second timeout applies
to each live model call, not the entire coordinator-led turn; successful calls
do not emit usage in these operational error logs.
