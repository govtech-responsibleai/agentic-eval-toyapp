# syntax=docker/dockerfile:1.7

FROM python:3.11-slim-bookworm AS python-base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*


FROM ghcr.io/astral-sh/uv:0.12.7 AS uv


FROM python-base AS builder

ENV UV_LINK_MODE=copy \
    UV_SYSTEM_CERTS=true

WORKDIR /app

COPY --from=uv /uv /uvx /bin/

RUN --mount=type=secret,id=toyapp_ca_bundle \
    if [ -f /run/secrets/toyapp_ca_bundle ]; then \
        cp /run/secrets/toyapp_ca_bundle /opt/toyapp-ca-bundle.pem; \
    else \
        cp /etc/ssl/certs/ca-certificates.crt /opt/toyapp-ca-bundle.pem; \
    fi \
    && chmod 0644 /opt/toyapp-ca-bundle.pem

ENV SSL_CERT_FILE=/opt/toyapp-ca-bundle.pem \
    REQUESTS_CA_BUNDLE=/opt/toyapp-ca-bundle.pem

COPY pyproject.toml uv.lock .python-version ./

RUN uv sync --frozen --no-dev --no-install-project

COPY README.md ./
COPY src/ ./src/

RUN uv sync --frozen --no-dev --no-editable

ARG EMBEDDING_MODEL_ID=sentence-transformers/all-MiniLM-L6-v2
ARG EMBEDDING_MODEL_REVISION=1110a243fdf4706b3f48f1d95db1a4f5529b4d41

COPY scripts/cache_embedding_model.py /tmp/cache_embedding_model.py

RUN /app/.venv/bin/python /tmp/cache_embedding_model.py \
        --model-id "${EMBEDDING_MODEL_ID}" \
        --revision "${EMBEDDING_MODEL_REVISION}" \
        --cache-dir /opt/models \
    && rm /tmp/cache_embedding_model.py


FROM python-base AS runtime

ENV PATH="/app/.venv/bin:${PATH}" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    SSL_CERT_FILE=/opt/toyapp-ca-bundle.pem \
    REQUESTS_CA_BUNDLE=/opt/toyapp-ca-bundle.pem \
    PORT=8000 \
    TOYAPP_HOST=0.0.0.0 \
    TOYAPP_PORT=8000 \
    TOYAPP_EMBEDDING_CACHE_DIR=/opt/models \
    TOYAPP_EMBEDDING_LOCAL_FILES_ONLY=true

WORKDIR /app

RUN groupadd --gid 999 app \
    && useradd --uid 999 --gid app --home-dir /nonexistent --shell /usr/sbin/nologin app

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /opt/models /opt/models
COPY --from=builder --chown=app:app /opt/toyapp-ca-bundle.pem /opt/toyapp-ca-bundle.pem

USER app

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=90s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; port = os.environ.get('PORT', '8000'); urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2).read()"]

CMD ["sh", "-c", "exec uvicorn agentic_eval_toyapp.app:app --host 0.0.0.0 --port \"${PORT:-8000}\""]
