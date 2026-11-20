.PHONY: help install run trace-cases-list trace-case docker-build docker-up docker-down docker-logs docker-smoke docker-smoke-run load-test lock-check lint lint-fix format format-check fix typecheck check test test-unit test-live test-model-live test-langfuse-live ci clean

PYTHON_SOURCES := src tests trace_cases scripts
# The image that docker-build makes and docker-up and docker-smoke-run start. The default names the commit.
IMAGE_TAG ?= agentic-eval-toyapp:$(shell git rev-parse --short HEAD 2>/dev/null || echo local)
# The platform to build for. Leave it empty for this computer; use linux/amd64 for Airbase.
PLATFORM ?=
# A CA bundle for a proxy that intercepts TLS. docker-build reads it as a BuildKit secret and
# docker-up mounts it, so the runtime image never holds it.
CA_BUNDLE ?= $(SSL_CERT_FILE)
DOCKER_SMOKE_PORT ?= 18000
DOCKER_SMOKE_TIMEOUT ?= 180
LOAD_TEST_URL ?=
LOAD_TEST_USERS ?= 60
LOAD_TEST_TIMEOUT ?= 60
COMMA := ,
CA_SECRET_ARGUMENT = $(if $(strip $(CA_BUNDLE)),--secret id=toyapp_ca_bundle$(COMMA)src="$(CA_BUNDLE)",)
PLATFORM_ARGUMENT = $(if $(strip $(PLATFORM)),--platform "$(PLATFORM)",)
# Compose takes the image name and the CA bundle from these variables, not from Make.
COMPOSE_ENVIRONMENT = IMAGE_TAG="$(IMAGE_TAG)" $(if $(strip $(CA_BUNDLE)),CA_BUNDLE="$(abspath $(CA_BUNDLE))",)
COMPOSE_FILE_ARGUMENTS = $(if $(strip $(CA_BUNDLE)),-f compose.yaml -f compose.ca.yaml,)
DOCKER_COMPOSE = $(COMPOSE_ENVIRONMENT) docker compose $(COMPOSE_FILE_ARGUMENTS)
TRACE_METRIC_ARGUMENT = $(if $(strip $(METRIC)),--metric "$(METRIC)",)
TRACE_VARIANT_ARGUMENT = $(if $(strip $(VARIANT)),--variant "$(VARIANT)",)

help:
	@printf "Available targets:\n"
	@printf "  make install       Install the package and development dependencies\n"
	@printf "  make run           Start the local development API\n"
	@printf "  make trace-cases-list  List supported controlled trace metrics\n"
	@printf "  make trace-case [METRIC=...] [VARIANT=...]  Generate controlled traces; omitted filters select all\n"
	@printf "  make docker-build  Build IMAGE_TAG (default agentic-eval-toyapp:<commit>); PLATFORM=linux/amd64 for Airbase\n"
	@printf "  make docker-up     Build and start IMAGE_TAG with Compose; CA_BUNDLE (default SSL_CERT_FILE) is mounted, not built in\n"
	@printf "  make docker-down   Stop and remove the Compose service\n"
	@printf "  make docker-logs   Follow Compose service logs\n"
	@printf "  make docker-smoke  Build IMAGE_TAG and verify RAG, coordinator, browser assets, and restart\n"
	@printf "  make docker-smoke-run  Verify an existing IMAGE_TAG without building it\n"
	@printf "  make load-test LOAD_TEST_URL=https://... [LOAD_TEST_USERS=60]  Run concurrent session/message flows\n"
	@printf "  make lock-check    Check that the uv lockfile is current\n"
	@printf "  make lint          Run Ruff lint checks\n"
	@printf "  make lint-fix      Apply safe Ruff lint fixes\n"
	@printf "  make format        Format Python files with Ruff\n"
	@printf "  make format-check  Check formatting without modifying files\n"
	@printf "  make fix           Apply lint fixes and formatting\n"
	@printf "  make typecheck     Run strict Pyright type checking\n"
	@printf "  make check         Run lint, format-check, and type checking\n"
	@printf "  make test          Run all offline tests\n"
	@printf "  make test-unit     Run unit tests only\n"
	@printf "  make test-live     Run opt-in tests that may download models or call services\n"
	@printf "  make test-model-live  Test the configured live model provider\n"
	@printf "  make test-langfuse-live  Test trace export against configured Langfuse\n"
	@printf "  make ci            Run the complete offline CI suite\n"
	@printf "  make clean         Remove generated Python and test caches\n"

install:
	uv sync --group dev

run:
	uv run uvicorn agentic_eval_toyapp.app:app \
		--host "$${TOYAPP_HOST:-127.0.0.1}" \
		--port "$${TOYAPP_PORT:-8000}" \
		--reload

trace-cases-list:
	uv run python -m trace_cases list

trace-case:
	uv run python -m trace_cases generate $(TRACE_METRIC_ARGUMENT) $(TRACE_VARIANT_ARGUMENT)

docker-build:
	docker build $(PLATFORM_ARGUMENT) $(CA_SECRET_ARGUMENT) --tag "$(IMAGE_TAG)" .

docker-up: docker-build
	@test -z "$(strip $(CA_BUNDLE))" || test -f "$(CA_BUNDLE)" || { printf "CA_BUNDLE %s not found\n" "$(CA_BUNDLE)" >&2; exit 2; }
	$(DOCKER_COMPOSE) up --detach --no-build

docker-down:
	$(DOCKER_COMPOSE) down

docker-logs:
	$(DOCKER_COMPOSE) logs --follow toyapp

docker-smoke: docker-build docker-smoke-run

docker-smoke-run:
	python3 -m scripts.docker_smoke \
		--image "$(IMAGE_TAG)" \
		--host-port "$(DOCKER_SMOKE_PORT)" \
		--timeout-seconds "$(DOCKER_SMOKE_TIMEOUT)"

load-test:
	@test -n "$(LOAD_TEST_URL)" || (printf "LOAD_TEST_URL is required\n" >&2; exit 2)
	uv run python -m scripts.load_test \
		--base-url "$(LOAD_TEST_URL)" \
		--users "$(LOAD_TEST_USERS)" \
		--timeout-seconds "$(LOAD_TEST_TIMEOUT)"

lock-check:
	uv lock --check

lint:
	uv run ruff check $(PYTHON_SOURCES)

lint-fix:
	uv run ruff check $(PYTHON_SOURCES) --fix

format:
	uv run ruff format $(PYTHON_SOURCES)

format-check:
	uv run ruff format $(PYTHON_SOURCES) --check

fix:
	$(MAKE) lint-fix
	$(MAKE) format

typecheck:
	uv run pyright

check:
	$(MAKE) lint
	$(MAKE) format-check
	$(MAKE) typecheck

test:
	uv run pytest

test-unit:
	uv run pytest tests/unit/

test-live:
	uv run pytest -m live

test-model-live:
	uv run pytest -m model_live

test-langfuse-live:
	uv run pytest -m langfuse_live tests/integration/test_langfuse_live.py::test_api_turn_is_retrievable_by_app_session_id
	uv run pytest -m langfuse_live tests/integration/test_langfuse_live.py::test_coordinator_turn_is_one_retrievable_root_with_nested_rag

ci:
	$(MAKE) lock-check
	$(MAKE) check
	$(MAKE) test

clean:
	find src tests trace_cases -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .pyright .coverage htmlcov
