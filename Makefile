.PHONY: help install run trace-cases-list trace-case lock-check lint lint-fix format format-check fix typecheck check test test-unit test-live test-model-live test-langfuse-live ci clean

PYTHON_SOURCES := src tests trace_cases
TRACE_METRIC_ARGUMENT = $(if $(strip $(METRIC)),--metric "$(METRIC)",)
TRACE_VARIANT_ARGUMENT = $(if $(strip $(VARIANT)),--variant "$(VARIANT)",)

help:
	@printf "Available targets:\n"
	@printf "  make install       Install the package and development dependencies\n"
	@printf "  make run           Start the local development API\n"
	@printf "  make trace-cases-list  List supported controlled trace metrics\n"
	@printf "  make trace-case [METRIC=...] [VARIANT=...]  Generate controlled traces; omitted filters select all\n"
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
	uv run pytest -m langfuse_live

ci:
	$(MAKE) lock-check
	$(MAKE) check
	$(MAKE) test

clean:
	find src tests trace_cases -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .pyright .coverage htmlcov
