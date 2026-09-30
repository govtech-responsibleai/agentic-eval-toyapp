.PHONY: help install run lock-check lint lint-fix format format-check fix typecheck check test test-unit test-live ci clean

PYTHON_SOURCES := src tests

help:
	@printf "Available targets:\n"
	@printf "  make install       Install the package and development dependencies\n"
	@printf "  make run           Start the local development API\n"
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
	@printf "  make ci            Run the complete offline CI suite\n"
	@printf "  make clean         Remove generated Python and test caches\n"

install:
	uv sync --group dev

run:
	uv run uvicorn agentic_eval_toyapp.app:app \
		--host "$${TOYAPP_HOST:-127.0.0.1}" \
		--port "$${TOYAPP_PORT:-8000}" \
		--reload

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

ci:
	$(MAKE) lock-check
	$(MAKE) check
	$(MAKE) test

clean:
	find src tests -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .pyright .coverage htmlcov
