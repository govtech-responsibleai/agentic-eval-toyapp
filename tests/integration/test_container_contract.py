"""Static contracts for reproducible and secret-safe container packaging."""

from pathlib import Path

from agentic_eval_toyapp.config import AppSettings

REPOSITORY_ROOT = Path(__file__).parents[2]


def test_dockerfile_pins_and_caches_the_configured_embedding_model() -> None:
    """Image construction and application defaults must use one model revision."""
    dockerfile = (REPOSITORY_ROOT / "Dockerfile").read_text(encoding="utf-8")
    settings = AppSettings()

    assert f"ARG EMBEDDING_MODEL_ID={settings.embedding_model_id}" in dockerfile
    assert (
        f"ARG EMBEDDING_MODEL_REVISION={settings.embedding_model_revision}"
        in dockerfile
    )
    assert "TOYAPP_EMBEDDING_LOCAL_FILES_ONLY=true" in dockerfile
    assert "HF_HUB_OFFLINE=1" in dockerfile
    assert "REQUESTS_CA_BUNDLE=/opt/toyapp-ca-bundle.pem" in dockerfile
    assert "--mount=type=secret,id=toyapp_ca_bundle" in dockerfile
    assert "chmod 0644 /opt/toyapp-ca-bundle.pem" in dockerfile
    assert "UV_COMPILE_BYTECODE" not in dockerfile
    assert "USER toyapp" in dockerfile
    assert "HEALTHCHECK" in dockerfile


def test_docker_context_excludes_secrets_tests_and_private_runs() -> None:
    """Credentials and local evaluation ground truth must not enter the image context."""
    entries = {
        line.strip()
        for line in (REPOSITORY_ROOT / ".dockerignore")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert ".env" in entries
    assert ".git" in entries
    assert "tests" in entries
    assert "trace_cases" in entries
    assert "design.md" in entries


def test_compose_keeps_credentials_at_runtime_and_model_local() -> None:
    """Compose must inject `.env` at runtime rather than adding it to the image."""
    compose = (REPOSITORY_ROOT / "compose.yaml").read_text(encoding="utf-8")

    assert "env_file:" in compose
    assert "- .env" in compose
    assert "TOYAPP_LANGFUSE_ENABLED: ${TOYAPP_LANGFUSE_ENABLED:-false}" in compose
    assert 'TOYAPP_EMBEDDING_LOCAL_FILES_ONLY: "true"' in compose
    assert "no-new-privileges:true" in compose


def test_pyproject_selects_cpu_only_pytorch_index() -> None:
    """The CPU-only target must not resolve CUDA runtime packages."""
    pyproject = (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'torch = { index = "pytorch-cpu" }' in pyproject
    assert 'url = "https://download.pytorch.org/whl/cpu"' in pyproject
    assert "explicit = true" in pyproject

    lockfile = (REPOSITORY_ROOT / "uv.lock").read_text(encoding="utf-8")
    assert "https://download.pytorch.org/whl/cpu" in lockfile
    assert "nvidia-" not in lockfile
    assert "triton" not in lockfile
