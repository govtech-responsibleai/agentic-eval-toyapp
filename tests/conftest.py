"""Shared pytest configuration."""

import pytest


@pytest.fixture
def anyio_backend() -> str:
    """Run anyio-marked tests on asyncio only; trio is not a dependency."""
    return "asyncio"
