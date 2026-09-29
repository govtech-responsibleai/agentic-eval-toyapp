"""Tests for the installable package scaffold."""

from importlib.metadata import version

import agentic_eval_toyapp


def test_package_version_matches_distribution_metadata() -> None:
    """Keep the importable and distribution versions aligned."""
    assert agentic_eval_toyapp.__version__ == version("agentic-eval-toyapp")
