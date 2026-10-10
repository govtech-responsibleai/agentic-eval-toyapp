"""Tests for the controlled-trace Make command interface."""

import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[2]


def _dry_run(*arguments: str) -> str:
    result = subprocess.run(
        ["make", "--dry-run", "trace-case", *arguments],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def test_trace_case_without_filters_uses_both_cli_defaults() -> None:
    """Omitting both filters must leave all metrics and variants selected."""
    command = _dry_run()

    assert "--metric" not in command
    assert "--variant" not in command


def test_trace_case_with_metric_only_uses_all_variant_default() -> None:
    """A metric filter with no variant must generate all of its variants."""
    command = _dry_run("METRIC=harmful_content_refusal")

    assert '--metric "harmful_content_refusal"' in command
    assert "--variant" not in command


def test_trace_case_with_variant_only_uses_all_metric_default() -> None:
    """A variant filter with no metric must apply across every metric."""
    command = _dry_run("VARIANT=control")

    assert "--metric" not in command
    assert '--variant "control"' in command


def test_trace_case_with_both_filters_generates_only_that_selection() -> None:
    """Both explicit filters must retain targeted generation."""
    command = _dry_run(
        "METRIC=harmful_content_refusal",
        "VARIANT=control",
    )

    assert '--metric "harmful_content_refusal"' in command
    assert '--variant "control"' in command
