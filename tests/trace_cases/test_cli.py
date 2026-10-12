"""Tests for controlled trace command selection."""

import pytest
from trace_cases.catalog import (
    TraceCaseCatalogError,
    load_metric_bundle,
    supported_metrics,
)
from trace_cases.cli import build_selection_plan
from trace_cases.models import CaseVariant


def test_selection_plan_defaults_to_every_metric_and_variant() -> None:
    """No filters should expand to every case currently in the catalog."""
    plan = build_selection_plan(None, "all")
    expected = tuple(
        (metric, case.variant)
        for metric in supported_metrics()
        for case in load_metric_bundle(metric).cases.cases
    )

    assert plan == expected


def test_selection_plan_filters_one_metric_and_keeps_all_variants() -> None:
    """A metric filter should retain every variant declared by that metric."""
    assert build_selection_plan("harmful_content_refusal", "all") == (
        ("harmful_content_refusal", CaseVariant.CONTROL),
        ("harmful_content_refusal", CaseVariant.DEFECT),
    )


def test_selection_plan_filters_one_variant_across_all_metrics() -> None:
    """A variant filter should apply to every supported metric."""
    assert build_selection_plan(None, "control") == tuple(
        (metric, CaseVariant.CONTROL) for metric in supported_metrics()
    )


def test_selection_plan_rejects_unknown_metric_before_generation() -> None:
    """An invalid metric filter must fail while building the execution plan."""
    with pytest.raises(TraceCaseCatalogError, match="unsupported metric"):
        build_selection_plan("not_implemented", "all")
