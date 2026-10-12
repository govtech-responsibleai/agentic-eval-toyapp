"""Tests for controlled-case and ground-truth catalog integrity."""

import json
from pathlib import Path

import pytest
from trace_cases.catalog import (
    TraceCaseCatalogError,
    load_metric_bundle,
    supported_metrics,
)
from trace_cases.models import CaseVariant


def test_catalog_exposes_only_the_initial_harmful_content_metric() -> None:
    """Stage 9 MVP must advertise exactly its implemented metric."""
    assert supported_metrics() == ("harmful_content_refusal",)
    bundle = load_metric_bundle("harmful_content_refusal")

    assert [case.variant for case in bundle.cases.cases] == [
        CaseVariant.CONTROL,
        CaseVariant.DEFECT,
    ]
    assert len({case.user_message for case in bundle.cases.cases}) == 1
    assert len({case.replacement_markdown for case in bundle.cases.cases}) == 2
    labels = {label.expected_pass for label in bundle.ground_truth.labels}
    assert labels == {False, True}


def test_catalog_rejects_an_unknown_or_blank_metric(tmp_path: Path) -> None:
    """Unsupported selection should fail before credentials or services are used."""
    with pytest.raises(TraceCaseCatalogError, match="unsupported metric"):
        load_metric_bundle("", tmp_path)
    with pytest.raises(TraceCaseCatalogError, match="unsupported metric"):
        load_metric_bundle("not_implemented", tmp_path)


def test_catalog_rejects_mismatched_ground_truth_ids(tmp_path: Path) -> None:
    """Every runtime case must have exactly one private expected label."""
    metric_directory = tmp_path / "example_metric"
    metric_directory.mkdir()
    cases = {
        "metric": "example_metric",
        "cases": [
            {
                "case_id": "case-0101",
                "variant": "control",
                "user_message": "Same input",
                "replacement_markdown": "Safe output",
            },
            {
                "case_id": "case-0102",
                "variant": "defect",
                "user_message": "Same input",
                "replacement_markdown": "Defective output",
            },
        ],
    }
    labels = {
        "metric": "example_metric",
        "labels": [
            {
                "case_id": "case-0101",
                "expected_pass": True,
                "rationale": "Expected control.",
            },
            {
                "case_id": "case-9999",
                "expected_pass": False,
                "rationale": "Wrong identifier.",
            },
        ],
    }
    (metric_directory / "cases.json").write_text(json.dumps(cases), encoding="utf-8")
    (metric_directory / "ground_truth.json").write_text(
        json.dumps(labels), encoding="utf-8"
    )

    with pytest.raises(TraceCaseCatalogError, match="invalid trace-case data"):
        load_metric_bundle("example_metric", tmp_path)
