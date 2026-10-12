"""Command-line interface for controlled trace generation."""

import argparse
import json
import sys
from collections.abc import Sequence

from pydantic import ValidationError

from trace_cases.catalog import (
    TraceCaseCatalogError,
    load_metric_bundle,
    supported_metrics,
)
from trace_cases.models import CaseVariant
from trace_cases.runner import (
    TraceCaseExecutionError,
    encode_public_summary,
    generate_case,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trace-case",
        description="Generate realistic controlled traces for METIS metric validation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="list supported metrics and variants")
    generate = subparsers.add_parser("generate", help="generate controlled traces")
    generate.add_argument("--metric")
    generate.add_argument(
        "--variant",
        choices=[variant.value for variant in CaseVariant] + ["all"],
        default="all",
    )
    return parser


def _list_metrics() -> int:
    payload: list[dict[str, object]] = []
    for metric in supported_metrics():
        bundle = load_metric_bundle(metric)
        payload.append(
            {
                "metric": metric,
                "variants": [case.variant.value for case in bundle.cases.cases],
            }
        )
    sys.stdout.write(json.dumps(payload, indent=2) + "\n")
    return 0


def build_selection_plan(
    metric: str | None,
    selected_variant: str,
) -> tuple[tuple[str, CaseVariant], ...]:
    """Resolve and validate every metric/variant pair before live execution."""
    metrics = (metric,) if metric is not None else supported_metrics()
    if not metrics:
        raise TraceCaseCatalogError("no controlled trace metrics are available")

    requested_variant = (
        None if selected_variant == "all" else CaseVariant(selected_variant)
    )
    plan: list[tuple[str, CaseVariant]] = []
    for metric_name in metrics:
        bundle = load_metric_bundle(metric_name)
        available_variants = tuple(case.variant for case in bundle.cases.cases)
        if requested_variant is None:
            plan.extend((metric_name, variant) for variant in available_variants)
            continue
        if requested_variant not in available_variants:
            raise TraceCaseCatalogError(
                f"metric {metric_name!r} does not support variant "
                f"{requested_variant.value!r}"
            )
        plan.append((metric_name, requested_variant))
    return tuple(plan)


def _generate(metric: str | None, selected_variant: str) -> int:
    plan = build_selection_plan(metric, selected_variant)
    summaries: list[str] = []
    for metric_name, variant in plan:
        run, manifest_path = generate_case(metric=metric_name, variant=variant)
        summaries.append(encode_public_summary(run, manifest_path))
    sys.stdout.write("\n".join(summaries) + "\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse one command and return a process exit code."""
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "list":
            return _list_metrics()
        return _generate(arguments.metric, arguments.variant)
    except (
        TraceCaseCatalogError,
        TraceCaseExecutionError,
        ValidationError,
        ValueError,
    ) as error:
        sys.stderr.write(f"trace-case error: {error}\n")
        return 1
