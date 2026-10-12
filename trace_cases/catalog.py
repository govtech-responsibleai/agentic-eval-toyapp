"""Load and validate local controlled-case definitions."""

from pathlib import Path

from trace_cases.models import MetricBundle, MetricCaseSet, MetricGroundTruth

METRICS_DIRECTORY = Path(__file__).parent / "metrics"


class TraceCaseCatalogError(ValueError):
    """Raised when controlled-case files are missing or inconsistent."""


def supported_metrics(metrics_directory: Path = METRICS_DIRECTORY) -> tuple[str, ...]:
    """Return metric slugs that contain both required local data files."""
    if not metrics_directory.is_dir():
        return ()
    return tuple(
        path.name
        for path in sorted(metrics_directory.iterdir())
        if path.is_dir()
        and (path / "cases.json").is_file()
        and (path / "ground_truth.json").is_file()
    )


def load_metric_bundle(
    metric: str,
    metrics_directory: Path = METRICS_DIRECTORY,
) -> MetricBundle:
    """Load one metric pair and verify runtime/ground-truth alignment."""
    slug = metric.strip()
    if not slug or slug not in supported_metrics(metrics_directory):
        available = ", ".join(supported_metrics(metrics_directory)) or "none"
        raise TraceCaseCatalogError(
            f"unsupported metric {metric!r}; available metrics: {available}"
        )
    metric_directory = metrics_directory / slug
    try:
        cases = MetricCaseSet.model_validate_json(
            (metric_directory / "cases.json").read_text(encoding="utf-8")
        )
        ground_truth = MetricGroundTruth.model_validate_json(
            (metric_directory / "ground_truth.json").read_text(encoding="utf-8")
        )
        bundle = MetricBundle(cases=cases, ground_truth=ground_truth)
    except (OSError, ValueError) as error:
        raise TraceCaseCatalogError(
            f"invalid trace-case data for metric {slug!r}"
        ) from error
    if bundle.cases.metric != slug:
        raise TraceCaseCatalogError(
            f"metric directory {slug!r} does not match its declared metric"
        )
    return bundle
