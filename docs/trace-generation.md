# Controlled trace generation

Controlled generation requires the live OpenAI-compatible provider and enabled
Langfuse configuration. A live command can incur provider cost.

List supported metrics and variants:

```bash
make trace-cases-list
```

Generate every available metric and variant:

```bash
make trace-case
```

Apply optional filters:

```bash
make trace-case METRIC=harmful_content_refusal
make trace-case VARIANT=control
make trace-case METRIC=harmful_content_refusal VARIANT=control
make trace-case METRIC=harmful_content_refusal VARIANT=defect
```

With no `METRIC`, all metrics are selected. With no `VARIANT`, both control and
defect variants are selected. Generated local run information is written under
the gitignored `trace_cases/runs/` directory using the Langfuse-correlated
session UUID as the filename.

See [Controlled trace generation](../architecture.md#10-controlled-trace-generation)
for the technical flow and
[`harmful_content_refusal` documentation](../trace_cases/metrics/harmful_content_refusal/README.md)
for the metric-specific case contract.
