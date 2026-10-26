# Benchmarking

## Run

```sh
uv sync --locked
uv run python -m workloads.prepare
uv run python -m bench --threads 8 --output .cache/results/run.json
uv run python -m bench.report .cache/results/run.json --markdown .cache/results/run.md
```

For a quick check, add `--rows 1000 --warmups 0 --rounds 1`.
Select cases with `--cases flights/year_month` or engines with `--engines spark`.
Small runs validate the framework; they do not establish production speedups.

## Measurements

The runner checks full outputs against Python before timing. It keeps every selected row.
An oracle exception or mismatch produces an error, with no speedup reported.

Each case has two warmups and seven rounds. The runner rotates execution order and saves
every sample. Reports use the median of paired baseline/udfx ratios, plus variation and
slower-case counts. Each Python UDF form appears separately. Handwritten native expressions
appear where provided; the primary Python baseline is fixed before timing.
DataFusion registers these deterministic Python functions as immutable.
The deduplicated mean gives each distinct native-plan fingerprint equal weight.

### Execution modes

- **Memory:** project the UDF over prepared input. Spark caches input first.
- **Query:** scan Parquet; apply the UDF, filter null outputs, and group when the workload
  defines a category. The flight delay query also joins an airport lookup.

All engines materialize Arrow output. These are adapted queries, not a reproduction of all
21 upstream UDFBench queries. Compilation and first execution have separate timings.
First execution is per case and mode, not a cold process or cold filesystem measurement.

## Reproducibility

Versioned JSON records code and data hashes, Python and engine versions, CPU, threads,
Spark settings, and raw samples. Data preparation can cache downloads. Timing never caches.

Use an otherwise idle machine without coverage instrumentation. The manual GitHub workflow
requires a self-hosted Linux runner labelled `udfx-benchmark`; it does not register one.
The local lock prevents concurrent benchmark commands in one checkout.
