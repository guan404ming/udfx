# Benchmarking

## Run

```sh
uv sync --locked
uv run python -m workloads.prepare
uv run python -m bench --threads 8 --output .cache/results/run.json
uv run python -m bench.report .cache/results/run.json --markdown .cache/results/run.md
```

For a quick check, add `--rows 1000 --warmups 0 --rounds 1`.
Select cases with `--cases udfbench/lowerize` or engines with `--engines spark`.
Small runs validate the framework. They do not establish production speedups.

## Measurements

- **Validation:** full outputs must match Python before timing. Every selected row stays.
  An oracle exception or mismatch is an error, with no speedup.
- **Timing:** two warmups and seven rounds per case, in rotated order. Every sample is saved.
- **Ratios:** the median of paired baseline/udfx ratios, plus variation and slower-case counts.
- **Baselines:** each Python UDF form appears separately. The primary one is fixed before timing.
  Apache DataFusion registers these deterministic functions as immutable.
- **Means:** Overall includes every timed case, Python-only cases too. Dedup gives each distinct
  compiled-plan fingerprint equal weight.
- **Status:** engine expressions, runtime UDFs, guarded execution, and Python fallback are
  reported separately.

### Execution modes

| Mode | What runs |
| --- | --- |
| Memory | The UDF over prepared input. Apache Spark caches input first. |
| Query | Apache Parquet scan, the UDF, a null-output filter, and grouping when the workload defines a category. |

- All engines materialize Apache Arrow output.
- These are adapted queries, not all 21 upstream UDFBench queries.
- Compilation and first execution have separate timings. First execution is per case and mode,
  not a cold process or cold filesystem.

### Published results

The [UDFBench and QURE results](index.md#performance) use this scope:

| Item | Scope |
| --- | --- |
| Machine | One machine, up to 10,000 rows per case |
| UDFBench | 25 UDFs |
| QURE | 36 functions: 27 Python and 9 Froid |
| Apache DataFusion | Patched build ([Large functions](backends/datafusion.md#large-functions)) |
| Apache Spark | JVM runtime UDFs ([Execution](backends/spark.md#execution)) |

## Reproducibility

- Versioned JSON records code and data hashes, Python and engine versions, CPU, threads,
  Apache Spark settings, and raw samples.
- Data preparation can cache downloads. Timing never caches.
- For user-supplied sources, see [Public Datasets > Local sources](datasets.md#local-sources).
- Add `--public` to `bench.report` before sharing. It omits private diagnostics and inputs.
- Use an idle machine without coverage instrumentation. The manual GitHub workflow needs a
  self-hosted Linux runner labelled `udfx-benchmark`.
- A local lock prevents concurrent benchmark commands in one checkout.
