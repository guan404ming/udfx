# Benchmarking

## Run

```sh
uv sync --locked
uv run python -m workloads.prepare
uv run python -m bench --threads 8 --output .cache/results/run.json
uv run python -m bench.report .cache/results/run.json --markdown .cache/results/run.md
```

| Option | Use |
| --- | --- |
| `--rows 1000 --warmups 0 --rounds 1` | Quick check, not a speedup claim |
| `--cases udfbench/lowerize` | Select cases |
| `--engines spark` | Select engines |

## Method

| Step | Rule |
| --- | --- |
| Validate | Full output must match Python before timing. A mismatch is an error. |
| Time | Two warmups and seven rotated rounds. Compilation and first run are timed separately. |
| Compare | Median of paired Python UDF to udfx ratios. The primary baseline is fixed before timing. |
| Summarize | Geometric mean over every case, Python-only cases too. Dedup weights each plan once. |

- Status splits native, runtime UDF, guarded, and Python fallback cases.
- Apache DataFusion registers Python UDF baselines as immutable.

| Mode | Runs |
| --- | --- |
| Memory | The UDF over prepared input. Apache Spark caches input first. |
| Query | Apache Parquet scan, the UDF, a null filter, and optional grouping. |

Both modes materialize Apache Arrow output. Queries are adapted, not the upstream UDFBench queries.

## Published results

[Performance](index.md#performance) uses one machine and up to 10,000 rows per case:

- UDFBench: all 25 UDFs.
- QURE: 36 functions, 27 Python and 9 Froid.
- Apache DataFusion: [patched build](backends/datafusion.md#large-functions).
- Apache Spark: [JVM runtime UDFs](backends/spark.md#execution).

## Reproducibility

- Result JSON records code and data hashes, versions, CPU, threads, Spark settings, and samples.
- Share with `bench.report --public`. It omits private diagnostics and inputs.
- Use an idle machine without coverage. The manual workflow needs a self-hosted
  `udfx-benchmark` runner.
