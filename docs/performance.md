# Performance

Measure your workload. Removing Python execution does not guarantee faster queries.

## Recorded results

Each engine ran 165 cases with one million rows per case. Speedup compares udfx with that
engine's Python UDF baseline. The table reports geometric means; above 1x means faster.

| Engine | Speedup | Slower cases |
| --- | ---: | ---: |
| Polars eager | 27.96x | 0/165 |
| Polars lazy | 39.53x | 0/165 |
| DuckDB | 23.79x | 0/165 |
| DataFusion | 18.77x | 0/165 |
| Spark | 1.43x | 16/165 |

## Measurement scope

These are archived development results from different code snapshots, not a current-release
comparison or a ranking of engine speeds.

Polars, DuckDB, and DataFusion use one sweep at `9164405`, with cached Python baselines.
Spark uses a later snapshot, fresh baselines, and median per-case speedups from three sweeps.
Each timed measurement uses the best of three repetitions.

Read the methods for
[Polars, DuckDB, and DataFusion](https://github.com/guan404ming/udfx/blob/main/docs/benchmarks/engines.md)
and [Spark](https://github.com/guan404ming/udfx/blob/main/docs/benchmarks/spark.md).

## Measure locally

From a checkout:

```sh
uv run python -m bench.speed --rows 1000000 --repeats 3
```

Check correctness before timing. Keep versions, input, hardware, and session settings fixed.
Report slower cases alongside averages.
