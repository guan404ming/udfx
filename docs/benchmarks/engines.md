# Polars, DuckDB, and DataFusion benchmark record

The [timings](engines.csv) contain 165 cases for each engine mode: Polars eager, Polars lazy,
DuckDB, and DataFusion. They come from the complete `9164405-repo-<engine>.md` reports
recorded on October 23, 2026. No cases were skipped.

## Source and method

The run used snapshot
[`9164405`](https://github.com/guan404ming/udfx/commit/9164405a3b5eecabf498a720e89636659013380c)
on an AMD Ryzen 9 9950X with 32 logical CPUs and Ubuntu 26.04.1.
Each case used one million rows and the best of three timings. Output checks preceded timing.
The command used `--baselines baselines.json`, so Python timings could come from earlier runs.
This is one sweep per engine, not a three-sweep median or a fresh paired comparison.

The CSV preserves the reports' six-decimal timings, baseline form, fallback flag, and pending share.
For each engine mode, divide Python seconds by udfx seconds for each case, then take the
geometric mean of all 165 ratios. Count ratios below 1 as slower cases.

## Limits

These results describe older code, not the current checkout or a published release.
They do not measure distributed jobs, ingestion, compilation, or arbitrary Python UDF coverage.
The [Spark record](spark.md) uses a later snapshot and a different aggregation procedure.
