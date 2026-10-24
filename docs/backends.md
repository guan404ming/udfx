# Backends

## Choose an engine

Use a backend name for a default engine context, or pass an instance that uses your existing context.

| Backend | Expression | Start here |
| --- | --- | --- |
| Polars | `polars.Expr` | [Polars example](backends/polars.md) |
| DuckDB | `duckdb.Expression` | [DuckDB example](backends/duckdb.md) |
| DataFusion | `datafusion.Expr` | [DataFusion example](backends/datafusion.md) |
| Spark | `pyspark.sql.Column` | [Spark example](backends/spark.md) |

## Compatibility

CI tests CPython 3.12, 3.13, and 3.14 against the locked engine versions.
Package smoke checks install each backend extra independently, using the lowest direct dependencies
on Python 3.12 and the highest on Python 3.14.

These checks do not establish support for every engine version, Spark Connect, managed Spark
services, or distributed clusters. Test your deployment with its own versions and data.

## Handle Python errors

Use `backend.execute(run)` when you need udfx to restore Python exceptions from engine errors.
See [Python semantics](python.md) for the fallback contract.
