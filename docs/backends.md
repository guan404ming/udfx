# Backends

## Choose an engine

Use a backend name for a default engine context, or pass an instance that uses your existing context.

| Backend | Expression | Supported versions | Start here |
| --- | --- | --- | --- |
| Polars | `polars.Expr` | 2.0 or later | [Polars](backends/polars.md) |
| DuckDB | `duckdb.Expression` | 1.4.1 or later | [DuckDB](backends/duckdb.md) |
| Apache DataFusion | `datafusion.Expr` | 55 or later | [Apache DataFusion](backends/datafusion.md) |
| Apache Spark | `pyspark.sql.Column` | 4.1 or later | [Apache Spark](backends/spark.md) |

## Compatibility

CI tests CPython 3.12, 3.13, and 3.14 against the locked engine versions:
Polars 2.0.0, DuckDB 1.5.6, Apache DataFusion 55.0.0, and Apache Spark 4.2.0.
Package smoke checks install each backend extra independently, using the lowest direct dependencies
on Python 3.12 and the highest on Python 3.14.

These checks do not establish support for every engine version, Apache Spark Connect, managed Apache Spark
services, or distributed clusters. Test your deployment with its own versions and data.

## Handle Python errors

Use `backend.execute(run)` when you need udfx to restore Python exceptions from engine errors.
See [Supported Patterns > Semantics](python.md#semantics) for the fallback contract.
