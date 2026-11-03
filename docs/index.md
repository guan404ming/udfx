# udfx

**An optimizing compiler for Python UDFs**

Run your Python UDFs as native expressions in Polars, DuckDB, Apache DataFusion, and Apache Spark,
without any manual rewrites.

## Performance

**Overall speedups on all four engines, in both execution modes.**

Geometric mean speedups across all 25 UDFBench UDFs, including Python fallback.
Each engine's Python UDF is the 1x baseline.

![UDFBench speedups, memory / query: Polars 2.21× / 1.78×. DuckDB 1.49× / 1.24×. Apache DataFusion 1.23× / 1.18×. Apache Spark 1.30× / 1.05×. Python UDF baseline is 1×.](assets/udfbench.svg)

Geometric mean speedups across 36 selected QURE functions, including Python fallback.

![QURE speedups, memory / query: Polars 4.49× / 3.34×. DuckDB 1.66× / 1.15×. Apache DataFusion 1.76× / 1.42×. Apache Spark 1.57× / 1.16×. Python UDF baseline is 1×.](assets/qure.svg)

- **Memory:** prepared input.
- **Query:** Apache Parquet, with filtering and grouping where defined.

See [Benchmarking](benchmarking.md) to reproduce the measurements.

## How It Works

![udfx architecture: Python UDFs pass through a shared compiler and one of four backend adapters, then run in a dataframe query with native execution or Python fallback.](assets/architecture.svg)

See [How It Works](how-it-works.md) for compilation and fallback.

## Get started

| Start here | What you will find |
| --- | --- |
| [User Guide](user-guide.md) | Install udfx and run your first function |
| [Backends](backends.md) | Examples for each engine |
| [Reference API](api.md) | Compile functions, declare types, and inspect results |
| [Contributor Guide](contributor-guide.md) | Build, test, and extend the compiler |
