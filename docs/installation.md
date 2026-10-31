# Installation

## Requirements

Use CPython 3.12, 3.13, or 3.14.
Apache Spark also requires Java 21. Install udfx and the function's referenced modules on every executor.

## Install a backend

Install the extra for your engine:

```sh
pip install "udfx[polars]"
```

| Backend | Package extra | Example |
| --- | --- | --- |
| Polars | `udfx[polars]` | [Polars](backends/polars.md) |
| DuckDB | `udfx[duckdb]` | [DuckDB](backends/duckdb.md) |
| Apache DataFusion | `udfx[datafusion]` | [Apache DataFusion](backends/datafusion.md) |
| Apache Spark | `udfx[spark]` | [Apache Spark](backends/spark.md) |

The extras declare the supported minimum dependencies. See
[pyproject.toml](https://github.com/guan404ming/udfx/blob/main/pyproject.toml) for their versions.

## From source

```sh
git clone https://github.com/guan404ming/udfx.git
cd udfx
uv sync --locked
```

The checkout installs all four engines and development tools. The optional `udfx[testing]`
extra provides the Python oracle and Hypothesis strategies for external use.

Run your first query with [Quick Start](quickstart.md).
