# udfx

[![CI](https://github.com/guan404ming/udfx/actions/workflows/ci.yml/badge.svg)](https://github.com/guan404ming/udfx/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue.svg)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

udfx compiles Python UDFs into native expressions for Polars, DuckDB, DataFusion, and Spark. It
traces the function bytecode and keeps Python semantics in the generated expression. Rows where
udfx cannot show the same result run the original Python function.

## Features

- One expression per function for Polars, DuckDB, DataFusion, and Spark.
- Python semantics for truthiness, exceptions, int64 overflow, Unicode case, and literal parsing.
- Row-level fallback to the original Python function.
- `udfx.explain` returns a report with the traced program and the native expression.

## Install

```
pip install "udfx[polars]"    # or duckdb, datafusion, spark
```

`udfx.testing` (the Python oracle and Hypothesis strategies) needs `udfx[testing]`.

## Quick start

```python
import polars as pl
import udfx
from udfx.types import STR


def year(date):
    if date:
        try:
            return int(date[:4])
        except Exception:
            return -1
    return None


translation = udfx.compile(year, {"date": STR.or_none()})
df = pl.DataFrame({"date": ["2024-05-17", None, "", "20x4-01-01", " 123-1"]})
print(df.select(year=translation.expr)["year"].to_list())
```

```
[2024, None, None, -1, 123]
```

`compile(function, schema, backend="polars")` returns a `Translation`: `function`, `ok`, `expr`, `program`,
`backend`, and `unsupported` (the reason when the function cannot be compiled).

A single parameter represents a row when the schema keys differ from its name:
`udfx.compile(lambda row: row["price"] * row.quantity, {"price": FLOAT, "quantity": INT})`.

## How it works

1. [`frontend/`](src/udfx/frontend) runs the bytecode symbolically, forks on branches, inlines
   helpers, and merges the return paths into one decision tree.
2. [`ops/`](src/udfx/ops) names each Python operation with its exact reference and a fault
   predicate that is true exactly when Python raises.
3. [`backends/`](src/udfx/backends) lowers each op per engine. A rule may return an `exact`
   predicate that is false on rows where the engine result can differ from Python.
4. Rows where `exact` is false, or where Python would raise, run the original function through
   the engine's own UDF path. A raising row fails the query.

## Supported Python

| Family | Operations |
|---|---|
| Core | `not`, `and`, `or`, `is None`, truthiness |
| Numbers | `+ - * / // %`, comparisons, unary `-`, `abs`, `int`, `float`, `str`, `math.log10`, `math.isnan`, `math.isinf` |
| Strings | `lower upper strip lstrip rstrip split find rfind count startswith endswith replace join`, `+`, `in`, `len`, indexing, slices |
| Rows | `row["name"]` with a constant key, `row.name` for schema columns |

Python features: `if`/`else`, conditional expressions, `and`/`or`/`not`, chained comparisons,
`try`/`except`/`finally`, nested helpers and closures, constant dicts and sets, slices.

Not supported, so `compile` returns `ok == False` and `unsupported` names the feature: loops,
comprehensions, f-strings, list and set literals, `with`, `raise`, generators, imports, global
assignment, attribute access on a column value, keyword arguments, `is` against anything but
`None`, calls without a registered op.

## Backends

| Backend | Expression | Extra |
|---|---|---|
| Polars | `polars.Expr` | `udfx[polars]` |
| DuckDB | `duckdb.Expression` | `udfx[duckdb]` |
| DataFusion | `datafusion.Expr` | `udfx[datafusion]` |
| Spark | `pyspark.sql.Column` | `udfx[spark]` |

Every op has a rule on every backend. `tests/conformance` checks each pair against Python with
Hypothesis and a set of trap values (`"1_2"`, `" 12 "`, `"İ"`, `"ß"`, int64 bounds, `nan`, `-0.0`).

## Guarantees and limits

- Each row equals the Python result, or the row runs the Python function. Checked by tests, not
  by proof.
- A raising row fails the query. Raw Polars queries rebuild the exception from its message,
  which works for builtin types and single-argument constructors. `backend.execute(run)`
  restores the exact exception object on Polars. DuckDB, DataFusion, and Spark raise their own
  UDF error whose message carries the Python type and message.
  `backend.execute(run)` restores builtin Python exception types that accept a message,
  for example `backend.execute(lambda: relation.fetchall())`.
- Globals and closure constants are frozen at compile time for both native and fallback paths.
- udfx does not validate data against the schema. Declare nullable columns with `.or_none()`.
- On Polars, a `LazyFrame` gives the best speed.
- Nested slice and round chains can grow rapidly on every engine; integer division chains can also grow rapidly on DuckDB, DataFusion, and Spark.
- Spark executors need the modules referenced by the function and the same udfx version.
- DuckDB expressions are bound to the connection object they were built on. Pass your connection with
  `DuckDBBackend(connection)` and run the query on it. Compiling the same function again reuses
  its fallback function on that connection object. Each thread must use its own connection or cursor.
- `requires-python` stops before 3.15 on purpose: bytecode support is per CPython version.

## Performance

`uv run python bench/speed.py`, one million rows, Polars, best of three:

| Function | Python UDF | udfx eager | udfx lazy |
|---|---|---|---|
| `euro` (dict lookup, `float`, `/`, `try`) | 0.79 s | 164x | 168x |
| `lower_strip` | 0.57 s | 7.6x | 34x |
| `split_part` (`split("::")[2]`, `try`) | 0.70 s | 6.4x | 40x |
| `year` (`int(date[:4])`, `try`) | 0.73 s | 1.6x | 8.8x |
| `safe_division` | 0.56 s | 140x | 302x |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0, see [LICENSE](LICENSE).
