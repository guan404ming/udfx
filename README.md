# udfx

Compile Python UDFs into native dataframe expressions without changing their results.

[![CI](https://github.com/guan404ming/udfx/actions/workflows/ci.yml/badge.svg)](https://github.com/guan404ming/udfx/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue.svg)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

## Why

A Python UDF is slow: every row crosses into the interpreter, and the optimizer cannot see inside.
Rewriting by hand is error prone. Polars 2.0.0 suggests `pl.col("a") is None` for
`lambda x: x is None` (a constant `False` in Python) and `cast(pl.Int64)` for `int(x)` (which
rejects `" 7"` and `"1_2"` that Python accepts).

udfx traces the function bytecode, keeps Python semantics in the generated expression, and runs the
original function on any row it cannot prove identical.

## Install

```
uv pip install -e .[polars]    # or duckdb, datafusion, spark
```

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

`compile(function, schema, backend="polars")` returns a `Translation`: `ok`, `expr`, `program`,
and `unsupported` (the reason when the function cannot be compiled). `udfx.explain(function,
schema)` prints the traced IR and the native expression.

## How it works

```
Python function
   | 1. trace bytecode: one symbolic state per path
Program: IR value + fault predicate
   | 2. lower each op with a backend rule and an exactness guard
native expression + exact predicate
   | 3. when(exact) native else run the Python function
```

1. [`frontend/`](src/udfx/frontend) runs the bytecode symbolically, forks on branches, inlines
   helpers, and merges the return paths into one decision tree.
2. [`ops/`](src/udfx/ops) names each Python operation with its exact reference and a fault
   predicate that is true exactly when Python raises.
3. [`backends/`](src/udfx/backends) lowers each op per engine. A rule may return an `exact`
   predicate when the engine differs from Python on some rows.
4. Rows where `exact` is false, or where Python would raise, run the original function through
   the engine's own UDF path. A raising row fails the query with the same exception type.

## Supported

| Family | Operations |
|---|---|
| Core | `not`, `and`, `or`, `is None`, truthiness |
| Numbers | `+ - * / // %`, comparisons, unary `-`, `abs`, `int`, `float`, `str`, `math.log10`, `math.isnan`, `math.isinf` |
| Strings | `lower upper strip lstrip rstrip split find rfind count startswith endswith replace join`, `+`, `in`, `len`, indexing, slices |

Python features: `if`/`else`, conditional expressions, `and`/`or`/`not`, chained comparisons,
`try`/`except`/`finally`, nested helpers and closures, constant dicts and sets, slices.

Not supported, so `compile` returns `ok == False`: loops, comprehensions, f-strings, keyword
arguments, `is` against anything but `None`, side effects, calls without a registered op.

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
- Globals and closure constants are read once at compile time.
- udfx does not validate data against the schema. Declare nullable columns with `.or_none()`.
- Polars eager mode has no common subexpression elimination; use a `LazyFrame` for the best speed.

## Performance

`uv run python bench/speed.py`, one million rows, Polars, best of three:

| Function | Python UDF | udfx eager | udfx lazy |
|---|---|---|---|
| `euro` (dict lookup, `float`, `/`, `try`) | 0.80 s | 29x | 110x |
| `lower_strip` | 0.58 s | 7.5x | 33x |
| `split_part` (`split("::")[2]`, `try`) | 0.70 s | 6.2x | 35x |
| `year` (`int(date[:4])`, `try`) | 0.73 s | 1.3x | 7.5x |
| `safe_division` | 0.57 s | 123x | 255x |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Apache-2.0, see [LICENSE](LICENSE).
