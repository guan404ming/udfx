# udfx

[![CI](https://github.com/guan404ming/udfx/actions/workflows/ci.yml/badge.svg)](https://github.com/guan404ming/udfx/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue.svg)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

udfx compiles Python UDFs into native expressions for Polars, DuckDB, DataFusion, and Spark. It
traces the function bytecode and keeps Python semantics in the generated expression. Rows where
udfx cannot show the same result run the original Python function.

## Install

```
pip install "udfx[polars]"
```

Other backend extras: `duckdb`, `datafusion`, `spark`.
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
`udfx.explain` reports the traced program and native expression.

A single parameter represents a row when the schema keys differ from its name:
`udfx.compile(lambda row: row["price"] * row.quantity, {"price": FLOAT, "quantity": INT})`.
Positional row access (`row[0]`, `row[-1]`) follows pandas 2.x Series behavior; pandas 3.0 raises KeyError instead.

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
| Core | Truthiness, `not`/`and`/`or`, `is None`, scalar `pandas.isna`/`notna` |
| Numbers | `+ - * / // %`, comparisons, unary `-`, `abs`, `math.log10`/`isnan`/`isinf` |
| Conversions | `int`, `float`, `str` |
| Math | Float power (`**`), `min`/`max`, `round`, `math.sqrt`/`ceil`/`floor`, selected NumPy scalar functions |
| Strings | Case conversion, trimming, padding, character tests, split/join, search, replace/translate, `+`, `in`, `len`, indexing, slices |
| Lists | Homogeneous literals, `list`, `+`, `in`, `len`, indexing, slices, `any`/`all`/`sum`; string sorting and sets |
| Regex | Literal patterns with `re.search`/`match`/`fullmatch`/`sub`/`findall`/`split`, compiled patterns, guarded `group(0)` |
| Datetimes | Naive microsecond `DATETIME`: fields, comparisons, `weekday`/`isoweekday`/`toordinal`, literal `strftime`/`strptime` formats |

Python features include row UDFs with `row["name"]` or `row.name`, branches, chained comparisons,
`try`/`except`/`finally`, helpers with constant defaults, keyword calls with inspectable signatures,
constant dict lookup and set membership, and POSIX string path operations. Tuple/list unpacking,
starred list unpacking, and tuple `*args` calls work.

`for` loops support constant tuples/lists of at most 64 items. List comprehensions support one
iterator with filters. Generator expressions must go directly to `any`, `all`, `sum`, `list`,
`sorted`, or string `join`. F-strings and literal `.format` positional templates support empty,
`d`, and `s` specs, `!s`, and numeric `!r`; literal `%s`/`%d` templates also work.

Main limits (`compile` returns `ok == False` with an `unsupported` reason):

- Dynamic `for` loops, `while`, nested comprehensions, list mutation, deferred generators, and
  comprehension faults under exception handlers.
- Dynamic format templates/specs, other format specs, string `!r`, and `!a`; nonempty `**kwargs`,
  `*args` without a tuple, and keyword calls without an inspectable signature.
- Regex patterns with lookaround, backreferences, or named groups. Matches expose only guarded
  `group()`/`group(0)`; `findall` allows at most one capture group, `split` none.
- Timezones, and timedelta values other than `.days`, `.seconds`, and `.total_seconds()` of a
  datetime difference. Datetime formats allow `%Y %m %d %H %M %S %f %j %y %%`.
- `with`, explicit `raise`, imports inside UDFs, global/nonlocal writes, file I/O, and unsupported calls.

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

Run `uv run python -m bench.speed` to compare udfx with Python UDFs on Polars eager and lazy,
DuckDB, DataFusion, and Spark; numbers depend on the machine.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0, see [LICENSE](LICENSE).
