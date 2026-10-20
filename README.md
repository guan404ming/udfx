# udfx

Compile Python UDFs into native dataframe expressions without changing their results.

[![CI](https://github.com/guan404ming/udfx/actions/workflows/ci.yml/badge.svg)](https://github.com/guan404ming/udfx/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue.svg)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

## Why

A Python UDF is slow because every row crosses the engine boundary into the interpreter. The
optimizer cannot see inside the function, so it cannot push filters, prune columns, or fuse work.

Rewriting the UDF by hand is error prone. Polars 2.0.0 warns on `map_elements` and suggests a
replacement. The suggestion can be wrong:

- For `lambda x: x is None`, it suggests `pl.col("a") is None`. In Python, that is the constant
  `False`. It does not test the column.
- For `lambda x: int(x)`, it suggests `pl.col("s").cast(pl.Int64)`. Python accepts `" 7"` and
  `"1_2"`. The cast rejects both with `InvalidOperationError`.

udfx compiles the function bytecode and keeps Python semantics. A row where the native expression
is not provably equal to Python runs the original function.

## Install

udfx is pre-alpha and not on PyPI yet. Install from source:

```
git clone https://github.com/guan404ming/udfx
cd udfx
uv pip install -e .[polars]
```
After release, use `pip install udfx[polars]`.

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
print(translation.ok)

df = pl.DataFrame({"date": ["2024-05-17", None, "", "20x4-01-01", " 123-1"]})
print(df.select(year=translation.expr)["year"].to_list())
print([year(value) for value in df["date"]])
```

Output:

```
True
[2024, None, None, -1, 123]
[2024, None, None, -1, 123]
```

The schema maps each parameter name to a type, in parameter order. `STR.or_none()` allows nulls.
`udfx.compile(function, schema, backend="polars")` returns a `Translation` with `ok`, `expr` (a
`pl.Expr` here), `program` (the traced `Program`), and `unsupported` (the reason for a failure).
A function with a loop does not translate:

```python
import udfx
from udfx.types import STR


def first(text):
    for char in text:
        return char


failed = udfx.compile(first, {"text": STR})
print(failed.ok, failed.expr, failed.program)
print(failed.unsupported)
```

Output:

```
False None None
opcode GET_ITER at first+4 GET_ITER
```

## How it works

```
Python function
   | 1. trace bytecode (symbolic execution, path conditions)
Program: IR value + fault predicate
   | 2. lower each op with a backend rule and an exactness guard
native expression + exact predicate
   | 3. fallback: when(exact).then(native).otherwise(run Python)
```

1. Tracing. [`src/udfx/frontend/`](src/udfx/frontend) walks the bytecode with one state per path.
   A state holds a stack, local names, and a path condition. Branches fork the state, and helper
   calls are inlined. [`trace.py`](src/udfx/frontend/trace.py) merges the returns and raises, each
   under its path condition, into one `Program`.
2. IR with Python semantics. [`src/udfx/ir.py`](src/udfx/ir.py) has four nodes: `Col`, `Lit`,
   `Call`, and `If`. A `Call` points to an `Op` in [`src/udfx/ops/`](src/udfx/ops). An op names the
   exact Python behavior and its faults. A fault is an IR predicate that is true exactly when
   Python raises. `Program.fault` is the union of all raising paths.
3. Lowering with exactness guards. [`src/udfx/backends/base.py`](src/udfx/backends/base.py) lowers
   each node with the rule for its op and argument kinds. A rule returns an expression and may
   return an `exact` predicate, true only on rows where the expression equals Python. For example,
   Polars lowercasing differs from `str.lower` on some code points, so the rule guards on them.
   Guards combine up the tree and join the fault predicate, so a row that would raise is never
   exact.
4. Fallback. `Backend.fallback` builds the final expression. The Polars version in
   [`backends/polars/backend.py`](src/udfx/backends/polars/backend.py) gathers the rows where
   `exact` is false or null and calls `Program.run`, which calls your function. If the function
   raises, the exception reaches the caller. If an `int` result does not fit in 64 bits,
   `Program.run` raises `OverflowError`.

## What is supported

Types, from `udfx.types`: `STR`, `INT` (64-bit), `FLOAT`, `BOOL`, lists of these, and `.or_none()`.

| Op family | Source | Operations |
|---|---|---|
| Core | [`ops/core.py`](src/udfx/ops/core.py) | `not`, `and`, `or`, `is None`, truthiness |
| Numbers | [`ops/numbers.py`](src/udfx/ops/numbers.py) | `+ - * / // %`, `< <= > >= == !=`, unary `-`, `abs`, `int`, `float`, `str(int)`, `math.log10`, `math.isnan`, `math.isinf` |
| Strings | [`ops/strings.py`](src/udfx/ops/strings.py) | `lower upper strip lstrip rstrip split find rfind count startswith endswith replace join`, `+`, `==`, `!=`, `in`, `len`, indexing, slices |

Python features the frontend handles:

- `if`/`else`, conditional expressions, truthiness, `x is None`, `x is not None`
- `and`, `or`, `not` (operands are returned, as in Python), chained comparisons
- `try`/`except` (one class, a tuple of classes, bare `except`, `as name`), nested `try`, `finally`
- Nested helper functions and closures, inlined
- Module-level dicts: `RATES[key]` becomes an equality chain, with `KeyError` as a fault
- Constant tuples, lists, and sets of all `str` or all `int`, used with `in`
- Subscripts and slices on strings and lists of strings, with `IndexError` tracked as a fault

Not supported. For these, `compile` returns `ok == False`:

- Loops, comprehensions, f-strings, `with`
- Keyword arguments, parameter defaults, `*args`, `**kwargs`, keyword-only parameters
- `is` and `is not` against anything other than `None`
- Side effects, nonlocal writes, recursion
- Calls with no registered op, such as `json.loads`, `random.gauss`, `str.casefold`, or `**`
- Mixed-type constant containers and dicts with non-scalar values

`uv run python bench/coverage.py` translates 32 of 37 acceptance cases. Five are expected to fail.

## Backends

| Backend | Status | Extra |
|---|---|---|
| Polars | Available | `udfx[polars]` |
| DuckDB | In progress | `udfx[duckdb]` |
| DataFusion | In progress | `udfx[datafusion]` |
| Spark | In progress | `udfx[spark]` |

`udfx.backends.installed()` returns the backends whose engine imports. The extras exist in
`pyproject.toml`, but the in-progress backend packages are not on `main` yet.
[CONTRIBUTING.md](CONTRIBUTING.md) describes the backend contract.

## Guarantees

For every row, the result equals what the Python function returns, or the row runs the Python
function itself. That includes rows where the function raises. Conformance tests enforce this:
- [`tests/conformance/test_ops.py`](tests/conformance/test_ops.py) checks every registered op
  against its Python `reference` on every installed backend, with Hypothesis. It checks values on
  rows where the guard holds. It checks that predicted faults match the exception Python raises.
- [`src/udfx/testing/strategies.py`](src/udfx/testing/strategies.py) mixes random values with trap
  values: strings such as `"1_2"`, `" 12 "`, `"İ"`, and `"ß"`, int64 bounds, `2**53 + 1`, `nan`,
  `inf`, and `-0.0`.
- Frontend tests and `uv run pytest -m corpus` run whole functions through `check_function`.

Known limits:
- The check is testing, not proof. A gap that no generated value hits can ship.
- CPython 3.12, 3.13, and 3.14. The handlers read 3.12 bytecode; [`frontend/opcodes/`](src/udfx/frontend/opcodes) rewrites newer streams to that set.
- Globals, closure cells, and constant containers are read once at compile time. Fallback rows
  call the live function, so the two paths can disagree if those values change later.
- udfx does not check data against the schema. Declare nullable columns with `.or_none()`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Report security issues as [SECURITY.md](SECURITY.md)
describes. This project follows the [Code of Conduct](CODE_OF_CONDUCT.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
