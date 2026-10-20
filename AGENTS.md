# udfx

udfx compiles Python UDFs into native dataframe expressions. Results must never change.

## Commands

- Test: `uv run pytest`
- Lint: `uv run ruff check . && uv run ruff format --check .`
- Types: `uv run pyright`
- All three must pass before you commit.

## Layout

| Path | Owns |
|---|---|
| `src/udfx/types.py` | IR types (`Kind`, `Type`) |
| `src/udfx/ir.py` | IR nodes: `Col`, `Lit`, `Call`, `If` |
| `src/udfx/ops/` | Op specs. One file per family: `core.py`, `strings.py`, `numbers.py` |
| `src/udfx/frontend/` | Bytecode to IR. Knows nothing about backends |
| `src/udfx/backends/base.py` | Generic lowering. Backends never reimplement it |
| `src/udfx/backends/polars/` | Polars rules. One file per op family, same names as `ops/` |
| `src/udfx/testing/` | Oracle, strategies, equivalence checks |
| `tests/conformance/` | Checks every op against Python on every backend |

## Rules

- Write no comments and no docstrings. License headers and lint or type directives are the only exceptions.
- Put imports at the top of the file.
- Use the simplest code that works. Do not add abstractions, options, or config that no code uses.
- Reuse existing helpers. Do not copy logic between files.
- Use full words in names. Use no abbreviations except `op`, `ir`, `expr`.
- Keep Python semantics in `ops/` and backend semantics in `backends/`.
- Python 3.12 bytecode only.

## Op contract

An op is an `Op(name, params, result, reference, faults)` in `src/udfx/ops/<family>.py`.

- `reference` is the exact Python behaviour. Use the real builtin or method when possible.
- A param with `nullable=True` accepts `None`. The frontend adds `TypeError` or `AttributeError` faults for `None` on other params.
- `faults(*args)` returns `Fault(kind, when)` items. `when` is an IR predicate. It must be true exactly when Python raises `kind`. Build it with `apply(name, *nodes)`.
- Fault order matters. The first true fault wins.

Op names:

| Python | Op name | Params |
|---|---|---|
| method `x.lower()` | `.lower` | receiver first |
| builtin `int(x)`, `len(x)` | `int`, `len` | |
| module function `math.log10(x)` | `math.log10` | |
| operator `a + b` | `__add__` | `a`, `b` |
| comparison `a < b` | `__lt__` | `a`, `b` |
| `x[i]` | `__getitem__` | container, index |
| `x[a:b]` | `slice` | container, start `INT?`, stop `INT?` |
| `item in container` | `__contains__` | container, item |
| `-x` | `__neg__` | `x` |
| truthiness `if x:` | `bool` | `x` |

## Backend rule contract

A rule is `@RULES(name, *kinds)` in `src/udfx/backends/polars/<family>.py`.

- A rule must never raise for any input. Where the op faults, any value is fine.
- If the rule matches Python only on some rows, return `Lowered(expr, exact)`. `exact` is true only on rows where `expr` equals Python. Other rows fall back to Python.
- Every op needs a rule. `tests/conformance/test_ops.py` enforces this.

## Tests

- New ops get conformance coverage automatically. Do not write per-op value tests.
- Add trap values to `src/udfx/testing/strategies.py` when you find a new semantic gap.
- Use `udfx.ops.registry.scoped()` to register temporary ops in a test.
- Commit messages: short imperative subject, capitalized, no trailing period.
