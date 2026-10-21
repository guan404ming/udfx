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
| `src/udfx/types.py` | IR types (`Kind`, `Type`), int64 bounds, `fits` |
| `src/udfx/ir.py` | IR nodes: `Col`, `Lit`, `Call`, `If` |
| `src/udfx/program.py` | `Program`: the traced function, its schema, value, fault, and `run` for fallback rows |
| `src/udfx/ops/` | Op specs. One file per family: `core.py`, `strings.py`, `numbers.py`, `conversions.py`, `math.py` |
| `src/udfx/semantics/` | Facts about Python that every backend needs: whitespace, literal grammars, operator tables, Unicode drift, and policies that build expressions from a primitives record |
| `src/udfx/frontend/` | Bytecode to IR. Knows nothing about backends. `handlers/` has one file per opcode family |
| `src/udfx/backends/base.py` | Generic lowering, fallback composition, and `evaluate` helpers. Backends never reimplement it |
| `src/udfx/backends/registration.py` | Rule types, `RuleSet`, boolean rules, and shared registration helpers |
| `src/udfx/backends/<name>/` | One package per engine: `backend.py`, `rules.py`, `dtypes.py`, and one rule file per op family with the same names as `ops/` |
| `src/udfx/testing/` | Oracle, strategies, equivalence checks |
| `tests/conformance/` | Checks every op against Python on every installed backend |
| `tests/corpus/` | End to end acceptance cases, run with `uv run pytest -m corpus` |

## Rules

- Write no comments and no docstrings. License headers and lint or type directives are the only exceptions.
- Put imports at the top of the file.
- Use the simplest code that works. Do not add abstractions, options, or config that no code uses.
- Reuse existing helpers. Do not copy logic between files.
- Use full words in names. Use no abbreviations except `op`, `ir`, `expr`.
- Keep Python semantics in `ops/` and backend semantics in `backends/`.
- Handlers target CPython 3.12 bytecode. `frontend/opcodes/` normalizes 3.13 and 3.14 streams to that set. Add a new version there, never in a handler.

## Op contract

An op is an `Op(name, parameters, result, reference, faults)` in `src/udfx/ops/<family>.py`.

- `reference` is the exact Python behaviour. Use the real builtin or method when possible.
- A param with `nullable=True` accepts `None`. The frontend adds `TypeError` or `AttributeError` faults for `None` on other params.
- `faults(*args)` returns `Fault(kind, when)` items. `when` is an IR predicate. It must be true exactly when Python raises `kind`. Build it with `apply(name, *nodes)`.
- Fault order matters. The first true fault wins.
- `not`, `and`, and `or` lower to `~`, `&`, and `|` on every engine. `RuleSet()` registers these rules. Do not write backend rules for them.

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

A rule is `@RULES(name, *kinds)` in `src/udfx/backends/<name>/<family>.py`.

- A rule must never raise or panic for any input, including literal arguments. Where the op faults, any value is fine.
- If the rule matches Python only on some rows, return `Lowered(expr, exact)`. `exact` is true only on rows where `expr` equals Python. Other rows fall back to Python.
- Every op needs a rule on every backend. `tests/conformance/test_ops.py` enforces this.
- Put Python facts in `semantics/`, never in a backend. A backend file holds only engine calls.
- For a Python policy, fill the primitives record from `semantics/` with engine calls and call its function. The records are `FloatPrimitives`, `OrderPrimitives`, `RoundPrimitives`, and `IntegerPrimitives` in `numbers.py`, `TextPrimitives` and `SpellingPrimitives` in `literals.py`, and `SlicePrimitives` and `SearchPrimitives` in `text.py`.
- Register mixed int and float rules with `register_float_variants`, `register_widened_variants`, `register_comparison`, `register_equality`, and `register_extremum` from `registration.py`. Give them the engine cast to float and the `OrderPrimitives` record.

## Backend contract

A backend is a package `src/udfx/backends/<name>/` whose `__init__.py` exports `BACKEND`, a `Backend` subclass. `udfx.backends.installed()` discovers it when its engine imports.

- `backend.py`: the `Backend` subclass. `column`, `literal`, and `select` build expressions. `both` and `negate` default to `&` and `~` in `base.py`. Override them only when the engine needs other operators.
- `fallback(native, exact, program)` runs Python on rows where `exact` is false and keeps `native` elsewhere. It calls `evaluate_pending_rows(program, pending, gather)`, which gathers the pending rows and calls `run_fallback_rows(program, rows)`.
- `evaluate(exprs, schema, rows)` runs expressions on in-memory rows for tests. Use `transpose_rows` for the input columns and `output_names` for the output aliases from `base.py`. Keep frame construction and result extraction in the backend.
- `rules.py`: `RULES = RuleSet()` from `registration.py` and engine helpers shared by rule files.
- `dtypes.py`: `Type` to engine type.
- `core.py`, `strings.py`, `numbers.py`, `conversions.py`, `math.py`: rules, one file per op family.
- Internal column names start with `RESERVED` (`"udfx "`) from `base.py`. `build` and `evaluate` reject user names that start with it after `fold_name`.
- Constructors never touch an engine. DuckDB uses `duckdb.default_connection()` unless a connection is given; DataFusion creates its `SessionContext` on first use; Spark uses the given session, else the active one, else `getOrCreate()`, and sets no config.
- `execute(run)` wraps every engine call so that a Python exception raised in a fallback row comes back as that exception: `run_fallback_rows` records it in thread-local state and raises a `FallbackError` whose message carries the type, and `unwrap` restores it from either. Engines that run the UDF in another thread or process (DuckDB, Spark) rely on the message, so only builtin exception types come back there.
- DuckDB names each fallback function by the program digest and a number per connection object. The catalog entry is shared by every connection of the database, but the Python callable lives only as long as the connection object that registered it. So two connection objects never share a name, and one connection object reuses its own.
- `fold_name(name)` returns the name as the engine compares it: identity for Polars and DataFusion, ASCII lower for DuckDB, and `str.lower` for Spark when `spark.sql.caseSensitive` is false. `build` rejects schemas whose folded names collide.
- Arrow backends implement the fallback UDF with `arrow.fallback`, which uses `arrow.make_gather`. Only batch unpacking and UDF registration stay in the backend.

## Tests

- New ops get conformance coverage automatically. Do not write per-op value tests.
- Add trap values to `src/udfx/testing/strategies.py` when you find a new semantic gap.
- Use `udfx.ops.registry.scoped()` to register temporary ops in a test.
- Commit messages: short imperative subject, capitalized, no trailing period.
