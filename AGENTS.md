# udfx

udfx compiles Python UDFs into native dataframe expressions. Results must never change.

## Commands

- Test: `uv run pytest` (fast tier). Full tier, as CI runs it: `UDFX_TESTS=full uv run pytest`.
- Lint: `uv run ruff check . && uv run ruff format --check .`
- Types: `uv run pyright`
- All three must pass before you commit.

## Layout

| Path | Owns |
|---|---|
| `src/udfx/types.py` | `Kind`: `DATETIME`, `STR`, `INT`, `FLOAT`, `BOOL`, `NONE`, `LIST`; `Type`, `list_of`, int64 bounds, `fits`. `DATETIME` is naive with microsecond precision |
| `src/udfx/ir.py` | IR nodes: `Col`, `Lit`, `Call`, `If`, `ListValue`, `Element`, `Comprehension` |
| `src/udfx/program.py` | `Program`: traced function, schema, value, fault, row mode, and `run` for fallback rows |
| `src/udfx/ops/` | Op specs: `core.py`, `strings.py`, `numbers.py`, `conversions.py`, `math.py`, `lists.py`, `regex.py`, `datetimes.py` |
| `src/udfx/semantics/` | Shared Python facts and policies for scalars, numbers, text, literals, lists, regex, datetimes, translation, and paths. Sequence, split, and padding policies live in `sequences.py`, `splitting.py`, and `padding.py` |
| `src/udfx/frontend/` | `trace.py` turns a function and schema into a program. The frontend knows nothing about backends |
| `src/udfx/frontend/decode/` | Decode version-specific bytecode and stack effects into blocks of semantic ops. `ops.py` defines the ops, `rules/` holds one module per opcode family plus `py313.py`, `py314.py`, and `py315.py` |
| `src/udfx/frontend/flow/` | Execute decoded blocks, helper calls, loops, and exception paths |
| `src/udfx/frontend/core/` | Shared values, bindings, evaluation results, and IR construction and simplification |
| `src/udfx/frontend/resolve/` | Resolve callables, constants, snapshots, and helper definitions and signatures |
| `src/udfx/frontend/operations/` | Evaluate Python operators, calls, containers, attributes, and formatting with explicit operands |
| `src/udfx/backends/base.py` | Backend interface, fallback composition, and `evaluate` helpers |
| `src/udfx/backends/lowering.py` | Generic lowering for calls, branches, and comprehensions. Backends never reimplement it |
| `src/udfx/backends/hoisting.py` | Finds a repeated list call in a comprehension body that does not read the element |
| `src/udfx/backends/rules.py` | Rule types, `RuleSet`, and boolean rules |
| `src/udfx/backends/registration.py` | Shared registration helpers and exports. Text and datetime helpers live in `registration_text.py` and `registration_datetimes.py` |
| `src/udfx/backends/<name>/` | One package per engine: `backend.py`, `rules.py`, and the same family files as `ops/`. Backends may also have `slicing.py` and `splitting.py`. Types use `dtypes.py`, except DataFusion uses `arrow.dtype` |
| `src/udfx/testing/` | Oracle, strategies, equivalence checks |
| `tests/conformance/` | Checks every op against Python on every installed backend |
| `tests/corpus/` | End to end acceptance cases, run with `uv run pytest -m corpus` |

## Rules

- Write no comments and no docstrings. License headers and lint or type directives are the only exceptions.
- Put imports at the top of the file.
- Keep each code file at 250 lines or fewer.
- Keep shared frontend dependencies in one direction: `operations/` uses `resolve/` and `core/`; `resolve/` uses `core/`. None imports `flow/` or `decode/`.
- Use the simplest code that works. Do not add abstractions, options, or config that no code uses.
- Reuse existing helpers. Do not copy logic between files.
- Use full words in names. Use no abbreviations except `op`, `ir`, `expr`.
- Keep op contracts in `ops/`, shared Python facts in `semantics/`, and engine calls in `backends/`.
- Only `frontend/decode/` knows the CPython version. `BASE` in `tables.py` is the 3.12 opcode table and `OVERRIDES` holds the changes of each later version. Add a new version there. Semantic ops (`Call`, `BinaryOp`, `Branch`, `ForEach`, and so on) carry explicit operands, so `flow/` and the shared op modules never read a stack.

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
| `x[a:b:c]` | `slice` | string or string list, start `INT?`, stop `INT?`, literal step `INT?` |
| `item in container` | `__contains__` | container, item |
| `-x` | `__neg__` | `x` |
| truthiness `if x:` | `bool` | `x` |

## Backend rule contract

A rule is `@RULES(name, *kinds)` in `src/udfx/backends/<name>/<family>.py`.

- Declare `literals=(1, 2)` on a rule to receive Python values for those zero-based parameters when the IR argument is a `Lit`. Nonliteral arguments stay engine expressions. Slice bounds, steps, and translation tables use this same contract.
- Declare `facts={"result": result_type}` to pass static facts as keyword arguments. Fact functions take the `Call` and live in `semantics/facts.py`. Keep op names out of generic lowering.
- A rule must never raise or panic for any input, including literal arguments. Where the op faults, any value is fine.
- If the rule matches Python only on some rows, return `Lowered(expr, exact)`. `exact` is true only on rows where `expr` equals Python. Other rows fall back to Python.
- Use `Lowered(expr, overflow=predicate)` when a native check detects int64 overflow. Generic lowering converts an intermediate overflow to an exactness guard. It uses a native fault for a final result where the backend supports it.
- Every op needs a rule on every backend. `tests/conformance/test_ops.py` enforces this.
- Put Python facts in `semantics/`, never in a backend. A backend file holds only engine calls.
- For a Python policy, fill its `*Primitives` record from `semantics/` with engine calls and call the policy function.
- Register mixed int and float rules with `register_float_variants`, `register_widened_variants`, `register_comparison`, `register_equality`, and `register_extremum` from `registration.py`. Give them the engine cast to float and the `OrderPrimitives` record.

## Backend contract

A backend is a package `src/udfx/backends/<name>/` whose `__init__.py` exports `BACKEND`, a `Backend` subclass. `udfx.backends.installed()` discovers it when its engine imports.

- `backend.py`: the `Backend` subclass. `column`, `literal`, and `select` build expressions. `both` and `negate` default to `&` and `~` in `base.py`. Override them only when the engine needs other operators.
- `list_value` builds lists. `list_transform` and `list_filter` enable native comprehensions. `base.py` uses Python fallback when a comprehension needs an unsupported list operation.
- `bind(value, body)` evaluates `value` once and passes it to `body`. Set `binds = True` to lower a repeated invariant list call once outside the comprehension lambdas. Only DuckDB binds. Spark cannot pass bound lambdas to a Python UDF.
- DuckDB `build` runs the whole program in Python when it uses a set operation or its expression has `SLOW_LAMBDAS` or more lambdas. Native DuckDB is slower than Python on those shapes.
- `fallback(native, exact, program)` runs Python on rows where `exact` is false and keeps `native` elsewhere. It calls `evaluate_pending_rows(program, pending, gather)`, which gathers the pending rows and calls `run_fallback_rows(program, rows)`.
- `evaluate(exprs, schema, rows)` runs expressions on in-memory rows for tests. Use `transpose_rows` for the input columns and `output_names` for the output aliases from `base.py`. Keep frame construction and result extraction in the backend.
- `rules.py`: `RULES = RuleSet()` from `registration.py` and engine helpers shared by rule files.
- `dtypes.py`: `Type` to engine type. DataFusion uses `arrow.dtype` instead.
- `core.py`, `strings.py`, `numbers.py`, `conversions.py`, `math.py`, `lists.py`, `regex.py`, `datetimes.py`: rules, one file per op family. Backends may put slice and split rules in `slicing.py` and `splitting.py`.
- Internal column names start with `RESERVED` (`"udfx "`) from `base.py`. `build` and `evaluate` reject user names that start with it after `fold_name`.
- Constructors never touch an engine. DuckDB uses `duckdb.default_connection()` unless a connection is given; DataFusion creates its `SessionContext` on first use; Spark uses the given session, else the active one, else `getOrCreate()`, and sets no config.
- `execute(run)` wraps every engine call so that a Python exception raised in a fallback row comes back as that exception: `run_fallback_rows` records it in thread-local state and raises a `FallbackError` whose message carries the type, and `unwrap` restores it from either. Engines that run the UDF in another thread or process (DuckDB, Spark) rely on the message, so only builtin exception types come back there.
- `native_raise(program)` returns a typed engine error expression or `None`. Encode row inputs as hexadecimal UTF-8 JSON after the program marker. `execute` replays that row through `Program.run` to recover the Python exception. Spark and DuckDB keep replay programs for the engine context lifetime.
- DuckDB names each fallback function by the program digest and a number per connection object. The catalog entry is shared by every connection of the database, but the Python callable lives only as long as the connection object that registered it. So two connection objects never share a name, and one connection object reuses its own.
- `fold_name(name)` returns the name as the engine compares it: identity for Polars and DataFusion, ASCII lower for DuckDB, and `str.lower` for Spark when `spark.sql.caseSensitive` is false. `build` rejects schemas whose folded names collide.
- Arrow backends implement the fallback UDF with `arrow.fallback`, which uses `arrow.make_gather`. Only batch unpacking and UDF registration stay in the backend.

## Tests

- New ops get conformance coverage automatically. Do not write per-op value tests.
- Add trap values to `src/udfx/testing/strategies.py` when you find a new semantic gap.
- Use `udfx.ops.registry.scoped()` to register temporary ops in a test.
- Commit messages: short imperative subject, capitalized, no trailing period.
