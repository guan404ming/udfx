# Contributing

Thank you for helping with udfx. This page tells you how to set up, what to check, and how to add
an op or a backend. Read the [Code of Conduct](CODE_OF_CONDUCT.md) first.

## Set up

udfx needs Python 3.12 and [uv](https://docs.astral.sh/uv/).

```
uv sync
```

This installs the package, all engines, and the test tools into `.venv`.

## Checks

Run all three before you push. CI runs the same checks on every pull request.

```
uv run ruff check . && uv run ruff format --check .
uv run pyright
uv run pytest
```

To run the end to end acceptance corpus, use `uv run pytest -m corpus`. To see which corpus
functions translate, use `uv run python bench/coverage.py`.

CI sets `HYPOTHESIS_PROFILE=ci`, which runs 300 examples per test. The default profile runs 60. To
match CI locally, run `HYPOTHESIS_PROFILE=ci uv run pytest`.

## Code rules

- Write no comments and no docstrings. License headers and lint or type directives are the only
  exceptions.
- Put imports at the top of the file.
- Use the simplest code that works. Do not add options or abstractions that no code uses.
- Reuse existing helpers. Do not copy logic between files.
- Use full words in names. The only abbreviations are `op`, `ir`, and `expr`.
- Keep Python semantics in `src/udfx/ops/` and `src/udfx/semantics/`. Keep engine calls in
  `src/udfx/backends/<name>/`.

## Add an op

An op is one Python operation with an exact reference. Change three places.

1. Register the op in `src/udfx/ops/<family>.py` with
   `Op(name, params, result, reference, faults)`.
   - `reference` is the exact Python behavior. Use the real builtin or method when possible.
   - A param with `nullable=True` accepts `None`. For other params, the frontend adds a `TypeError`
     or `AttributeError` fault for `None`.
   - `faults(*args)` returns `Fault(kind, when)` items. `when` is an IR predicate. It must be true
     exactly when Python raises `kind`. Build it with `apply(name, *nodes)`. The first true fault
     wins, so order matters.
   - Names follow Python: `.lower` for a method, `int` for a builtin, `math.log10` for a module
     function, `__add__` for an operator, `__getitem__`, `slice`, `__contains__`, `__neg__`, and
     `bool` for truthiness.
2. Tell the frontend about the name. Methods (`x.lower()`) and the operators in
   `src/udfx/frontend/handlers/operators.py` resolve by name, so they need no change. A builtin or
   module function needs an entry in `CALLABLES` in `src/udfx/frontend/calls.py`.
3. Add a rule to every backend in `src/udfx/backends/<name>/<family>.py`:
   `@RULES(name, *kinds)`.
   - A rule must never raise or panic for any input, including literal arguments. Where the op
     faults, any value is fine.
   - If the rule matches Python only on some rows, return `Lowered(expr, exact)`. `exact` is true
     only on rows where `expr` equals Python. Other rows run the Python function.
   - Put Python facts, such as whitespace sets and literal grammars, in `src/udfx/semantics/`.
     A backend file holds only engine calls.

Do not write per-op value tests. `tests/conformance/test_ops.py` picks up every registered op and
checks it against `reference` on every installed backend. It also fails when a backend has no rule
for a registered op.

## Add a backend

A backend is a package `src/udfx/backends/<name>/`. Its `__init__.py` exports `BACKEND`, a
`Backend` subclass. `udfx.backends.installed()` finds it when its engine imports. Use
`src/udfx/backends/polars/` as the model.

- `backend.py`: the `Backend` subclass.
  - `column`, `literal`, `select`, `both`, and `negate` build expressions.
  - `fallback(native, exact, program)` runs `program.run` on the rows where `exact` is false and
    keeps `native` elsewhere.
  - `evaluate(exprs, schema, rows)` runs expressions on in-memory rows. Tests use it.
- `rules.py`: `RULES = RuleSet()` and engine helpers that several rule files share.
- `dtypes.py`: maps a udfx `Type` to an engine type.
- `core.py`, `strings.py`, `numbers.py`: rules, one file per op family, with the same names as
  `src/udfx/ops/`.
- Internal column names contain a space, for example `"udfx exact"`, so they cannot collide with
  user columns.
- Add the engine to the optional dependencies in `pyproject.toml`. The extras for DuckDB,
  DataFusion, and Spark already exist.

Never reimplement lowering or fallback composition. `src/udfx/backends/base.py` owns them. A new
backend is ready when `uv run pytest` passes with its engine installed. The conformance suite then
covers it with no further test code.

## Conformance tests

`tests/conformance/test_ops.py` runs two checks for every pair of installed backend and registered
op.

- `test_every_op_has_a_rule` fails if the backend lacks a rule for any op.
- `test_op_matches_python` draws rows with Hypothesis and calls `check_op`. Rows mix random values
  with trap values from `src/udfx/testing/strategies.py`. Sometimes one argument becomes a
  literal, because rules can treat literals differently from columns.

`check_op` lowers the op, evaluates the value, the exact predicates, and the fault predicates, and
compares each row with `op.reference`. It skips rows where an exact predicate is false. Whole
functions go through `check_function`, which compares every row with the Python function and checks
that a row that raises also raises in the engine.

A failure shows a `Mismatch` with the row, the expected Python outcome, and what the engine did.
Hypothesis shrinks the row to a small example.

| `actual` | Meaning |
|---|---|
| a value such as `'7'` | The engine returned a different value or type than Python |
| `faults [...]` | Python raised, but the first true fault predicate has a different kind, or none is true |
| `spurious faults [...]` | Python returned a value, but a fault predicate was true |
| `predicates [...]` | A guard or fault predicate returned something other than true, false, or null |
| `batch did not raise` | Python raised on a row, but the engine ran the batch without error |

Fix the rule, the fault predicate, or the `exact` guard. Do not special case the failing row.

When you find a semantic gap that the generators did not hit, add a trap value to
`STRING_TRAPS`, `INT_TRAPS`, or `FLOAT_TRAPS` in `src/udfx/testing/strategies.py`. Add it in the
same pull request as the fix. Use `udfx.ops.registry.scoped()` to register a temporary op in a
test.

## Commits and pull requests

- Write a short imperative subject of about 50 characters. Capitalize it. Do not end it with a
  period. Example: `Add rule for str.count`.
- Keep each pull request small and on one topic.
- Every check must pass. A pull request with a red check is not merged.
- Fill in the pull request template.
