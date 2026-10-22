# Contributing

## Set up

```
uv sync
```

From a checkout, `uv sync` installs the package, all four engines, and the test tools. Spark
needs Java 21.

## Checks

```
uv run ruff check . && uv run ruff format --check .
uv run pyright
uv run pytest
uv run pytest -m corpus
```

All must pass before a pull request. CI runs them on Python 3.12, 3.13, and 3.14 with
`HYPOTHESIS_PROFILE=ci` (100 examples per test instead of 25).

`uv run pytest` runs the tests in parallel with `-n auto` and prints the five slowest with
`--durations=5`. Pass `-n 0` to run in one process.

## Mine notebook UDFs

Run `uv run python -m bench.mine <csv> <output>` from the project root. The CSV must contain
`kernel_id` and `source` columns. The command writes `candidates.jsonl` and `summary.md` to the
output directory.

## Benchmark performance

Run `uv run python -m bench.speed` from the project root, or select a smaller run with
`uv run python -m bench.speed --rows 200 --engines polars duckdb --cases lower_strip row_total --repeats 1`.
The benchmark checks results against Python before timing and reports K2 (each speedup is at
least 1), K3 (deduplicated geometric mean speedup is at least 3), and K5 (cases with no fallback
in the plan). Each engine groups cases by the text of the built native expression, with input
columns renamed by schema position. The report shows case and group counts, both geometric means,
and duplicate groups with their fingerprint hashes, case names, and speedups. The deduplicated
mean gives each group equal weight, using the geometric mean of its case speedups.

## Code rules

- No comments, no docstrings. Lint and type directives are the only exceptions.
- Imports at the top of the file.
- The simplest code that works. No options or abstractions that nothing uses.
- Reuse helpers. Never copy logic between files.
- Full words in names. The only abbreviations are `op`, `ir`, and `expr`.
- Python facts live in `src/udfx/ops/` and `src/udfx/semantics/`. Engine calls live in
  `src/udfx/backends/<name>/`.

## Add an op

1. Register it in `src/udfx/ops/<family>.py`: `Op(name, params, result, reference, faults)`.
   `reference` is the real Python callable. `faults(*args)` returns `Fault(kind, when)` items whose
   `when` predicate is true exactly when Python raises `kind`.
2. A builtin or module function also needs an entry in `CALLABLES` in
   `src/udfx/frontend/calls.py`. Methods and operators resolve by name.
3. Add `@RULES(name, *kinds)` in every backend's `src/udfx/backends/<name>/<family>.py`. A rule
   never raises. If it matches Python only on some rows, return `Lowered(expr, exact)`.

`tests/conformance` covers the new op on every backend. Do not write per-op value tests.

## Add a backend

Copy the layout of `src/udfx/backends/polars/`: `backend.py` (a `Backend` subclass exported as
`BACKEND`), `rules.py`, `dtypes.py`, and one rule file per op family. `base.py` owns lowering and
fallback composition; never reimplement them. Internal column names contain a space. The backend
is ready when `uv run pytest` passes with its engine installed.

## Read a mismatch

A conformance failure prints a `Mismatch(row, expected, actual)`.

| `actual` | Meaning |
|---|---|
| a value | The engine returned something else than Python |
| `faults [...]` | Python raised, but the fault predicates say otherwise |
| `spurious faults [...]` | Python returned, but a fault predicate was true |
| `batch did not raise` | Python raised on a row, the engine ran the batch without error |

Fix the rule, the fault, or the `exact` guard. Never special case the row. When a gap was found by
hand, add the value to `src/udfx/testing/strategies.py` in the same pull request.

## Commits

Short imperative subject, capitalized, no trailing period: `Add rule for str.count`. One topic per
pull request.
