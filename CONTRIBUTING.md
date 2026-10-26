# Contributing

## Set up

```
uv sync
```

From a checkout, `uv sync` installs the package, all four engines, and the test tools. Spark
needs Java 21.

For the documentation site, run `uv run --group docs zensical serve`. See the
[documentation guide](docs/contributing.md) for build commands and page-length guidelines.

## Checks

```
uv run ruff check . && uv run ruff format --check .
uv run pyright
uv run pytest
uv run pytest -m corpus
```

All must pass before a pull request. PR CI uses Python 3.12 and 25 examples per test.
Nightly CI runs Python 3.12, 3.13, and 3.14 with `HYPOTHESIS_PROFILE=ci` (100 examples).

The default run uses Spark with ANSI on and checks the frontend through Polars.
Conformance and offline public-data workloads run on all four engines.
Set `UDFX_TESTS=full` to include all frontend engines and Spark with ANSI off.
The separate `corpus` and `public` markers select acceptance and downloaded upstream cases.

CI also builds the sdist and then the wheel. It installs the wheel with one extra per clean
environment and checks native, fallback, and error paths. It uses the lowest declared direct
dependencies on Python 3.12 and the highest on Python 3.14. The release workflow publishes the
same artifacts only after these checks pass.

`uv run pytest` uses at most four workers and groups backend cases by engine.
This reuses engine sessions and avoids starting Spark in every worker.
It prints the five slowest tests. Pass `-n 0` to use one process.

## Mine notebook UDFs

Run `uv run python -m bench.mine <csv> <output>` from the project root. The CSV must contain
`kernel_id` and `source` columns. The command writes `candidates.jsonl` and `summary.md` to the
output directory.

## Benchmark performance

Prepare public data explicitly, then run the four engines:

```sh
uv run python -m workloads.prepare
uv run python -m bench --threads 8 --output .cache/results/run.json
uv run python -m bench.report .cache/results/run.json --markdown .cache/results/run.md
```

The runner validates every result before two warmups and seven paired rounds.
It retains raw timings, errors, unsupported UDFs, data checksums, and engine settings.
It does not download data or reuse timing baselines.

See [benchmark methods](docs/benchmarking.md), [public data](docs/datasets.md), and
[test coverage](docs/testing.md). Notebook mining remains exploratory and does not enter
the fixed public-workload coverage denominator.

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
