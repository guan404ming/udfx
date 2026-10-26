# Tests and coverage

## Run checks

```sh
uv run pytest
UDFX_TESTS=full uv run pytest -m corpus
uv run python -m workloads.prepare --suites udfbench --rows 32
uv run pytest -m public
```

The fast tier checks conformance and offline data workloads on all four engines.
The full tier adds every frontend backend and Spark with ANSI off.
Nightly CI also checks Python 3.12, 3.13, and 3.14.
The public test checks upstream UDF results against Python without running benchmark timings.
It requires at least 22 of 24 deterministic UDFBench UDFs on each engine. Guarded cases must
execute at least one sampled row natively. One nondeterministic UDF stays excluded.

## Coverage dimensions

- **Code:** separate line and branch coverage for the compiler and the benchmark framework.
- **Semantics:** executed op/type signatures by backend, including observed values, nulls,
  faults, literal arguments, and guarded rows. Unexecuted signatures remain visible.
- **Public UDFs:** native, guarded, unsupported, error, or excluded, with a fixed inventory.
- **Rows:** evaluate the actual native guard over the selected data.
- **Queries:** validate complete materialized outputs before any warm timing.

Native means there is no Python fallback path. A successful compilation can still be guarded.
Custom UDFs on public data are reported separately from original upstream UDFs.
Signature coverage does not imply complete coverage of arbitrary Python programs.

## Generate reports

```sh
UDFX_TESTS=full uv run pytest --cov --cov-report= --junitxml=.cache/tests.xml
uv run coverage json -o .cache/code.json
uv run python -m bench.coverage --code .cache/code.json --junit .cache/tests.xml --output .cache/coverage.json
```

CI combines coverage across shards and Python versions. It checks the recorded floors in
`tests/coverage-baseline.json`. Semantic observations reuse existing oracle evaluations.
Performance runs disable coverage and use a separate machine.
