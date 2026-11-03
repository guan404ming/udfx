# Public Datasets

| Suite | Input | Functions |
| --- | --- | --- |
| [UDFBench](https://github.com/athenarc/UDFBench) | OpenAIRE tiny, commit `8da9875` | Unmodified upstream Python UDFs |
| [QURE](https://www.microsoft.com/en-us/research/project/qure/) | Your QURE copy and TPC-H `.tbl` files | Selected upstream scalar functions |

## Prepare

```sh
uv run python -m workloads.prepare
uv run pytest -m public
```

- Downloads go to `.cache/workloads`. Tests never fetch data.
- Manifests pin URLs, checksums, and provenance. A checksum change fails preparation.
- `--rows` takes a prefix.

## UDFBench

- 25 UDFs, including `jsonparse` from the upstream `Spark` directory.
- `addnoise` is random, so validation compares distributions.
- Aggregate and table UDFs are out of scope.

## QURE

QURE uses the Microsoft Research License: non-commercial research only, no redistribution.
Obtain it yourself and keep its license.

```sh
uv run python -m workloads.prepare --suites qure \
  --source qure.code="/path/to/QURE Benchmark" \
  --source qure.tpch="/path/to/tpch"
uv run python -m bench --suites qure
```

- Runs scalar functions from `python/` and `froid/`, not full queries.
- Preparation materializes joins. Start small. `--cases` builds only the joins it needs.
- Keep sources, inputs, and raw reports private. Share with `bench.report --public`.

## Add a suite

1. Add a source manifest and a preparation function.
2. Register a loader in `workloads/catalog.py`.
3. Declare each workload's function, argument types, column bindings, and result type.
4. Add adapter correctness checks. A new query operator needs adapter code and tests.
