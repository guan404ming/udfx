# Public Datasets

## Included suites

| Suite | Fixed input | UDF origin |
| --- | --- | --- |
| [UDFBench](https://github.com/athenarc/UDFBench) | OpenAIRE tiny, commit `8da9875` | Unmodified upstream Python functions |
| [QURE](https://www.microsoft.com/en-us/research/project/qure/) | User-supplied code and TPC-H inputs | Selected upstream scalar functions |

## Prepare explicitly

```sh
uv run python -m workloads.prepare
uv run pytest -m public
```

- Downloads go to `.cache/workloads`. Tests never fetch data.
- Source manifests pin URLs, checksums, selection, and provenance. A checksum change fails
  preparation.
- `--rows` takes a prefix without repeating rows.

### UDFBench

- Source stays in the local cache with its upstream GPL provenance.
- 25 functions: the README lists 24, and the `Spark` directory adds `jsonparse`.
  The inventory keeps all 25, including unsupported ones.
- `addnoise` is random, so it uses distribution validation.
- [Performance](index.md#performance) covers all 25. Aggregate and table UDFs are out of scope.

## Local sources

QURE uses the Microsoft Research License: non-commercial, non-revenue-generating research only.
Redistribution is prohibited. Obtain QURE, retain its license, and supply TPC-H `.tbl` files.

```sh
uv run python -m workloads.prepare --suites qure \
  --source qure.code="/path/to/QURE Benchmark" \
  --source qure.tpch="/path/to/tpch"
uv run python -m bench --suites qure
```

- Runs scalar functions from `python/` and `froid/`, not full queries. Import fixes produce
  `adapted` results.
- QURE is opt-in. Preparation materializes joins, so start with small inputs.
  `--cases` builds only the joins those cases read.
- Keep cached sources, inputs, and raw reports private.
  Export with `python -m bench.report RESULT.json --public`.

## Extend

1. Add a source manifest and a preparation function.
2. Register a suite loader in `workloads/catalog.py`.
3. Declare each workload's original function, argument types, column bindings, result type,
   and optional query category.
4. Add adapter correctness checks. The runner needs no changes. A new query operator needs
   explicit adapter implementations and tests.
