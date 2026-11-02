# Public Datasets

## Included suites

| Suite | Fixed input | UDF origin |
| --- | --- | --- |
| [UDFBench](https://github.com/athenarc/UDFBench) | OpenAIRE tiny, commit `8da9875` | Unmodified upstream Python functions |
| [BTS Flights](https://transtats.bts.gov/ONTIME/) | January 2024 | Custom udfx workloads |
| [NYC 311](https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-2019/76ig-c548) | January 2019, first 50,000 IDs | Custom udfx workloads |
| [QURE](https://www.microsoft.com/en-us/research/project/qure/) | User-supplied code and TPC-H inputs | Selected upstream scalar functions |

## Prepare explicitly

```sh
uv run python -m workloads.prepare
uv run pytest -m public
```

Downloads live under `.cache/workloads`. Source manifests pin URLs, checksums, selection,
and provenance. A checksum change fails preparation. Tests never fetch data.
The repository contains 32-row BTS and NYC samples for offline checks.
`--rows` takes a prefix without repeating rows.

UDFBench source stays in the local cache with its upstream GPL provenance. Its README
lists 24 scalar UDFs. The `Spark` directory contains 25 Python files, including `jsonparse`.
The inventory keeps all 25. `addnoise` uses distribution validation because it is random.
Unsupported deterministic functions remain in the inventory. The current
[Performance](index.md#performance) covers 24 functions and excludes `stem` because of
large-plan costs. Aggregate and table UDFs are outside this scalar compiler evaluation.

## Local sources

QURE uses the Microsoft Research License: non-commercial, non-revenue-generating research only.
Redistribution is prohibited. Obtain QURE and retain its license.
Supply TPC-H `.tbl` files.

```sh
uv run python -m workloads.prepare --suites qure \
  --source qure.code="/path/to/QURE Benchmark" \
  --source qure.tpch="/path/to/tpch"
uv run python -m bench --suites qure
```

Runs scalar functions from `python/` and `froid/`, excluding full queries.
Import fixes produce `adapted` results.
QURE is opt-in. Start with small inputs because preparation materializes joins.

Keep cached sources, inputs, and raw reports private.
Export with `python -m bench.report RESULT.json --public`.

## Extend

Add a source manifest and preparation function, then register a suite loader in
`workloads/catalog.py`. Each workload declares its original function, argument types,
column bindings, result type, and optional query category, lookup, or native expression.

Add an offline fixture and adapter correctness checks. No runner changes are needed for
another dataset. A new query operator needs explicit adapter implementations and tests.
