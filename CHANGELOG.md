# Changelog

## Unreleased

- Compile Python UDF bytecode into Polars, DuckDB, DataFusion, and Spark expressions.
- Keep Python semantics: truthiness, exceptions as faults, int64 overflow, Unicode case drift, literal parsing.
- Fall back to the Python function on rows that are not provably identical.
- `udfx.compile` and `udfx.explain`.
- CPython 3.12, 3.13, and 3.14.
