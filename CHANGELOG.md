# Changelog

All notable changes to this project are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - unreleased

### Added

- Report execution status, native guards, fallback reasons, fault handling, and compilation time in `udfx.explain`.
- Verify built wheels with isolated backend extras before publication.

- Compile Python UDF bytecode into Polars, DuckDB, DataFusion, and Spark expressions.
- Keep Python semantics: truthiness, exceptions as faults, int64 overflow, Unicode case drift, literal parsing.
- Fall back to the Python function on rows that are not provably identical.
- `udfx.compile`, `udfx.explain`, `udfx.Translation`, `udfx.Unsupported`, `udfx.types`, and `udfx.__version__`.
- `Unsupported` messages that name the Python feature, such as loops or f-strings.
- A `py.typed` marker for type checkers.
- CPython 3.12, 3.13, and 3.14.
- Engines keep their own sessions: udfx sets no Spark or DuckDB configuration.
- Unicode case mapping differences are computed once per engine version and cached on disk.

### Fixed

- Exclude versioned virtual environments from source distributions.
- Require DuckDB 1.4.1 and DataFusion 55 for the APIs in use; declare DataFusion's Arrow dependency.

- Cache DuckDB fallback registrations per connection object; use a separate connection or cursor per thread.
- Reduce operand repetition in int64 clamps, integer division, slicing, rounding, and min/max; document the remaining nested-chain growth limits.
- Round large half-integers to even on DuckDB.
- Rebuild exceptions from their messages in raw Polars queries for builtin types and single-argument constructors; restore the exact exception object through `backend.execute(run)`.
- Preserve rebuilt fallback error messages so `backend.execute(run)` can restore builtin exception types that accept a message.
- Raise a clear `RuntimeError` when an explicit Spark session has stopped.
- Escape string literals and schema names in `udfx.explain` output.

[0.1.0]: https://github.com/guan404ming/udfx/compare/ea70aef...v0.1.0
