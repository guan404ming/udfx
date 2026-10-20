# Changelog

All notable changes to this project are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - unreleased

### Added

- Compile Python UDF bytecode into Polars, DuckDB, DataFusion, and Spark expressions.
- Keep Python semantics: truthiness, exceptions as faults, int64 overflow, Unicode case drift, literal parsing.
- Fall back to the Python function on rows that are not provably identical.
- `udfx.compile`, `udfx.explain`, `udfx.Translation`, `udfx.Unsupported`, `udfx.types`, and `udfx.__version__`.
- `Unsupported` messages that name the Python feature, such as loops or f-strings.
- A `py.typed` marker for type checkers.
- CPython 3.12, 3.13, and 3.14.
- Engines keep their own sessions: udfx sets no Spark or DuckDB configuration.
- Unicode case mapping differences are computed once per engine version and cached on disk.

[0.1.0]: https://github.com/guan404ming/udfx/compare/ea70aef...v0.1.0
