# udfx

[![CI](https://github.com/guan404ming/udfx/actions/workflows/ci.yml/badge.svg)](https://github.com/guan404ming/udfx/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue.svg)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**An optimizing compiler for Python UDFs**

Run your Python UDFs as native expressions in Polars, DuckDB, DataFusion, and Spark,
without any manual rewrites.

## Install

```sh
pip install "udfx[polars]"
```

Choose the extra for your backend: `polars`, `duckdb`, `datafusion`, or `spark`.
Spark also needs Java 21.

## Quick start

```python
import polars as pl
import udfx
from udfx.types import STR


def normalize(text: str) -> str:
    return text.strip().lower()


translation = udfx.compile(normalize, {"text": STR})
assert translation.ok, translation.unsupported

frame = pl.DataFrame({"text": [" Alice ", "BOB"]})
result = frame.lazy().select(translation.expr.alias("name")).collect()
print(result["name"].to_list())
```

```text
['alice', 'bob']
```

Use `udfx.explain(normalize, {"text": STR})` to inspect native execution and fallback reasons.
Declare nullable columns with `.or_none()`. Unsupported functions return `ok == False` with a reason.

## Documentation

- [User Guide](https://guan404ming.github.io/udfx/): installation, backends, and Python support
- [API Reference](https://guan404ming.github.io/udfx/api/): compile, types, and explain
- [Contributor Guide](https://guan404ming.github.io/udfx/contributing/): development and documentation

Explore [measured speedups and trade-offs](https://guan404ming.github.io/udfx/performance/)
across all four engines.

[Contributing](CONTRIBUTING.md) · [Changelog](CHANGELOG.md) · [Apache-2.0 license](LICENSE)
