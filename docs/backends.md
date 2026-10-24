# Backend quick starts

Install the extra for the engine you use. These examples return `hel` and `wor`.
Check `translation.ok` before passing `translation.expr` to an engine.

## Polars

Install `udfx[polars]`.

```python
import polars as pl
import udfx
from udfx.types import STR


def prefix(text: str) -> str:
    return text[:3]


translation = udfx.compile(prefix, {"text": STR})
assert translation.ok, translation.unsupported
frame = pl.DataFrame({"text": ["hello", "world"]})
print(frame.lazy().select(translation.expr.alias("prefix")).collect())
```

## DuckDB

Install `udfx[duckdb]`. Build and execute on the same connection object.

```python
import duckdb
import udfx
from udfx.backends.duckdb import DuckDBBackend
from udfx.types import STR


def prefix(text: str) -> str:
    return text[:3]


connection = duckdb.connect()
backend = DuckDBBackend(connection)
translation = udfx.compile(prefix, {"text": STR}, backend)
assert translation.ok, translation.unsupported
frame = connection.sql("SELECT * FROM (VALUES ('hello'), ('world')) AS inputs(text)")
print(backend.execute(lambda: frame.select(translation.expr.alias("prefix")).fetchall()))
connection.close()
```

## DataFusion

Install `udfx[datafusion]`.

```python
import udfx
from datafusion import SessionContext
from udfx.backends.datafusion import DataFusionBackend
from udfx.types import STR


def prefix(text: str) -> str:
    return text[:3]


context = SessionContext()
backend = DataFusionBackend(context)
translation = udfx.compile(prefix, {"text": STR}, backend)
assert translation.ok, translation.unsupported
frame = context.from_pydict({"text": ["hello", "world"]})
print(backend.execute(lambda: frame.select(translation.expr.alias("prefix")).collect()))
```

## Spark

Install `udfx[spark]` and Java 21. Use your existing session in an application. This example
creates a local session. Install the same udfx version and the UDF's modules on every executor.

```python
import udfx
from pyspark.sql import SparkSession
from udfx.backends.spark import SparkBackend
from udfx.types import STR


def prefix(text: str) -> str:
    return text[:3]


session = SparkSession.builder.master("local[2]").getOrCreate()
backend = SparkBackend(session)
translation = udfx.compile(prefix, {"text": STR}, backend)
assert translation.ok, translation.unsupported
frame = session.createDataFrame([("hello",), ("world",)], ["text"])
print(backend.execute(lambda: frame.select(translation.expr.alias("prefix")).collect()))
session.stop()
```

See [Spark with Comet](comet.md) for execution-plan checks and compatibility limits.

## Compatibility scope

udfx supports CPython 3.12, 3.13, and 3.14. The package extras declare the minimum engine and
Arrow versions. CI runs the full suite against the lockfile on all three Python versions.
Package checks also install each extra in isolation: the lowest declared direct dependencies
on Python 3.12 and the highest resolution on Python 3.14. Transitive dependencies use the
highest compatible versions. The checks exercise native results, Python
fallback, exception restoration, and explain output from the built wheel.

These package checks are smoke tests, not full conformance tests for every engine version.
Spark package checks use local worker processes. They do not establish compatibility with
Spark Connect, managed Spark distributions, or a multi-machine cluster.
