# Spark

## Install

Install `udfx[spark]` and Java 21.
Install the same udfx version and the UDF's modules on every executor.

## Run a query

Use your existing session in an application. This example creates a local session.

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

The result contains `hel` and `wor`.

See [Comet integration](../comet.md) for native execution with Comet.
