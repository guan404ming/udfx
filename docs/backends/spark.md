# Apache Spark

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

## Execution

udfx uses Apache Spark expressions and a bundled JVM runtime for word frequency ranking
and JSON string encoding. Apache Spark loads the runtime when a query first needs it.
No additional download or compilation is required.

`udfx.explain()` identifies runtime UDFs separately from built-in expressions
and Python fallback. If Apache Spark rejects runtime registration, udfx reports a warning
and uses its existing expression implementation.

Runtime UDFs execute in Apache Spark's JVM. They do not imply native execution in Apache DataFusion Comet.

See [Apache DataFusion Comet](../comet.md) for native execution with Apache DataFusion Comet.
