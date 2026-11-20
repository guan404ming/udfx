# Apache Spark

## Install

Install `udfx[spark]` and Java 21.
It supports Apache Spark 4.1 or later. CI tests 4.2.0.
Install the same udfx version and the UDF's modules on every executor.

## Run a query

Use your existing session in an application. This example creates a local session.

```python
import udfx
from pyspark.sql import SparkSession
from udfx.backends.engines.spark import SparkBackend
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

Functions with very large expressions, or with lambdas that need Python fallback, run in Python.

Runtime UDFs execute in Apache Spark's JVM. They do not imply native execution in Apache DataFusion Comet.

See [Apache DataFusion Comet](../comet.md) for native execution with Apache DataFusion Comet.

## Python UDF transpiler

Apache Spark builds that include `pyspark.sql.transpile` can replace a Python UDF with Apache Spark expressions.
`transpiler.install()` registers udfx as a transpiler and puts it before the transpilers that the session already uses.
It also sets `spark.sql.experimental.optimizer.transpilePyUDFs` to `true`.

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from udfx.backends.engines.spark import transpiler

session = SparkSession.builder.getOrCreate()
transpiler.install(session)


@udf("bigint", useArrow=False)
def score(value: int) -> int:
    return value * 3 + 7 if value > 0 else -value


session.range(-2, 3).select(score("id")).show()
```

Apache Spark transpiles only regular Python UDFs in ANSI mode.
Set `useArrow=False` on the UDF, or set `spark.sql.execution.pythonUDF.arrow.enabled` to `false`.

udfx returns an expression only when the function runs natively and gives the Python result for every row.
For other functions, Apache Spark tries the next transpiler or runs the UDF in Python.
When Python raises an exception for a row, the query fails with an error that names the function.
When an integer result does not fit in `bigint`, the query fails with `ARITHMETIC_OVERFLOW`.
The Python UDF returns NULL for that row.
