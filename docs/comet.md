# Spark with Comet

Use `SparkBackend(session)` with a Spark session that already has Comet installed.
udfx produces Spark expressions. Comet chooses which parts of the Spark plan it executes.
The DataFusion backend uses the standalone DataFusion Python API and does not configure Comet.

Install a Comet artifact that matches your Spark version, Scala version, operating system,
and CPU architecture. Configure Comet before the Spark context starts. Follow the
[Comet installation guide](https://datafusion.apache.org/comet/user-guide/1.1/installation.html).
udfx does not change Spark or Comet configuration.

Keep Comet's compatibility checks enabled. Do not enable `allowIncompatible` options to increase
native coverage. udfx requires exact Python results, including null, overflow, Unicode, and
exception behavior. The measurements below use `spark.comet.exec.strictFloatingPoint=true`.

## Check the executed plan

```python
import udfx
from udfx.backends.spark import SparkBackend
from udfx.types import STR


def prefix(text: str) -> str:
    return text[:3]


backend = SparkBackend(session)
translation = udfx.compile(prefix, {"text": STR}, backend)
assert translation.ok, translation.unsupported
query = frame.select(translation.expr.alias("prefix"))
print(backend.execute(query.collect))
query.explain("formatted")
```

`udfx.explain(function, schema, backend)` describes udfx lowering. The dataframe plan shows Spark execution.
`CometProject` confirms that Comet runs the projection. It can contain both Rust expressions
and JVM codegen dispatch. `ArrowEvalPython` identifies Python execution. Use Comet's
[plan guide](https://datafusion.apache.org/comet/user-guide/1.1/understanding-comet-plans.html) to inspect
fallback reasons and codegen dispatch.

## Performance limits

Native coverage does not guarantee a speedup. Small jobs can spend more time on planning,
scheduling, shuffle setup, and row or columnar conversion than on the expression itself.
Spark's cache can introduce a conversion that a Comet Parquet scan does not need.
JVM codegen dispatch also has Arrow and JNI costs. Compare complete actions on your input
source and partition count. Create a fresh query for each timed action so adaptive execution
does not reuse a completed shuffle.

See [the Comet measurements](benchmarks/comet.md) for the tested versions, coverage, and limits.
