# Apache DataFusion Comet

Pass an Apache Spark session configured for Apache DataFusion Comet to `SparkBackend(session)`.
udfx generates Apache Spark expressions. Apache DataFusion Comet decides which operators it can execute.
The standalone Apache DataFusion backend does not configure it.

## Set up

1. Install an Apache DataFusion Comet artifact that matches Apache Spark, Scala, and the host platform.
2. Configure Apache DataFusion Comet before creating the Apache Spark context.
3. Compile with `SparkBackend(session)` and inspect the resulting query.

Follow the [Apache DataFusion Comet installation guide](https://datafusion.apache.org/comet/user-guide/1.1/installation.html).
udfx does not change session configuration. Keep compatibility checks enabled, including strict
floating-point behavior. Avoid incompatible-expression opt-ins.

## Inspect execution

Use `query.explain("formatted")` on the dataframe query:

- `CometProject` means Apache DataFusion Comet executes the projection. It can include JVM codegen dispatch.
- `ArrowEvalPython` means the plan includes Python execution.
- `udfx.explain(...)` reports compilation and fallback guards, not the executed Apache Spark plan.
