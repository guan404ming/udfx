# Spark with Comet

Pass a Comet-enabled Spark session to `SparkBackend(session)`. udfx generates Spark expressions;
Comet decides which operators it can execute. The standalone DataFusion backend does not configure Comet.

## Set up

1. Install a Comet artifact that matches Spark, Scala, and the host platform.
2. Configure Comet before creating the Spark context.
3. Compile with `SparkBackend(session)` and inspect the resulting query.

Follow the [Comet installation guide](https://datafusion.apache.org/comet/user-guide/1.1/installation.html).
udfx does not change session configuration. Keep compatibility checks enabled, including strict
floating-point behavior; avoid incompatible-expression opt-ins.

## Inspect execution

Use `query.explain("formatted")` on the dataframe query:

- `CometProject` means Comet executes the projection; it can include JVM codegen dispatch.
- `ArrowEvalPython` means the plan includes Python execution.
- `udfx.explain(...)` reports compilation and fallback guards, not the executed Spark plan.
