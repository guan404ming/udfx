# How It Works

## Compile

Pass your function, input types, and backend to [udfx.compile](compile.md).
The compiler returns an engine expression or a reason it cannot compile the function.

```mermaid
flowchart LR
    accTitle: Compilation result
    accDescr: udfx compiles a function with input types and a backend into an engine expression, or reports why it is unsupported.
    function["Python function"] --> compiler["udfx.compile"]
    inputs["Input types + backend"] --> compiler
    compiler --> supported{"Supported?"}
    supported -->|Yes| expression["translation.expr"]
    supported -->|No| reason["translation.unsupported"]
```

Compilation does not process input rows. For example, `text.strip().lower()` becomes
string operations for your engine.

## Execute

Check `translation.ok`, then pass `translation.expr` to your dataframe query.
Your engine optimizes and executes the query.

## Fallback

Guards select native results where they preserve Python behavior.
Other rows run the original Python function.

```mermaid
flowchart LR
    accTitle: Row execution
    accDescr: Guards select a native result when valid. Other rows run the original Python function.
    row["Input row"] --> guard{"Native path valid?"}
    guard -->|Yes| native["Use native result"]
    guard -->|No| fallback["Run Python UDF"]
    native --> result["Result"]
    fallback --> result
```

Native errors or Python fallback can raise an exception and fail the query.

Use [udfx.explain](explain.md) to inspect native execution and fallback conditions.

Compile your first function with [Quick Start](quickstart.md).
