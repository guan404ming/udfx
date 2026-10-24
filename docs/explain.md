# Explain a compilation

```python
import udfx
from udfx.types import FLOAT


def float_text(value: float) -> str:
    return str(value)


print(udfx.explain(float_text, {"value": FLOAT}, backend="polars"))
```

`explain(function, schema, backend)` returns a string. It uses the same compilation path as
`compile`, once. It does not run input rows or call the original function. It can initialize
the backend and run its normal capability probes. Spark therefore needs a session and Java.

| Field | Meaning |
|---|---|
| `Status: native` | The built expression needs no Python fallback UDF |
| `Status: guarded` | The built expression includes a Python fallback path |
| `Status: unsupported` | Compilation failed; the report includes the reason and available location |
| `Native guard` | The actual backend condition that selects native results |
| `Reason` | A value-equivalence check, a fault-equivalence check, or Python error handling requires fallback |
| `Fault handling` | The backend raises errors natively, uses Python fallback, or needs no fault path |
| `Compile time` | Wall time for this compilation, including backend initialization and probes |
| `Value`, `Fault`, `Native` | The traced value, Python fault condition, and final engine expression |

The native guard must be true for a row to use its native result. False and null guards send
the row through the original Python function. A guarded expression can still process every
row natively for a particular dataset. The report says `Fallback rows: not measured`; it does
not estimate a percentage or execute a query to count rows.

Native execution does not guarantee a speedup. Compile time is not query runtime, and a first
call can include cache or session setup. Use the benchmark or measure your actual query to
choose a deployment.

The report describes the final lowering attempt, including Spark's retry without bindings.
Reasons describe guard categories. They do not yet identify the source line of each guarded
operation. A custom backend that bypasses shared lowering reports fallback status as unknown.

## Relation to PyTorch

[`torch._dynamo.explain`](https://github.com/pytorch/pytorch/blob/main/torch/_dynamo/backends/debugging.py)
reports captured graphs, graph breaks, reasons, guards, and compile times. Its
[implementation](https://github.com/pytorch/pytorch/blob/main/torch/_dynamo/eval_frame.py)
runs a function with example arguments to collect that information.

udfx compiles from a declared schema. Its relevant boundary is the row-level native guard.
The report therefore shows native execution and guarded fallback instead of graph-break
counts. The guard comes from lowering; the report does not parse engine plan text.
