# udfx.explain

## Generate a report

`udfx.explain(function, schema, backend="polars", *, result=None)` returns diagnostic text.
Use the same schema and result type as [udfx.compile](compile.md).

```python
import udfx
from udfx.types import FLOAT


def float_text(value: float) -> str:
    return str(value)


print(udfx.explain(float_text, {"value": FLOAT}))
```

## Read the result

| Field | Meaning |
| --- | --- |
| `Status: native` | Engine expressions need no Python fallback or runtime UDF |
| `Status: runtime` | Compiled runtime UDFs run without Python fallback |
| `Status: guarded` | Some rows may need Python fallback |
| `Status: Python` | Nondeterministic functions run all rows in Python |
| `Status: unsupported` | Compilation failed with a reason |
| `Native guard` | False or null sends a row to Python |
| `Reason` | Why a fallback path exists |
| `Fault handling` | Native errors, Python fallback, or no fault path |
| `Compile time` | Compilation time, including engine setup and probes |
| `Runtime UDFs` | Emitted runtime calls, not measured invocations |

## Limits

The report uses the normal compilation path. It does not run input rows or measure the share
of fallback rows. Apache Spark still needs a session and Java.

A native plan can be slower than a Python UDF. Measure the actual query with your data.
The report format is diagnostic text and is not a stable API.

See [Performance](index.md#performance) for measured speedups.
