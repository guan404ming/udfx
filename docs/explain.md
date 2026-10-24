# Explain

## Generate a report

Use `udfx.explain(function, schema, backend="polars")` to inspect a compilation.
It returns diagnostic text.

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
| `Status: native` | No Python fallback UDF is needed |
| `Status: guarded` | Some rows may need Python fallback |
| `Status: unsupported` | Compilation failed; a reason follows |
| `Native guard` | False or null sends a row to Python |
| `Reason` | Why a fallback path exists |
| `Fault handling` | Native errors, Python fallback, or no fault path |
| `Compile time` | Compilation time, including engine setup and probes |

## Limits

The report uses the normal compilation path. It does not run input rows or measure the share
of fallback rows. Spark still needs a session and Java.

A native plan can be slower than a Python UDF. Measure the actual query with your data.
The report format is diagnostic text and is not a stable API.

Next: [performance measurements](performance.md).
