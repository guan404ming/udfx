# DataFusion

## Install

Install `udfx[datafusion]`.

## Run a query

```python
import udfx
from datafusion import SessionContext
from udfx.backends.datafusion import DataFusionBackend
from udfx.types import STR


def prefix(text: str) -> str:
    return text[:3]


context = SessionContext()
backend = DataFusionBackend(context)
translation = udfx.compile(prefix, {"text": STR}, backend)
assert translation.ok, translation.unsupported
frame = context.from_pydict({"text": ["hello", "world"]})
print(backend.execute(lambda: frame.select(translation.expr.alias("prefix")).collect()))
```

The result contains `hel` and `wor`.

## Large functions

Large helper chains require an engine with the nested lambda planning fixes.
The stock DataFusion Python 55.0.0 build does not support these plans.
udfx checks planning support and reports unsupported functions before execution.

The [engine patches](https://github.com/guan404ming/udfx/tree/main/patches/datafusion)
include build instructions and validation results.
