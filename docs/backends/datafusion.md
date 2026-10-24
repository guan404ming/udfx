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
