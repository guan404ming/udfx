# Quick start

Install [the Polars extra](installation.md), then run these steps in order.

## 1. Compile a function

```python
import polars as pl
import udfx
from udfx.types import STR


def normalize(text: str) -> str:
    return text.strip().lower()


translation = udfx.compile(normalize, {"text": STR})
assert translation.ok, translation.unsupported
```

## 2. Run a query

```python
frame = pl.DataFrame({"text": [" Alice ", "BOB"]})
result = frame.lazy().select(translation.expr.alias("name")).collect()
print(result["name"].to_list())
```

```text
['alice', 'bob']
```

## 3. Inspect the compilation

Use `udfx.explain(normalize, {"text": STR})` to inspect native execution and fallback reasons.
Declare nullable columns with `.or_none()`. Unsupported functions return `ok == False` with a reason.

Next, choose a [backend](backends.md).
