# Quick Start

Install the Polars extra using [Installation](installation.md), then follow these steps.

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

See [Backends](backends.md) for examples on each engine.
