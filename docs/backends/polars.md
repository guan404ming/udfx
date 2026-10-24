# Polars

## Install

Install `udfx[polars]`.

## Run a query

```python
import polars as pl
import udfx
from udfx.types import STR


def prefix(text: str) -> str:
    return text[:3]


translation = udfx.compile(prefix, {"text": STR})
assert translation.ok, translation.unsupported
frame = pl.DataFrame({"text": ["hello", "world"]})
print(frame.lazy().select(translation.expr.alias("prefix")).collect())
```

The result contains `hel` and `wor`.
