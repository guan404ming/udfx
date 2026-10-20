# udfx

Turn Python UDFs into native dataframe expressions, without changing their results.

```python
import polars as pl
import udfx
from udfx.types import STR


def extract_funder(project):
    if project:
        try:
            return project.split("::")[0]
        except Exception:
            return None
    return None


translation = udfx.compile(extract_funder, {"project": STR.or_none()})
df.select(translation.expr)
```

udfx traces the function bytecode symbolically, builds a backend-neutral IR that keeps Python
semantics explicit, and lowers it to native expressions. Rows where the native expression is not
provably identical to Python fall back to the original function, so results never change.

## Status

Pre-alpha. Python 3.12. Backend: Polars.
