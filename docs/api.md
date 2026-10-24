# Compile and types

## Compile

`compile(function, schema, backend="polars")` returns a `Translation`.
Pass a backend name or an existing [backend instance](backends.md).

### Example

```python
import udfx
from udfx.types import INT, STR


def prefix(text: str) -> str:
    return text[:3]


translation = udfx.compile(prefix, {"text": STR}, backend="polars")
assert translation.ok, translation.unsupported
```

### Return value

| Field | Meaning |
| --- | --- |
| `ok` | An engine expression was built |
| `expr` | The engine expression, or `None` |
| `unsupported` | The reason compilation was rejected |
| `backend` | The backend name |
| `function` | The original callable |
| `program` | The traced program for diagnostics |

## Types

Use `BOOL`, `INT`, `FLOAT`, `STR`, `DATETIME`, or `list_of(item_type)` from `udfx.types`.
`DATETIME` is naive with microsecond precision.

### Nullable columns

Declare nullable columns with `.or_none()`.

```python
schema = {"text": STR.or_none(), "count": INT}
```

The schema describes inputs; udfx does not validate the data against it.

## API stability

Public API signatures may change in minor releases before 1.0. Patch releases preserve them.
Internal IR and lowering modules can change between releases.

Next: [inspect a compilation](explain.md).
