# udfx.compile

## Usage

`udfx.compile(function, schema, backend="polars", *, result=None)` returns a `Translation`.
Pass a backend name or instance. See [Backends](backends.md).
Define the schema with [udfx.types](types.md).

## Example

```python
import udfx
from udfx.types import STR


def prefix(text: str) -> str:
    return text[:3]


translation = udfx.compile(prefix, {"text": STR}, backend="polars")
assert translation.ok, translation.unsupported
```

## JSON results

Use `result=STR.or_none()` when a parsed JSON value returns a string or null.
Scalar and scalar-list result types define the engine column type. A result that does not
fit that type raises `OverflowError`. udfx does not silently stringify objects or truncate integers.

Nested JSON values can remain dynamic inside the function. An untyped JSON return is unsupported.

## Return value

| Field | Meaning |
| --- | --- |
| `ok` | An engine expression was built |
| `expr` | The engine expression, or `None` |
| `unsupported` | The reason compilation was rejected |
| `backend` | The backend name |
| `function` | The original callable |
| `program` | The traced program for diagnostics |

## API stability

Public API signatures may change in minor releases before 1.0. Patch releases preserve them.
Internal IR and lowering modules can change between releases.

Inspect native execution and fallback with [udfx.explain](explain.md).
