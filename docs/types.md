# udfx.types

Define the schema for [udfx.compile](compile.md) with type values from `udfx.types`.

## Scalar types

| Type | Values |
| --- | --- |
| `BOOL` | Booleans |
| `INT` | Signed 64-bit integers |
| `FLOAT` | Floating-point numbers |
| `STR` | Strings |
| `DATETIME` | Naive datetimes with microsecond precision |

## Nullable columns

Declare nullable columns with `.or_none()`.

```python
from udfx.types import INT, STR

schema = {"text": STR.or_none(), "count": INT}
```

## Lists

Use `list_of(item_type)` to declare list columns.

```python
from udfx.types import STR, list_of

schema = {"words": list_of(STR)}
```

The schema describes inputs. udfx does not validate data against it.

Pass your schema to [udfx.compile](compile.md).
