# udfx.compile

## Usage

`udfx.compile(function, schema, backend="polars", *, result=None, series=False)` returns a `Translation`.
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

## pandas scalar UDFs

Pass `series=True` to compile a pandas scalar UDF. The function takes one `pandas.Series` per
schema column and returns a `pandas.Series` of the same length. Columns must be non-null `STR`,
`INT`, `FLOAT`, or `BOOL`.

```python
import numpy as np
import pandas as pd
import udfx
from udfx.types import FLOAT


def sigmoid(price: pd.Series) -> pd.Series:
    return pd.Series([1 / (1 + np.exp(-value)) for value in price])


translation = udfx.compile(sigmoid, {"price": FLOAT}, series=True)
```

udfx turns loops over the series and vectorized series operations into one native value per row.
A function whose rows depend on other rows, such as `len(values)`, is unsupported.

Supported series operations:

- Operators: `+`, `-`, `*`, `/`, `**`, comparisons, `&`, `|`, `~`, and unary `-`.
- Methods: `add`, `sub`, `mul`, `div`, `truediv`, `pow`, their `r` forms, `eq`, `ne`, `lt`,
  `le`, `gt`, `ge`, `abs`, `round`, `clip`, `between`, `where`, `mask`, `isna`, `isnull`,
  `notna`, `notnull`, `fillna`, `astype(float)`, `astype(str)`, and `map` with a dict.
- The `str` accessor: `len`, `slice`, `contains`, `count`, `replace`, and the Python string
  methods `lower`, `upper`, `strip`, `lstrip`, `rstrip`, `title`, `capitalize`, `swapcase`,
  `casefold`, `startswith`, `endswith`, `find`, `rfind`, `zfill`, `center`, `ljust`, `rjust`,
  and the `is` predicates.
- Functions: `pandas.isna`, `pandas.notna`, `numpy.where`, `numpy.clip`, `numpy.power`,
  `numpy.minimum` and `numpy.maximum` on integers, and the numpy math functions that udfx knows.

Pattern, flag, and bound arguments must be literals.
Rows that the engine cannot match exactly run the original function on those rows.

For pandas aggregate UDFs, see [udfx.aggregate](aggregate.md).

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
