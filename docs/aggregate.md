# udfx.aggregate

## Usage

`udfx.aggregate(function, schema, backend="polars", *, result=None)` returns a `Translation`
whose `expr` is an `Aggregate`. The function takes one `pandas.Series` per schema column and
returns a scalar.

## Example

```python
import pandas as pd
import udfx
from udfx.types import FLOAT


def revenue(price: pd.Series, discount: pd.Series) -> float:
    valid = discount[(discount >= 0) & (discount <= 1)]
    return (price * (1.0 - valid)).sum()


translation = udfx.aggregate(revenue, {"price": FLOAT, "discount": FLOAT})
plan = translation.expr
outputs = frame.select([expr.alias(f"c{index}") for index, expr in enumerate(plan.columns)])
value = plan.finish(
    [column.to_numpy() for column in outputs.get_columns()],
    lambda: [frame[name].to_numpy() for name in ("price", "discount")],
)
```

## Execution

1. Select `plan.columns`, the native per-row expressions, in one engine query.
2. Pass the outputs to `plan.finish(columns, inputs)`. It runs the pandas reductions and
   computes the result.
3. If a reduction result could make the function raise or differ, `finish` calls `inputs()` and
   runs the original function on the full input columns.

Supported reductions: `sum`, `mean`, `var`, `std`, `max`, `min`, `count`, `corr`, and `len`.

### Native aggregates

If every reduction is `max`, `min`, or `len`, `plan.reduced` is true and `plan.columns` are
native aggregate expressions.

| Engine | One aggregate query |
| --- | --- |
| DuckDB | `relation.aggregate(columns)` |
| Apache DataFusion | `frame.aggregate([], columns)` |

- Pass the one-row columns to `finish`.
- `finish` calls `inputs()` when a native result is null, NaN, or zero, because pandas may differ.

### Row order

Floating sums depend on row order, as in pandas. Pass the columns in the order the engine
returns them.
