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

`plan.columns` are native per-row expressions. Select them in one engine query.
`plan.finish(columns, inputs)` runs the pandas reductions `sum`, `mean`, `var`, `std`, `max`,
`min`, `count`, `corr`, and `len` on those columns and computes the result. If a reduction
result would make the function raise or behave differently, `finish` calls `inputs()` and runs
the original function on the full input columns instead.

If every reduction is `max`, `min`, or `len`, `plan.reduced` is true and `plan.columns` are
native aggregate expressions. Run them in one aggregate query, for example
`relation.aggregate(columns)` in DuckDB or `frame.aggregate([], columns)` in DataFusion, and pass
the one-row columns to `finish`. `finish` calls `inputs()` when a native result is null, NaN, or
zero, because pandas may differ there.

Floating sums depend on row order, as in pandas. Pass the columns in the order the engine
returns them.
