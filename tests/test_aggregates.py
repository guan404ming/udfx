import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
import pytest
from conftest import backend_params

from udfx import aggregate
from udfx.backends.base import Backend
from udfx.testing.outcome import same
from udfx.types import FLOAT, INT, STR, Type

ROWS = [
    (100.0, 0.05, "R", 3),
    (250.5, 1.5, "N", 7),
    (80.25, 0.0, "R", 2),
    (13.0, -0.1, "A", 7),
    (42.0, 0.08, "N", 1),
]
SCHEMA = {"price": FLOAT, "tax": FLOAT, "flag": STR, "count": INT}


def taxed(price: pd.Series, tax: pd.Series, flag: pd.Series, count: pd.Series) -> float:
    valid = tax[(tax >= 0) & (tax <= 1)]
    return (price * valid).sum()


def mapped(price: pd.Series, tax: pd.Series, flag: pd.Series, count: pd.Series) -> float:
    return flag.map({"R": 50.0, "N": 30.5}).sum()


def returned(price: pd.Series, tax: pd.Series, flag: pd.Series, count: pd.Series) -> float:
    total = len(flag)
    if total == 0:
        return 0.0
    return len(flag[flag == "R"]) / total


def margin(price: pd.Series, tax: pd.Series, flag: pd.Series, count: pd.Series) -> float:
    sales = (price * (1.0 - tax)).sum()
    if sales != 0.0:
        return (sales - (price * tax).sum()) / sales * 100.0
    return 0.0


def spread(price: pd.Series, tax: pd.Series, flag: pd.Series, count: pd.Series) -> float:
    if len(count) < 2:
        return 0.0
    return count.var() + price.std() + tax[tax > 0].max() + price.corr(count) + count.mean()


def unguarded(price: pd.Series, tax: pd.Series, flag: pd.Series, count: pd.Series) -> float:
    return price.sum() / tax[tax > 10.0].count()  # pyright: ignore[reportAttributeAccessIssue]


def peak(price: pd.Series, tax: pd.Series, flag: pd.Series, count: pd.Series) -> float:
    return tax[(tax >= 0) & (tax <= 1)].max() - price.min() + len(flag[count > 2]) + price.max()


CASES = [taxed, mapped, returned, margin, spread, unguarded, peak]


def finish(
    function: Callable[..., object], backend: Backend[Any], rows: Sequence[tuple[object, ...]]
) -> object:
    return run(function, SCHEMA, backend, rows)


def run(
    function: Callable[..., object],
    schema: dict[str, Type],
    backend: Backend[Any],
    rows: Sequence[tuple[object, ...]],
) -> object:
    translation = aggregate(function, schema, backend)
    assert translation.expr is not None, translation.unsupported
    plan = translation.expr
    assert plan.reduced == (function in (returned, peak, highest))
    outputs = backend.evaluate(list(plan.columns), schema, rows, plan.reduced)
    columns = [list(column) for column in zip(*rows, strict=True)]
    return plan.finish([np.asarray(output) for output in outputs], lambda: columns)


def highest(value: pd.Series) -> float:
    return value.max()


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("function", CASES, ids=[case.__name__ for case in CASES])
def test_aggregates_match_pandas(backend: Backend[Any], function: Callable[..., object]) -> None:
    expected = function(*(pd.Series(list(column)) for column in zip(*ROWS, strict=True)))
    item = getattr(expected, "item", None)
    assert same(item() if callable(item) else expected, finish(function, backend, ROWS))


@pytest.mark.parametrize("backend", backend_params())
def test_native_reductions_match_pandas_zeros_and_nan(backend: Backend[Any]) -> None:
    for rows in ([(-0.0,), (0.0,)], [(0.0,), (-0.0,)], [(1.0,), (math.nan,)], [(math.nan,)]):
        expected = highest(pd.Series([row[0] for row in rows]))
        assert same(expected, run(highest, {"value": FLOAT}, backend, rows))


def moments(value: pd.Series, other: pd.Series) -> float:
    return value.sum() + value.mean() + value.var() + value.std() + value.corr(other)


@pytest.mark.parametrize("backend", backend_params())
def test_reductions_skip_nan_like_pandas(backend: Backend[Any]) -> None:
    generator = np.random.default_rng(0)
    for values in (generator.normal(size=1000).tolist(), [1.5, math.nan, -2.25, 4.0]):
        others = [value * 3.0 - index for index, value in enumerate(values)]
        rows = list(zip(values, others, strict=True))
        expected = moments(pd.Series(values), pd.Series(others))
        assert same(expected, run(moments, {"value": FLOAT, "other": FLOAT}, backend, rows))


def test_aggregates_reject_series_results() -> None:
    def shifted(price: pd.Series, tax: pd.Series, flag: pd.Series, count: pd.Series) -> pd.Series:
        return price + 1.0

    def compared(price: pd.Series, tax: pd.Series, flag: pd.Series, count: pd.Series) -> object:
        return (price[tax > 0] > tax).sum()

    for function in (shifted, compared):
        assert not aggregate(function, SCHEMA).ok
