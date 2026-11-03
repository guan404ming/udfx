import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
import pytest
from conftest import backend_params

from udfx import compile
from udfx.backends.base import Backend
from udfx.frontend import trace
from udfx.testing.outcome import same
from udfx.types import FLOAT, INT, STR, Type


def check(
    function: Callable[..., object],
    schema: dict[str, Type],
    backend: Backend[Any],
    rows: Sequence[tuple[object, ...]],
) -> None:
    translation = compile(function, schema, backend, series=True)
    assert translation.expr is not None, translation.unsupported
    columns = [pd.Series(list(column)) for column in zip(*rows, strict=True)]
    expected = pd.Series(function(*columns)).tolist()
    (actual,) = backend.evaluate([translation.expr], schema, rows)
    assert len(actual) == len(expected)
    assert all(same(left, right) for left, right in zip(expected, actual, strict=True))


def priced(price: pd.Series, mode: pd.Series) -> pd.Series:
    rates = {"AIR": 0.08, "MAIL": 0.09}
    results = []
    for value, name in zip(price, mode, strict=False):
        if value < 0.0:
            summary = "Invalid"
        else:
            rate = rates.get(name, 0.0)
            summary = name + ": " + str(value * (1 + rate))
        results.append(summary)
    return pd.Series(results)


def sigmoid(price: pd.Series) -> pd.Series:
    values = []
    for value in price:
        scaled = 1 / (1 + np.exp(-value))
        values.append(scaled)
    return pd.Series(values)


def redacted(comment: pd.Series) -> pd.Series:
    words = ["bad", "good"]
    results = []
    for text in comment:
        cleaned = text
        for word in words:
            cleaned = cleaned.replace(word, "*")
        results.append(cleaned + str(sum(cleaned.count(word) for word in words)))
    return pd.Series(results)


def distance(first: pd.Series, second: pd.Series) -> pd.Series:
    return pd.Series(np.sqrt(np.power(np.abs(first - second), 2) + np.minimum(first, second)))


def ratio(first: pd.Series, second: pd.Series) -> pd.Series:
    return first.add(-second).abs() / second


def wrapped(first: pd.Series, second: pd.Series) -> pd.Series:
    return first * 3 - (-first).abs() + second**2


def summed(first: pd.Series, second: pd.Series) -> pd.Series:
    return first + second


def cleaned(comment: pd.Series) -> pd.Series:
    stripped = comment.str.strip().str.upper()
    return stripped + comment.str.len().astype(str) + comment.str.slice(1, 4)


def searched(comment: pd.Series) -> pd.Series:
    found = comment.str.contains("ab+c") | comment.str.startswith("x")
    return found & ~comment.str.contains("+", regex=False)


def substituted(comment: pd.Series) -> pd.Series:
    replaced = comment.str.replace("a", "b").str.replace(r"\d+", "#", regex=True)
    return replaced + comment.str.count("[aeiou]").astype(str)


def bounded(price: pd.Series) -> pd.Series:
    return price.fillna(0).round(1).clip(-5, 5) + price.where(price > 0).fillna(-1.0)


def flagged(price: pd.Series) -> pd.Series:
    return pd.isna(price) | price.between(0, 1, inclusive="left") | price.notna().eq(False)


def chosen(price: pd.Series, quantity: pd.Series) -> pd.Series:
    selected = pd.Series(np.where(price > 0, price, quantity))
    return selected + price.mask(price < 1, 3) + quantity.rsub(10).astype(float)


def compared(first: pd.Series, second: pd.Series) -> pd.Series:
    return pd.Series(np.clip(first, 0, 10)).gt(second) & first.ne(3)


PRICES = [(math.nan,), (math.inf,), (-math.inf,), (-0.0,), (2.25,), (0.5,), (1e300,), (-7.0,)]
TEXTS = [("  ab c ",), ("xabbbc+12",), ("",), ("a1b22 é",), ("ß+",)]

CASES = [
    (priced, {"price": FLOAT, "mode": STR}, [(10.0, "AIR"), (-1.0, "MAIL"), (2.5, "SHIP")]),
    (sigmoid, {"price": FLOAT}, [(900.0,), (0.5,), (-3.0,), (0.0,)]),
    (redacted, {"comment": STR}, [("bad good",), ("",), ("goodbad bad",)]),
    (distance, {"first": INT, "second": INT}, [(3, 7), (2**62, -(2**62)), (-5, 0)]),
    (ratio, {"first": INT, "second": INT}, [(3, 2), (4, 0), (0, 0), (2**63 - 1, -1)]),
    (wrapped, {"first": INT, "second": INT}, [(2**63 - 1, 1), (-(2**63), -(2**63)), (3, -4)]),
    (summed, {"first": INT, "second": INT}, [(2**63 - 1, 1), (1, 2)]),
    (cleaned, {"comment": STR}, TEXTS),
    (searched, {"comment": STR}, TEXTS),
    (substituted, {"comment": STR}, TEXTS),
    (bounded, {"price": FLOAT}, PRICES),
    (flagged, {"price": FLOAT}, PRICES),
    (chosen, {"price": FLOAT, "quantity": INT}, [(1.5, 2), (-1.0, 4), (math.nan, -3), (0.5, 0)]),
    (compared, {"first": INT, "second": INT}, [(3, 2), (-4, -1), (12, 9), (7, 7)]),
]


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize(
    ("function", "schema", "rows"), CASES, ids=[case[0].__name__ for case in CASES]
)
def test_series_functions_match_pandas(
    backend: Backend[Any],
    function: Callable[..., object],
    schema: dict[str, Type],
    rows: Sequence[tuple[object, ...]],
) -> None:
    check(function, schema, backend, rows)


def test_series_loops_become_row_expressions() -> None:
    program = trace(sigmoid, {"price": FLOAT}, pandas="scalar")
    assert program.series is pd.Series
    assert program.run([(900.0,), (0.0,)]) == [1.0, 0.5]
    assert program.batch([pd.Series([900.0, 0.0])]).tolist() == [1.0, 0.5]


def test_series_rejects_batch_dependent_functions() -> None:
    def scaled(values: pd.Series) -> pd.Series:
        return pd.Series([value / len(values) for value in values])

    def positions(values: pd.Series) -> pd.Series:
        return pd.Series([index for index, _ in enumerate(values)])

    def total(values: pd.Series) -> float:
        return sum(values)

    for function in (scaled, positions, total):
        assert not compile(function, {"values": FLOAT}, series=True).ok


def test_series_rejects_nullable_columns() -> None:
    translation = compile(sigmoid, {"price": FLOAT.or_none()}, series=True)
    assert "non-null" in str(translation.unsupported)


@pytest.mark.parametrize("backend", backend_params())
def test_series_faults_raise_python_exceptions(backend: Backend[Any]) -> None:
    def rated(mode: pd.Series) -> pd.Series:
        rates = {"AIR": 1.5}
        return pd.Series([rates[name] for name in mode])

    check(rated, {"mode": STR}, backend, [("AIR",), ("AIR",)])
    translation = compile(rated, {"mode": STR}, backend, series=True)
    with pytest.raises(KeyError):
        backend.evaluate([translation.expr], {"mode": STR}, [("AIR",), ("SHIP",)])
