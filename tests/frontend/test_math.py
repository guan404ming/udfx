import math
from collections.abc import Callable

import pytest

from udfx.backends.polars import PolarsBackend
from udfx.frontend import trace
from udfx.ir import Call
from udfx.testing import check_function
from udfx.types import FLOAT, INT, STR, Type

POLARS = PolarsBackend()


def power(a: float, b: float) -> float:
    return a**b


def larger(a: int, b: int) -> int:
    return max(a, b)


def smaller(a: str, b: str) -> str:
    return min(a, b)


def nearest(x: float) -> int:
    return round(x)


def cents(x: float, digits: int) -> float:
    return round(x, digits)


def ceiling(x: float) -> int:
    return math.ceil(x)


def flooring(x: int) -> int:
    return math.floor(x)


def root(x: float) -> float:
    return math.sqrt(x)


CASES: list[tuple[Callable[..., object], dict[str, Type], str]] = [
    (power, {"a": FLOAT, "b": FLOAT}, "__pow__"),
    (power, {"a": INT, "b": FLOAT}, "__pow__"),
    (power, {"a": FLOAT, "b": INT}, "__pow__"),
    (larger, {"a": INT, "b": INT}, "max"),
    (smaller, {"a": STR, "b": STR}, "min"),
    (nearest, {"x": FLOAT}, "round"),
    (cents, {"x": FLOAT, "digits": INT}, "round"),
    (ceiling, {"x": FLOAT}, "math.ceil"),
    (flooring, {"x": INT}, "math.floor"),
    (root, {"x": FLOAT}, "math.sqrt"),
]


@pytest.mark.parametrize(("function", "schema", "name"), CASES)
def test_call_becomes_op(
    function: Callable[..., object], schema: dict[str, Type], name: str
) -> None:
    program = trace(function, schema)
    assert isinstance(program.value, Call)
    assert program.value.op.name == name


def test_complex_power_raises() -> None:
    rows = [(-8.0, 0.5), (2.0, 0.5)]
    assert check_function(power, {"a": FLOAT, "b": FLOAT}, POLARS, rows) == []
