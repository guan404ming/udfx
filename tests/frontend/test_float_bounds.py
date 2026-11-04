import math
from collections.abc import Callable

import pytest

from udfx.backends.engines.polars import PolarsBackend
from udfx.ir import Col, lit
from udfx.ops import apply
from udfx.semantics.numbers.comparisons import float_fact
from udfx.testing import check_function
from udfx.types import FLOAT

POLARS = PolarsBackend()
VALUES = [0.0, -0.0, 1.0, -1.0, 2.5, 1e-300, 1e200, -1e200, math.inf, -math.inf, math.nan]


def projection(x: float, y: float) -> float:
    angle = math.atan2(y, x)
    return round(math.sqrt((x - math.cos(angle)) ** 2.0 + (y - math.sin(angle)) ** 2.0), 4)


def clamped(x: float, y: float) -> float:
    total = x**2.0 + y**2.0
    root = math.sqrt(total - 1.0) if total >= 1.0 else 0.0
    return math.acos(min(1.0, root / 1e-6))


def ordered(x: float, y: float) -> bool:
    return min(1.0, x) <= 1.0 and max(x * y, -2.0) >= -2.0 and abs(x - y) >= 0.0


@pytest.mark.parametrize("function", [projection, clamped, ordered])
def test_float_bounds_match_python(function: Callable[..., object]) -> None:
    rows = [(x, y) for x in VALUES for y in VALUES]
    assert check_function(function, {"x": FLOAT, "y": FLOAT}, POLARS, rows) == []


def test_float_bounds_keep_undefined_values() -> None:
    first, second = (apply("math.exp", Col(name, FLOAT)) for name in ("x", "y"))
    difference = apply("__sub__", first, second)
    assert float_fact(apply("math.isnan", difference)) is None
    assert float_fact(apply("math.isinf", apply("math.atan2", first, second))) is False


def test_float_bounds_divide_by_signed_intervals() -> None:
    root = apply("math.sqrt", Col("x", FLOAT))
    clamped = apply("min", apply("__truediv__", root, lit(1e-6)), lit(1.0))
    assert float_fact(apply("__lt__", clamped, lit(-1.0))) is False
    spanning = apply("__truediv__", root, apply("math.sin", Col("y", FLOAT)))
    assert float_fact(apply("__lt__", spanning, lit(-1.0))) is None
