from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

FLOAT_INT_MAX = 2**53
EXACT_POWERS_OF_TEN = 22
HALF_INTEGER_LIMIT = 2.0**52
ROUND_OVERFLOW_LIMIT = 1e308
REPR_FIXED_MIN = 1e-4
REPR_FIXED_MAX = 1e16

Binary = Callable[[Any, Any], Any]

ARITHMETIC: dict[str, Binary] = {
    "__add__": operator.add,
    "__sub__": operator.sub,
    "__mul__": operator.mul,
}

COMPARISONS: dict[str, Binary] = {
    "__lt__": operator.lt,
    "__le__": operator.le,
    "__gt__": operator.gt,
    "__ge__": operator.ge,
}


@dataclass(frozen=True)
class FloatPrimitives:
    when: Callable[[Any, Any, Any], Any]
    floor: Callable[[Any], Any]
    sign: Callable[[Any], Any]
    truncated_remainder: Callable[[Any, Any], tuple[Any, Any | None]]


def float_divmod(primitives: FloatPrimitives, left: Any, right: Any) -> tuple[Any, Any, Any]:
    remainder, exact = primitives.truncated_remainder(left, right)
    quotient = (left - remainder) / right
    adjust = (remainder != 0.0) & ((right < 0.0) != (remainder < 0.0))
    remainder = primitives.when(
        remainder == 0.0,
        primitives.sign(right) * 0.0,
        primitives.when(adjust, remainder + right, remainder),
    )
    quotient = primitives.when(adjust, quotient - 1.0, quotient)
    floored = primitives.floor(quotient)
    snapped = primitives.when(quotient - floored > 0.5, floored + 1.0, floored)
    quotient = primitives.when(quotient != 0.0, snapped, left / right * 0.0)
    return quotient, remainder, exact
