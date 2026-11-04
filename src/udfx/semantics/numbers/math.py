from __future__ import annotations

import math
import sys
from collections.abc import Callable
from typing import Any

from udfx.semantics.numbers.operations import RoundPrimitives

EXPONENT_UNDERFLOW = -746.0
EXPONENT_OVERFLOW = 710.0
EXPONENTIAL_LIMIT = math.log(sys.float_info.max)
SMALL_ANGLE = 2.0**-27
POWER_FINITE_BOUND = 1e308


def power_hazards(exponent: object) -> tuple[bool, bool]:
    if not isinstance(exponent, float):
        return True, True
    return exponent < 0.0, not (math.isinf(exponent) or exponent.is_integer())


def trigonometric_argument(
    primitives: RoundPrimitives, is_nan: Callable[[Any], Any], name: str, value: Any
) -> Any:
    magnitude = primitives.abs(value)
    invalid = (
        (magnitude > primitives.literal(1.0)) & ~is_nan(value)
        if name == "acos"
        else magnitude == primitives.literal(math.inf)
    )
    return primitives.when(invalid, primitives.literal(0.0), value)


def guarded_trigonometric(
    primitives: RoundPrimitives,
    is_nan: Callable[[Any], Any],
    name: str,
    value: Any,
    native: Any,
) -> tuple[Any, Any]:
    magnitude = primitives.abs(value)
    if name == "acos":
        return native, (
            (magnitude >= primitives.literal(1.0))
            | (value == primitives.literal(0.0))
            | is_nan(value)
        )
    small = magnitude <= primitives.literal(SMALL_ANGLE)
    return primitives.when(small, value if name == "sin" else primitives.literal(1.0), native), (
        small | (magnitude == primitives.literal(math.inf)) | is_nan(value)
    )


def guarded_atan2(
    primitives: RoundPrimitives,
    is_nan: Callable[[Any], Any],
    native: Callable[[Any, Any], Any],
    vertical: Any,
    horizontal: Any,
) -> tuple[Any, Any]:
    zero = primitives.literal(0.0)
    infinity = primitives.literal(math.inf)
    vertical_size, horizontal_size = primitives.abs(vertical), primitives.abs(horizontal)
    positive_axis = (vertical == zero) & (horizontal > zero) & ~is_nan(horizontal)
    exact = (
        is_nan(vertical)
        | is_nan(horizontal)
        | positive_axis
        | (
            (vertical != zero)
            & (
                (horizontal == zero)
                | (vertical_size == infinity)
                | (horizontal_size == infinity)
                | (vertical_size == horizontal_size)
            )
        )
    )
    return primitives.when(positive_axis, vertical, native(vertical, horizontal)), exact


def transcendental(
    primitives: RoundPrimitives, is_nan: Callable[[Any], Any], name: str, value: Any
) -> tuple[Any, Any]:
    zero = primitives.literal(0.0)
    one = primitives.literal(1.0)
    infinity = primitives.literal(math.inf)
    missing = is_nan(value)
    if name in {"log", "log2", "log10", "log1p"}:
        boundary = primitives.literal(-1.0) if name == "log1p" else zero
        neutral = zero if name == "log1p" else one
        exact = (value <= boundary) | (value == neutral) | (value == infinity) | missing
        result = primitives.when(
            value == boundary,
            primitives.literal(-math.inf),
            primitives.when(
                (value < boundary) | missing,
                primitives.literal(math.nan),
                primitives.when(value == neutral, value if name == "log1p" else zero, value),
            ),
        )
        return result, exact
    underflow = (value < primitives.literal(EXPONENT_UNDERFLOW)) & ~missing
    overflow = (value > primitives.literal(EXPONENT_OVERFLOW)) & ~missing
    exact = (value == zero) | underflow | overflow | missing
    result = primitives.when(
        value == zero,
        value if name == "expm1" else one,
        primitives.when(
            underflow,
            primitives.literal(-1.0) if name == "expm1" else zero,
            primitives.when(overflow, infinity, value),
        ),
    )
    return result, exact
