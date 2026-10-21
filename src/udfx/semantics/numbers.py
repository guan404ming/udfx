from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

from udfx.types import INT64_MIN

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


@dataclass(frozen=True)
class OrderPrimitives:
    when: Callable[[Any, Any, Any], Any]
    either_nan: Callable[[Any, Any], Any]
    is_null: Callable[[Any], Any]
    within_exact_float_range: Callable[[Any], Any]


def float_compare(primitives: OrderPrimitives, compare: Binary, left: Any, right: Any) -> Any:
    return compare(left, right) & ~primitives.either_nan(left, right)


def widened_exact(primitives: OrderPrimitives, value: Any) -> Any:
    return primitives.is_null(value) | primitives.within_exact_float_range(value)


def float_extremum(
    primitives: OrderPrimitives, pick: Binary, left: Any, right: Any
) -> tuple[Any, Any]:
    zeros = (left == 0.0) & (right == 0.0)
    return pick(left, right), ~primitives.either_nan(left, right) & ~zeros


@dataclass(frozen=True)
class RoundPrimitives:
    literal: Callable[[object], Any]
    power: Callable[[Any, Any], Any]
    floor: Callable[[Any], Any]
    abs: Callable[[Any], Any]
    to_float: Callable[[Any], Any]


class ScaledRound(NamedTuple):
    scale: Any
    scaled: Any
    exact: Any


def scaled_round(primitives: RoundPrimitives, value: Any, digits: Any) -> ScaledRound:
    literal = primitives.literal
    scale = primitives.power(literal(10.0), primitives.to_float(digits))
    scaled = value * scale
    tie = scaled - primitives.floor(scaled) == literal(0.5)
    exact = (
        (digits >= literal(0))
        & (digits <= literal(EXACT_POWERS_OF_TEN))
        & (primitives.abs(scaled) < literal(HALF_INTEGER_LIMIT))
        & ((digits == literal(0)) | ~tie)
    )
    return ScaledRound(scale, scaled, exact)


def round_overflow_safe(primitives: RoundPrimitives, value: Any, digits: Any) -> Any:
    literal = primitives.literal
    return (digits >= literal(0)) | (primitives.abs(value) < literal(ROUND_OVERFLOW_LIMIT))


@dataclass(frozen=True)
class IntegerPrimitives:
    literal: Callable[[object], Any]
    when: Callable[[Any, Any, Any], Any]
    truncated_divide: Callable[[Any, Any], Any]


def division_overflows(literal: Callable[[object], Any], left: Any, right: Any) -> Any:
    return (left == literal(INT64_MIN)) & (right == literal(-1))


def int_divmod(primitives: IntegerPrimitives, left: Any, right: Any) -> tuple[Any, Any, Any]:
    literal, when = primitives.literal, primitives.when
    overflow = division_overflows(literal, left, right)
    divisor = when((right == literal(0)) | overflow, literal(1), right)
    quotient = primitives.truncated_divide(left, divisor)
    remainder = left - quotient * divisor
    adjust = (remainder != literal(0)) & ((remainder < literal(0)) != (divisor < literal(0)))
    return (
        when(adjust, quotient - literal(1), quotient),
        when(adjust, remainder + divisor, remainder),
        ~overflow,
    )
