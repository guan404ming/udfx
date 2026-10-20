from __future__ import annotations

import operator

from duckdb import Expression

from udfx.backends.base import Lowered
from udfx.backends.duckdb.numbers import either_nan, float_int, floating
from udfx.backends.duckdb.rules import RULES, constant, function, when
from udfx.ops.math import EXACT_POWERS_OF_TEN, HALF_INTEGER_LIMIT, ROUND_OVERFLOW_LIMIT
from udfx.semantics.numbers import Binary
from udfx.types import Kind


def half_even(value: Expression) -> Expression:
    return function("round_even", value, constant(0))


def power(a: Expression, b: Expression) -> Lowered[Expression]:
    complex_result = (a < constant(0.0)) & (function("floor", b) != b)
    return Lowered(function("pow", a, b), ~complex_result)


def pick(name: str, compare: Binary) -> None:
    def ordered(a: Expression, b: Expression) -> Expression:
        return when(compare(b, a) & ~either_nan(a, b), b, a)

    def total(a: Expression, b: Expression) -> Expression:
        return when(compare(b, a), b, a)

    RULES(name, Kind.FLOAT, Kind.FLOAT)(ordered)
    RULES(name, Kind.INT, Kind.INT)(total)
    RULES(name, Kind.STR, Kind.STR)(total)


pick("max", operator.gt)
pick("min", operator.lt)

RULES("__pow__", Kind.FLOAT, Kind.FLOAT)(power)
RULES("__pow__", Kind.INT, Kind.FLOAT)(lambda a, b: power(floating(a), b))
RULES("__pow__", Kind.FLOAT, Kind.INT)(lambda a, b: power(a, floating(b)))
RULES("round", Kind.FLOAT)(lambda a: float_int(half_even(a)))


@RULES("pow_overflows", Kind.FLOAT, Kind.FLOAT)
def pow_overflows(a: Expression, b: Expression) -> Lowered[Expression]:
    result = power(a, b)
    finite = function("isfinite", a) & function("isfinite", b) & (a != constant(0.0))
    return Lowered(function("isinf", result.expr) & finite, result.exact)


@RULES("round", Kind.FLOAT, Kind.INT)
def round_digits(a: Expression, digits: Expression) -> Lowered[Expression]:
    scale = function("pow", constant(10.0), floating(digits))
    scaled = a * scale
    tie = scaled - function("floor", scaled) == constant(0.5)
    exact = (
        digits.between(constant(0), constant(EXACT_POWERS_OF_TEN))
        & (function("abs", scaled) < constant(HALF_INTEGER_LIMIT))
        & ((digits == constant(0)) | ~tie)
    )
    return Lowered(half_even(scaled) / scale, exact)


@RULES("round_overflows", Kind.FLOAT, Kind.INT)
def round_overflows(a: Expression, digits: Expression) -> Lowered[Expression]:
    safe = (digits >= constant(0)) | (function("abs", a) < constant(ROUND_OVERFLOW_LIMIT))
    return Lowered(constant(False), safe)
