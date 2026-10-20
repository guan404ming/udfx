from __future__ import annotations

import operator

from duckdb import Expression

from udfx.backends.base import Lowered
from udfx.backends.duckdb.conversions import float_int
from udfx.backends.duckdb.numbers import cast_float64, either_nan
from udfx.backends.duckdb.rules import RULES, constant, function, when
from udfx.semantics.numbers import (
    EXACT_POWERS_OF_TEN,
    HALF_INTEGER_LIMIT,
    ROUND_OVERFLOW_LIMIT,
    Binary,
)
from udfx.types import Kind


def half_even(value: Expression) -> Expression:
    return function("round_even", value, constant(0))


def identity(value: Expression) -> Expression:
    return value


def power(base: Expression, exponent: Expression) -> Lowered[Expression]:
    complex_result = (base < constant(0.0)) & (function("floor", exponent) != exponent)
    return Lowered(function("pow", base, exponent), ~complex_result)


def root(value: Expression) -> Expression:
    return function("sqrt", when(value < constant(0.0), constant(0.0), value))


def register_extremum(name: str, compare: Binary) -> None:
    def ordered(left: Expression, right: Expression) -> Expression:
        return when(compare(right, left) & ~either_nan(left, right), right, left)

    def total(left: Expression, right: Expression) -> Expression:
        return when(compare(right, left), right, left)

    RULES(name, Kind.FLOAT, Kind.FLOAT)(ordered)
    RULES(name, Kind.INT, Kind.INT)(total)
    RULES(name, Kind.STR, Kind.STR)(total)


register_extremum("max", operator.gt)
register_extremum("min", operator.lt)

RULES("__pow__", Kind.FLOAT, Kind.FLOAT)(power)
RULES("__pow__", Kind.INT, Kind.FLOAT)(lambda base, exponent: power(cast_float64(base), exponent))
RULES("__pow__", Kind.FLOAT, Kind.INT)(lambda base, exponent: power(base, cast_float64(exponent)))
RULES("math.sqrt", Kind.FLOAT)(root)
RULES("math.sqrt", Kind.INT)(lambda value: root(cast_float64(value)))
RULES("math.ceil", Kind.FLOAT)(lambda value: float_int(function("ceil", value)))
RULES("math.floor", Kind.FLOAT)(lambda value: float_int(function("floor", value)))
RULES("math.ceil", Kind.INT)(identity)
RULES("math.floor", Kind.INT)(identity)
RULES("round", Kind.FLOAT)(lambda value: float_int(half_even(value)))


@RULES("pow_overflows", Kind.FLOAT, Kind.FLOAT)
def pow_overflows(base: Expression, exponent: Expression) -> Lowered[Expression]:
    result = power(base, exponent)
    finite = function("isfinite", base) & function("isfinite", exponent) & (base != constant(0.0))
    return Lowered(function("isinf", result.expr) & finite, result.exact)


@RULES("round", Kind.FLOAT, Kind.INT)
def round_digits(value: Expression, digits: Expression) -> Lowered[Expression]:
    scale = function("pow", constant(10.0), cast_float64(digits))
    scaled = value * scale
    tie = scaled - function("floor", scaled) == constant(0.5)
    exact = (
        digits.between(constant(0), constant(EXACT_POWERS_OF_TEN))
        & (function("abs", scaled) < constant(HALF_INTEGER_LIMIT))
        & ((digits == constant(0)) | ~tie)
    )
    return Lowered(half_even(scaled) / scale, exact)


@RULES("round_overflows", Kind.FLOAT, Kind.INT)
def round_overflows(value: Expression, digits: Expression) -> Lowered[Expression]:
    safe = (digits >= constant(0)) | (function("abs", value) < constant(ROUND_OVERFLOW_LIMIT))
    return Lowered(constant(False), safe)
