from __future__ import annotations

import operator

from datafusion import Expr, functions, lit

from udfx.backends.base import Lowered, register_float_variants
from udfx.backends.datafusion.conversions import float_int
from udfx.backends.datafusion.numbers import (
    INFINITY,
    cast_float64,
    either_nan,
    finite,
    is_nan,
)
from udfx.backends.datafusion.rules import RULES
from udfx.semantics.numbers import (
    EXACT_POWERS_OF_TEN,
    HALF_INTEGER_LIMIT,
    ROUND_OVERFLOW_LIMIT,
    Binary,
)
from udfx.types import Kind


def half_even(value: Expr) -> Expr:
    floored = functions.floor(value)
    fraction = value - floored
    odd = floored % lit(2.0) != lit(0.0)
    up = (fraction > lit(0.5)) | ((fraction == lit(0.5)) & odd)
    return functions.when(up, floored + lit(1.0)).otherwise(floored)


def identity(value: Expr) -> Expr:
    return value


def below_zero(value: Expr) -> Expr:
    return (value < lit(0.0)) & ~is_nan(value)


def power(base: Expr, exponent: Expr) -> Lowered[Expr]:
    undefined = (base == lit(0.0)) & below_zero(exponent)
    safe_base = functions.when(undefined, lit(1.0)).otherwise(base)
    value = functions.when(undefined, lit(INFINITY)).otherwise(functions.power(safe_base, exponent))
    complex_result = below_zero(base) & (functions.floor(exponent) != exponent)
    return Lowered(value, ~complex_result)


def root(value: Expr) -> Expr:
    return functions.sqrt(functions.when(below_zero(value), lit(0.0)).otherwise(value))


def register_extremum(name: str, compare: Binary) -> None:
    def ordered(left: Expr, right: Expr) -> Expr:
        return functions.when(compare(right, left) & ~either_nan(left, right), right).otherwise(
            left
        )

    def total(left: Expr, right: Expr) -> Expr:
        return functions.when(compare(right, left), right).otherwise(left)

    RULES(name, Kind.FLOAT, Kind.FLOAT)(ordered)
    RULES(name, Kind.INT, Kind.INT)(total)
    RULES(name, Kind.STR, Kind.STR)(total)


register_extremum("max", operator.gt)
register_extremum("min", operator.lt)

register_float_variants(RULES, "__pow__", power, cast_float64)
RULES("math.sqrt", Kind.FLOAT)(root)
RULES("math.sqrt", Kind.INT)(lambda value: root(cast_float64(value)))
RULES("math.ceil", Kind.FLOAT)(lambda value: float_int(functions.ceil(value)))
RULES("math.floor", Kind.FLOAT)(lambda value: float_int(functions.floor(value)))
RULES("math.ceil", Kind.INT)(identity)
RULES("math.floor", Kind.INT)(identity)
RULES("round", Kind.FLOAT)(lambda value: float_int(half_even(value)))


@RULES("pow_overflows", Kind.FLOAT, Kind.FLOAT)
def pow_overflows(base: Expr, exponent: Expr) -> Lowered[Expr]:
    result = power(base, exponent)
    overflow = functions.abs(result.expr) == lit(INFINITY)
    return Lowered(overflow & finite(base) & finite(exponent) & (base != lit(0.0)), result.exact)


@RULES("round", Kind.FLOAT, Kind.INT)
def round_digits(value: Expr, digits: Expr) -> Lowered[Expr]:
    scale = functions.power(lit(10.0), cast_float64(digits))
    scaled = value * scale
    rounded = half_even(scaled)
    tie = scaled - functions.floor(scaled) == lit(0.5)
    exact = (
        digits.between(lit(0), lit(EXACT_POWERS_OF_TEN))
        & (functions.abs(scaled) < lit(HALF_INTEGER_LIMIT))
        & ((digits == lit(0)) | ~tie)
    )
    result = functions.when(rounded == lit(0.0), value * lit(0.0)).otherwise(rounded / scale)
    return Lowered(result, exact)


@RULES("round_overflows", Kind.FLOAT, Kind.INT)
def round_overflows(value: Expr, digits: Expr) -> Lowered[Expr]:
    safe = (digits >= lit(0)) | (functions.abs(value) < lit(ROUND_OVERFLOW_LIMIT))
    return Lowered(lit(False), safe)
