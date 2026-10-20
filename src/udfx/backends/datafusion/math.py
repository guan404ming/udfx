from __future__ import annotations

import operator

from datafusion import Expr, functions, lit

from udfx.backends.base import Lowered
from udfx.backends.datafusion.numbers import INFINITY, finite, float_int, floating, is_nan
from udfx.backends.datafusion.rules import RULES
from udfx.ops.math import EXACT_POWERS_OF_TEN, HALF_INTEGER_LIMIT, ROUND_OVERFLOW_LIMIT
from udfx.semantics.numbers import Binary
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


def power(a: Expr, b: Expr) -> Lowered[Expr]:
    undefined = (a == lit(0.0)) & below_zero(b)
    base = functions.when(undefined, lit(1.0)).otherwise(a)
    value = functions.when(undefined, lit(INFINITY)).otherwise(functions.power(base, b))
    complex_result = below_zero(a) & (functions.floor(b) != b)
    return Lowered(value, ~complex_result)


def root(a: Expr) -> Expr:
    return functions.sqrt(functions.when(below_zero(a), lit(0.0)).otherwise(a))


def pick(name: str, compare: Binary) -> None:
    def ordered(a: Expr, b: Expr) -> Expr:
        return functions.when(compare(b, a) & ~is_nan(a) & ~is_nan(b), b).otherwise(a)

    def total(a: Expr, b: Expr) -> Expr:
        return functions.when(compare(b, a), b).otherwise(a)

    RULES(name, Kind.FLOAT, Kind.FLOAT)(ordered)
    RULES(name, Kind.INT, Kind.INT)(total)
    RULES(name, Kind.STR, Kind.STR)(total)


pick("max", operator.gt)
pick("min", operator.lt)

RULES("__pow__", Kind.FLOAT, Kind.FLOAT)(power)
RULES("__pow__", Kind.INT, Kind.FLOAT)(lambda a, b: power(floating(a), b))
RULES("__pow__", Kind.FLOAT, Kind.INT)(lambda a, b: power(a, floating(b)))
RULES("math.sqrt", Kind.FLOAT)(root)
RULES("math.sqrt", Kind.INT)(lambda a: root(floating(a)))
RULES("math.ceil", Kind.FLOAT)(lambda a: float_int(functions.ceil(a)))
RULES("math.floor", Kind.FLOAT)(lambda a: float_int(functions.floor(a)))
RULES("math.ceil", Kind.INT)(identity)
RULES("math.floor", Kind.INT)(identity)
RULES("round", Kind.FLOAT)(lambda a: float_int(half_even(a)))


@RULES("pow_overflows", Kind.FLOAT, Kind.FLOAT)
def pow_overflows(a: Expr, b: Expr) -> Lowered[Expr]:
    result = power(a, b)
    overflow = functions.abs(result.expr) == lit(INFINITY)
    return Lowered(overflow & finite(a) & finite(b) & (a != lit(0.0)), result.exact)


@RULES("round", Kind.FLOAT, Kind.INT)
def round_digits(a: Expr, digits: Expr) -> Lowered[Expr]:
    scale = functions.power(lit(10.0), floating(digits))
    scaled = a * scale
    rounded = half_even(scaled)
    tie = scaled - functions.floor(scaled) == lit(0.5)
    exact = (
        digits.between(lit(0), lit(EXACT_POWERS_OF_TEN))
        & (functions.abs(scaled) < lit(HALF_INTEGER_LIMIT))
        & ((digits == lit(0)) | ~tie)
    )
    value = functions.when(rounded == lit(0.0), a * lit(0.0)).otherwise(rounded / scale)
    return Lowered(value, exact)


@RULES("round_overflows", Kind.FLOAT, Kind.INT)
def round_overflows(a: Expr, digits: Expr) -> Lowered[Expr]:
    safe = (digits >= lit(0)) | (functions.abs(a) < lit(ROUND_OVERFLOW_LIMIT))
    return Lowered(lit(False), safe)
