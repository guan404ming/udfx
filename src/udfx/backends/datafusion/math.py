from __future__ import annotations

from dataclasses import replace

from datafusion import Expr, functions, lit

from udfx.backends.datafusion.conversions import float_int
from udfx.backends.datafusion.numbers import (
    INFINITY,
    ORDER,
    cast_float64,
    finite,
    is_nan,
)
from udfx.backends.datafusion.rules import RULES, when
from udfx.backends.registration import (
    Lowered,
    register_extremum,
    register_float_variants,
    register_numpy_math,
)
from udfx.semantics.numbers import (
    RoundPrimitives,
    round_half_even,
    round_overflow_safe,
    scaled_round,
)
from udfx.types import Kind


def half_even(value: Expr) -> Expr:
    return round_half_even(ROUND, value)


def identity(value: Expr) -> Expr:
    return value


ROUND = RoundPrimitives(
    lit, functions.power, Expr.between, functions.abs, cast_float64, functions.round, when
)


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


register_extremum(RULES, "max", functions.greatest, ORDER)
register_extremum(RULES, "min", functions.least, ORDER)

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
    rounding = scaled_round(ROUND, value, digits)
    rounded = half_even(rounding.scaled)
    result = functions.when(rounded == lit(0.0), value * lit(0.0)).otherwise(
        rounded / rounding.scale
    )
    return Lowered(result, rounding.exact)


@RULES("round_overflows", Kind.FLOAT, Kind.INT)
def round_overflows(value: Expr, digits: Expr) -> Lowered[Expr]:
    return Lowered(lit(False), round_overflow_safe(ROUND, value, digits))


register_numpy_math(
    RULES,
    replace(ROUND, round=half_even),
    root=root,
    floor=functions.floor,
    ceil=functions.ceil,
    is_nan=functions.isnan,
    round_digits=round_digits,
)
