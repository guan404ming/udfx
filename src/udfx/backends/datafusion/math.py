from __future__ import annotations

import math
from dataclasses import replace

import pyarrow as pa
from datafusion import Expr, ScalarUDF, functions, lit, udf

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
    register_math,
    register_trigonometry,
)
from udfx.semantics.math import power_hazards
from udfx.semantics.numbers import (
    RoundPrimitives,
    round_half_even,
    round_overflow_safe,
    round_scaled,
)
from udfx.semantics.scalars import numpy_values
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


def power(base: Expr, exponent: Expr | float) -> Lowered[Expr]:
    negative, fractional = power_hazards(exponent)
    exponent = exponent if isinstance(exponent, Expr) else lit(exponent)
    value = functions.power(base, exponent)
    if negative:
        undefined = (base == lit(0.0)) & below_zero(exponent)
        safe_base = functions.when(undefined, lit(1.0)).otherwise(base)
        power_value = functions.power(safe_base, exponent)
        value = functions.when(undefined, lit(INFINITY)).otherwise(power_value)
    if not fractional:
        return Lowered(value)
    return Lowered(value, ~(below_zero(base) & (functions.floor(exponent) != exponent)))


def root(value: Expr) -> Expr:
    return functions.sqrt(functions.when(below_zero(value), lit(0.0)).otherwise(value))


register_extremum(RULES, "max", functions.greatest, ORDER, when)
register_extremum(RULES, "min", functions.least, ORDER, when)

register_float_variants(RULES, "__pow__", power, cast_float64, literals=(1,))
RULES("math.sqrt", Kind.FLOAT)(root)
RULES("math.sqrt", Kind.INT)(lambda value: root(cast_float64(value)))
RULES("math.ceil", Kind.FLOAT)(lambda value: float_int(functions.ceil(value)))
RULES("math.floor", Kind.FLOAT)(lambda value: float_int(functions.floor(value)))
RULES("math.ceil", Kind.INT)(identity)
RULES("math.floor", Kind.INT)(identity)
RULES("round", Kind.FLOAT)(lambda value: float_int(half_even(value)))


@RULES("pow_overflows", Kind.FLOAT, Kind.FLOAT, literals=(1,))
def pow_overflows(base: Expr, exponent: Expr | float) -> Lowered[Expr]:
    result = power(base, exponent)
    overflow = (functions.abs(result.expr) == lit(INFINITY)) & finite(base) & (base != lit(0.0))
    if isinstance(exponent, Expr):
        return Lowered(overflow & finite(exponent), result.exact)
    return Lowered(overflow if math.isfinite(exponent) else lit(False), result.exact)


@RULES("round", Kind.FLOAT, Kind.INT, literals=(1,))
def round_digits(value: Expr, digits: Expr | int) -> Lowered[Expr]:
    rounding = round_scaled(ROUND, value, digits, half_even)
    rounded = rounding.scaled
    result = functions.when(rounded == lit(0.0), value * lit(0.0)).otherwise(
        rounded / rounding.scale
    )
    return Lowered(result, rounding.exact)


@RULES("round_overflows", Kind.FLOAT, Kind.INT, literals=(1,))
def round_overflows(value: Expr, digits: Expr | int) -> Lowered[Expr]:
    return Lowered(lit(False), round_overflow_safe(ROUND, value, digits))


def numpy_udf(name: str) -> ScalarUDF:
    def run(values: pa.Array) -> pa.Array:
        return pa.array(numpy_values(name, values.to_numpy(zero_copy_only=False)), pa.float64())

    return udf(run, [pa.float64()], pa.float64(), "immutable", f"numpy {name}")


register_math(
    RULES,
    replace(ROUND, round=half_even),
    root=root,
    floor=functions.floor,
    ceil=functions.ceil,
    is_nan=functions.isnan,
    round_digits=round_digits,
    numpy_values=lambda name, value: numpy_udf(name)(value),
)


register_trigonometry(
    RULES,
    ROUND,
    unary={"sin": functions.sin, "cos": functions.cos, "acos": functions.acos},
    atan2=functions.atan2,
    is_nan=is_nan,
)
