from __future__ import annotations

import math
from dataclasses import replace

from duckdb import Expression

from udfx.backends.duckdb.conversions import float_int
from udfx.backends.duckdb.numbers import ORDER, cast_float64
from udfx.backends.duckdb.rules import RULES, constant, function, when
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
from udfx.types import Kind


def half_even(value: Expression) -> Expression:
    return round_half_even(ROUND, value)


def identity(value: Expression) -> Expression:
    return value


ROUND = RoundPrimitives(
    constant,
    lambda base, exponent: function("pow", base, exponent),
    Expression.between,
    lambda value: function("abs", value),
    cast_float64,
    lambda value: function("round", value),
    when,
)


def power(base: Expression, exponent: Expression | float) -> Lowered[Expression]:
    fractional = power_hazards(exponent)[1]
    exponent = exponent if isinstance(exponent, Expression) else constant(exponent)
    value = function("pow", base, exponent)
    if not fractional:
        return Lowered(value)
    return Lowered(value, ~((base < constant(0.0)) & (function("floor", exponent) != exponent)))


def root(value: Expression) -> Expression:
    return function("sqrt", when(value < constant(0.0), constant(0.0), value))


register_extremum(RULES, "max", lambda left, right: function("greatest", left, right), ORDER, when)
register_extremum(RULES, "min", lambda left, right: function("least", left, right), ORDER, when)

register_float_variants(RULES, "__pow__", power, cast_float64, literals=(1,))
RULES("math.sqrt", Kind.FLOAT)(root)
RULES("math.sqrt", Kind.INT)(lambda value: root(cast_float64(value)))
RULES("math.ceil", Kind.FLOAT)(lambda value: float_int(function("ceil", value)))
RULES("math.floor", Kind.FLOAT)(lambda value: float_int(function("floor", value)))
RULES("math.ceil", Kind.INT)(identity)
RULES("math.floor", Kind.INT)(identity)
RULES("round", Kind.FLOAT)(lambda value: float_int(half_even(value)))


@RULES("pow_overflows", Kind.FLOAT, Kind.FLOAT, literals=(1,))
def pow_overflows(base: Expression, exponent: Expression | float) -> Lowered[Expression]:
    result = power(base, exponent)
    overflow = function("isinf", result.expr) & function("isfinite", base) & (base != constant(0.0))
    if isinstance(exponent, Expression):
        return Lowered(overflow & function("isfinite", exponent), result.exact)
    return Lowered(overflow if math.isfinite(exponent) else constant(False), result.exact)


@RULES("round", Kind.FLOAT, Kind.INT, literals=(1,))
def round_digits(value: Expression, digits: Expression | int) -> Lowered[Expression]:
    rounding = round_scaled(ROUND, value, digits, half_even)
    return Lowered(rounding.scaled / rounding.scale, rounding.exact)


@RULES("round_overflows", Kind.FLOAT, Kind.INT, literals=(1,))
def round_overflows(value: Expression, digits: Expression | int) -> Lowered[Expression]:
    return Lowered(constant(False), round_overflow_safe(ROUND, value, digits))


register_math(
    RULES,
    replace(ROUND, round=half_even),
    root=root,
    floor=lambda value: function("floor", value),
    ceil=lambda value: function("ceil", value),
    is_nan=lambda value: function("isnan", value),
    round_digits=round_digits,
)


register_trigonometry(
    RULES,
    ROUND,
    unary={
        name: (lambda value, name=name: function(name, value)) for name in ("sin", "cos", "acos")
    },
    atan2=lambda vertical, horizontal: function("atan2", vertical, horizontal),
    is_nan=lambda value: function("isnan", value),
)
