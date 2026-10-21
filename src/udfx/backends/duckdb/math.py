from __future__ import annotations

from duckdb import Expression

from udfx.backends.duckdb.conversions import float_int
from udfx.backends.duckdb.numbers import ORDER, cast_float64
from udfx.backends.duckdb.rules import RULES, constant, function, when
from udfx.backends.registration import Lowered, register_extremum, register_float_variants
from udfx.semantics.numbers import (
    RoundPrimitives,
    round_half_even,
    round_overflow_safe,
    scaled_round,
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


def power(base: Expression, exponent: Expression) -> Lowered[Expression]:
    complex_result = (base < constant(0.0)) & (function("floor", exponent) != exponent)
    return Lowered(function("pow", base, exponent), ~complex_result)


def root(value: Expression) -> Expression:
    return function("sqrt", when(value < constant(0.0), constant(0.0), value))


register_extremum(RULES, "max", lambda left, right: function("greatest", left, right), ORDER)
register_extremum(RULES, "min", lambda left, right: function("least", left, right), ORDER)

register_float_variants(RULES, "__pow__", power, cast_float64)
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
    rounding = scaled_round(ROUND, value, digits)
    return Lowered(half_even(rounding.scaled) / rounding.scale, rounding.exact)


@RULES("round_overflows", Kind.FLOAT, Kind.INT)
def round_overflows(value: Expression, digits: Expression) -> Lowered[Expression]:
    return Lowered(constant(False), round_overflow_safe(ROUND, value, digits))
