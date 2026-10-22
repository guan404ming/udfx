from __future__ import annotations

import polars as pl

from udfx.backends.polars.conversions import float_int
from udfx.backends.polars.numbers import ORDER, cast_float64
from udfx.backends.polars.rules import RULES, broadcast, when
from udfx.backends.registration import (
    Lowered,
    register_extremum,
    register_float_variants,
    register_math,
)
from udfx.semantics.numbers import RoundPrimitives, round_overflow_safe, scaled_round
from udfx.types import Kind


def half_even(value: pl.Expr) -> pl.Expr:
    return value.round(0, mode="half_to_even")


def identity(value: pl.Expr) -> pl.Expr:
    return value


ROUND = RoundPrimitives(
    pl.lit, pl.Expr.pow, pl.Expr.is_between, pl.Expr.abs, cast_float64, half_even, when
)


def power(base: pl.Expr, exponent: pl.Expr) -> Lowered[pl.Expr]:
    base, exponent = broadcast(base), broadcast(exponent)
    complex_result = (base < 0.0) & (exponent.floor() != exponent)
    return Lowered(base.pow(exponent), (pl.len() > 1) & ~complex_result)


register_extremum(RULES, "max", pl.max_horizontal, ORDER)
register_extremum(RULES, "min", pl.min_horizontal, ORDER)

register_float_variants(RULES, "__pow__", power, cast_float64)
RULES("math.sqrt", Kind.FLOAT)(pl.Expr.sqrt)
RULES("math.sqrt", Kind.INT)(lambda value: cast_float64(value).sqrt())
RULES("math.ceil", Kind.FLOAT)(lambda value: float_int(value.ceil()))
RULES("math.floor", Kind.FLOAT)(lambda value: float_int(value.floor()))
RULES("math.ceil", Kind.INT)(identity)
RULES("math.floor", Kind.INT)(identity)
RULES("round", Kind.FLOAT)(lambda value: float_int(half_even(value)))


@RULES("pow_overflows", Kind.FLOAT, Kind.FLOAT)
def pow_overflows(base: pl.Expr, exponent: pl.Expr) -> Lowered[pl.Expr]:
    result = power(base, exponent)
    finite = base.is_finite() & exponent.is_finite() & (base != 0.0)
    return Lowered(result.expr.is_infinite() & finite, result.exact)


@RULES("round", Kind.FLOAT, Kind.INT)
def round_digits(value: pl.Expr, digits: pl.Expr) -> Lowered[pl.Expr]:
    rounding = scaled_round(ROUND, value, digits)
    rounded = broadcast(half_even(rounding.scaled)) / broadcast(rounding.scale)
    return Lowered(rounded, rounding.exact)


@RULES("round_overflows", Kind.FLOAT, Kind.INT)
def round_overflows(value: pl.Expr, digits: pl.Expr) -> Lowered[pl.Expr]:
    return Lowered(pl.lit(False), round_overflow_safe(ROUND, value, digits))


register_math(
    RULES,
    ROUND,
    root=pl.Expr.sqrt,
    floor=pl.Expr.floor,
    ceil=pl.Expr.ceil,
    is_nan=pl.Expr.is_nan,
    round_digits=round_digits,
)
