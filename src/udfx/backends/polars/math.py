from __future__ import annotations

import operator

import polars as pl

from udfx.backends.base import Lowered
from udfx.backends.polars.conversions import float_int
from udfx.backends.polars.numbers import cast_float64, either_nan
from udfx.backends.polars.rules import RULES, broadcast
from udfx.semantics.numbers import (
    EXACT_POWERS_OF_TEN,
    HALF_INTEGER_LIMIT,
    ROUND_OVERFLOW_LIMIT,
    Binary,
)
from udfx.types import Kind


def half_even(value: pl.Expr) -> pl.Expr:
    return value.round(0, mode="half_to_even")


def identity(value: pl.Expr) -> pl.Expr:
    return value


def power(base: pl.Expr, exponent: pl.Expr) -> Lowered[pl.Expr]:
    base, exponent = cast_float64(broadcast(base)), cast_float64(broadcast(exponent))
    complex_result = (base < 0.0) & (exponent.floor() != exponent)
    return Lowered(base.pow(exponent), (pl.len() > 1) & ~complex_result)


def register_extremum(name: str, compare: Binary) -> None:
    def ordered(left: pl.Expr, right: pl.Expr) -> pl.Expr:
        return pl.when(compare(right, left) & ~either_nan(left, right)).then(right).otherwise(left)

    def total(left: pl.Expr, right: pl.Expr) -> pl.Expr:
        return pl.when(compare(right, left)).then(right).otherwise(left)

    RULES(name, Kind.FLOAT, Kind.FLOAT)(ordered)
    RULES(name, Kind.INT, Kind.INT)(total)
    RULES(name, Kind.STR, Kind.STR)(total)


register_extremum("max", operator.gt)
register_extremum("min", operator.lt)

RULES("__pow__", Kind.FLOAT, Kind.FLOAT)(power)
RULES("__pow__", Kind.INT, Kind.FLOAT)(power)
RULES("__pow__", Kind.FLOAT, Kind.INT)(power)
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
    scale = pl.lit(10.0).pow(cast_float64(digits))
    scaled = value * scale
    tie = scaled - scaled.floor() == 0.5
    exact = (
        digits.is_between(0, EXACT_POWERS_OF_TEN)
        & (scaled.abs() < HALF_INTEGER_LIMIT)
        & ((digits == 0) | ~tie)
    )
    return Lowered(broadcast(half_even(scaled)) / broadcast(scale), exact)


@RULES("round_overflows", Kind.FLOAT, Kind.INT)
def round_overflows(value: pl.Expr, digits: pl.Expr) -> Lowered[pl.Expr]:
    return Lowered(pl.lit(False), (digits >= 0) | (value.abs() < ROUND_OVERFLOW_LIMIT))
