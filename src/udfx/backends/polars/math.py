from __future__ import annotations

import operator

import polars as pl

from udfx.backends.base import Lowered
from udfx.backends.polars.numbers import either_nan, float_int
from udfx.backends.polars.rules import RULES, broadcast
from udfx.ops.math import EXACT_POWERS_OF_TEN, HALF_INTEGER_LIMIT, ROUND_OVERFLOW_LIMIT
from udfx.semantics.numbers import Binary
from udfx.types import Kind


def real(value: pl.Expr) -> pl.Expr:
    return value.cast(pl.Float64)


def half_even(value: pl.Expr) -> pl.Expr:
    return value.round(0, mode="half_to_even")


def identity(value: pl.Expr) -> pl.Expr:
    return value


def power(a: pl.Expr, b: pl.Expr) -> Lowered[pl.Expr]:
    base, exponent = real(broadcast(a)), real(broadcast(b))
    complex_result = (base < 0.0) & (exponent.floor() != exponent)
    return Lowered(base.pow(exponent), (pl.len() > 1) & ~complex_result)


def pick(name: str, compare: Binary) -> None:
    def ordered(a: pl.Expr, b: pl.Expr) -> pl.Expr:
        return pl.when(compare(b, a) & ~either_nan(a, b)).then(b).otherwise(a)

    def total(a: pl.Expr, b: pl.Expr) -> pl.Expr:
        return pl.when(compare(b, a)).then(b).otherwise(a)

    RULES(name, Kind.FLOAT, Kind.FLOAT)(ordered)
    RULES(name, Kind.INT, Kind.INT)(total)
    RULES(name, Kind.STR, Kind.STR)(total)


pick("max", operator.gt)
pick("min", operator.lt)

RULES("__pow__", Kind.FLOAT, Kind.FLOAT)(power)
RULES("__pow__", Kind.INT, Kind.FLOAT)(power)
RULES("__pow__", Kind.FLOAT, Kind.INT)(power)
RULES("math.sqrt", Kind.FLOAT)(pl.Expr.sqrt)
RULES("math.sqrt", Kind.INT)(lambda a: real(a).sqrt())
RULES("math.ceil", Kind.FLOAT)(lambda a: float_int(a.ceil()))
RULES("math.floor", Kind.FLOAT)(lambda a: float_int(a.floor()))
RULES("math.ceil", Kind.INT)(identity)
RULES("math.floor", Kind.INT)(identity)
RULES("round", Kind.FLOAT)(lambda a: float_int(half_even(a)))


@RULES("pow_overflows", Kind.FLOAT, Kind.FLOAT)
def pow_overflows(a: pl.Expr, b: pl.Expr) -> Lowered[pl.Expr]:
    result = power(a, b)
    finite = a.is_finite() & b.is_finite() & (a != 0.0)
    return Lowered(result.expr.is_infinite() & finite, result.exact)


@RULES("round", Kind.FLOAT, Kind.INT)
def round_digits(a: pl.Expr, digits: pl.Expr) -> Lowered[pl.Expr]:
    scale = pl.lit(10.0).pow(real(digits))
    scaled = a * scale
    tie = scaled - scaled.floor() == 0.5
    exact = (
        digits.is_between(0, EXACT_POWERS_OF_TEN)
        & (scaled.abs() < HALF_INTEGER_LIMIT)
        & ((digits == 0) | ~tie)
    )
    return Lowered(broadcast(half_even(scaled)) / broadcast(scale), exact)


@RULES("round_overflows", Kind.FLOAT, Kind.INT)
def round_overflows(a: pl.Expr, digits: pl.Expr) -> Lowered[pl.Expr]:
    return Lowered(pl.lit(False), (digits >= 0) | (a.abs() < ROUND_OVERFLOW_LIMIT))
