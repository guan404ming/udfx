from __future__ import annotations

import operator

import polars as pl

from udfx.backends.base import Lowered
from udfx.backends.polars.numbers import either_nan
from udfx.backends.polars.rules import RULES, broadcast
from udfx.semantics.numbers import Binary
from udfx.types import Kind


def real(value: pl.Expr) -> pl.Expr:
    return value.cast(pl.Float64)


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


@RULES("pow_overflows", Kind.FLOAT, Kind.FLOAT)
def pow_overflows(a: pl.Expr, b: pl.Expr) -> Lowered[pl.Expr]:
    result = power(a, b)
    finite = a.is_finite() & b.is_finite() & (a != 0.0)
    return Lowered(result.expr.is_infinite() & finite, result.exact)
