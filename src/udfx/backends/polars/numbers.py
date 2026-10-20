from __future__ import annotations

import polars as pl

from udfx.backends.base import Lowered
from udfx.backends.polars.rules import RULES
from udfx.types import Kind

INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1


def fits_int64(wide: pl.Expr) -> pl.Expr:
    return wide.is_between(INT64_MIN, INT64_MAX)


@RULES("__add__", Kind.INT, Kind.INT)
def int_add(a: pl.Expr, b: pl.Expr) -> Lowered[pl.Expr]:
    wide = a.cast(pl.Int128) + b.cast(pl.Int128)
    return Lowered(a + b, fits_int64(wide))


@RULES("__gt__", Kind.INT, Kind.INT)
def int_gt(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    return a > b
