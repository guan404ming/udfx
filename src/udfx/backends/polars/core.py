from __future__ import annotations

import polars as pl

from udfx.backends.polars.rules import RULES
from udfx.types import Kind


@RULES("not", Kind.BOOL)
def not_(a: pl.Expr) -> pl.Expr:
    return ~a


@RULES("and", Kind.BOOL, Kind.BOOL)
def and_(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    return a & b


@RULES("or", Kind.BOOL, Kind.BOOL)
def or_(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    return a | b


for _kind in (Kind.STR, Kind.INT, Kind.FLOAT, Kind.BOOL):
    RULES("is_none", _kind)(pl.Expr.is_null)


@RULES("bool", Kind.STR)
def str_truth(a: pl.Expr) -> pl.Expr:
    return a.is_not_null() & (a.str.len_bytes() > 0)


@RULES("bool", Kind.INT)
def int_truth(a: pl.Expr) -> pl.Expr:
    return a.is_not_null() & (a != 0)


@RULES("bool", Kind.FLOAT)
def float_truth(a: pl.Expr) -> pl.Expr:
    return a.is_not_null() & (a.is_nan() | (a != 0.0))


@RULES("bool", Kind.BOOL)
def bool_truth(a: pl.Expr) -> pl.Expr:
    return a.fill_null(False)
