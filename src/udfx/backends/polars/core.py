from __future__ import annotations

import polars as pl

from udfx.backends.polars.rules import RULES
from udfx.types import Kind

for _kind in (Kind.STR, Kind.INT, Kind.FLOAT, Kind.BOOL, Kind.LIST):
    RULES("is_none", _kind)(pl.Expr.is_null)


@RULES("bool", Kind.STR)
def str_truth(value: pl.Expr) -> pl.Expr:
    return value.is_not_null() & (value.str.len_bytes() > 0)


@RULES("bool", Kind.INT)
def int_truth(value: pl.Expr) -> pl.Expr:
    return value.is_not_null() & (value != 0)


@RULES("bool", Kind.FLOAT)
def float_truth(value: pl.Expr) -> pl.Expr:
    return value.is_not_null() & (value.is_nan() | (value != 0.0))


@RULES("bool", Kind.BOOL)
def bool_truth(value: pl.Expr) -> pl.Expr:
    return value.fill_null(False)


@RULES("bool", Kind.LIST)
def list_truth(value: pl.Expr) -> pl.Expr:
    return value.is_not_null() & (value.list.len() > 0)
