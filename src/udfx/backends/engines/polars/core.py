from __future__ import annotations

import polars as pl

from udfx.backends.engines.polars.rules import RULES, broadcast
from udfx.backends.registration.shared import register_missing
from udfx.types import Kind

register_missing(RULES, pl.Expr.is_null, pl.Expr.is_nan)

for _kind in (Kind.STR, Kind.INT, Kind.FLOAT, Kind.BOOL, Kind.LIST, Kind.DATE, Kind.DATETIME):
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


for _method in ("max", "min"):
    for _kind in (Kind.INT, Kind.FLOAT):
        RULES(f"aggregate {_method}", _kind, Kind.BOOL)(
            lambda value, present, method=_method: getattr(
                broadcast(value).filter(broadcast(present)), method
            )()
        )
RULES("aggregate len", Kind.BOOL)(
    lambda present: broadcast(present).fill_null(False).sum().cast(pl.Int64)
)
