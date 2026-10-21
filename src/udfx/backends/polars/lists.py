from __future__ import annotations

import polars as pl

from udfx.backends.polars.rules import RULES
from udfx.backends.polars.strings import list_contains
from udfx.backends.registration import Lowered
from udfx.semantics.lists import SET_ORDER_MAX_SIZE
from udfx.types import Kind

for kind in (Kind.INT, Kind.BOOL):
    RULES("__contains__", Kind.LIST, kind)(list_contains)


@RULES("list", Kind.LIST)
def copy(value: pl.Expr) -> pl.Expr:
    return value


@RULES("__add__", Kind.LIST, Kind.LIST)
def concatenate(left: pl.Expr, right: pl.Expr) -> pl.Expr:
    return left.list.concat(right)


@RULES("any", Kind.LIST)
def any_value(value: pl.Expr) -> pl.Expr:
    return value.list.any()


@RULES("all", Kind.LIST)
def all_values(value: pl.Expr) -> pl.Expr:
    return value.list.all()


@RULES("sum", Kind.LIST)
def total(value: pl.Expr) -> Lowered[pl.Expr]:
    result = value.list.eval(pl.element().cast(pl.Int128)).list.sum().cast(pl.Int64, strict=False)
    return Lowered(result, result.is_not_null())


@RULES("sum_float", Kind.LIST)
def float_total(value: pl.Expr) -> Lowered[pl.Expr]:
    return Lowered(pl.lit(0.0), pl.lit(False))


@RULES("sorted", Kind.LIST)
def ordered(value: pl.Expr) -> pl.Expr:
    return value.list.sort()


@RULES("set_size", Kind.LIST)
def distinct_count(value: pl.Expr) -> pl.Expr:
    return value.list.n_unique().cast(pl.Int64)


@RULES("set_list", Kind.LIST)
def distinct(value: pl.Expr) -> Lowered[pl.Expr]:
    return Lowered(value.list.unique(), value.list.n_unique() <= SET_ORDER_MAX_SIZE)


@RULES("sorted_set", Kind.LIST)
def ordered_distinct(value: pl.Expr) -> pl.Expr:
    return value.list.unique().list.sort()


@RULES("__contains__", Kind.LIST, Kind.FLOAT)
def float_contains(value: pl.Expr, item: pl.Expr) -> Lowered[pl.Expr]:
    return Lowered(list_contains(value, item), item.is_not_nan())
