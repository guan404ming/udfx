from __future__ import annotations

from datafusion import Expr, functions, lit

from udfx.backends.datafusion.rules import RULES
from udfx.backends.registration import register_missing
from udfx.types import Kind

register_missing(RULES, Expr.is_null, functions.isnan)

for _kind in (Kind.STR, Kind.INT, Kind.FLOAT, Kind.BOOL, Kind.LIST):
    RULES("is_none", _kind)(Expr.is_null)


@RULES("bool", Kind.STR)
def str_truth(value: Expr) -> Expr:
    return value.is_not_null() & (functions.octet_length(value) > lit(0))


@RULES("bool", Kind.INT)
def int_truth(value: Expr) -> Expr:
    return value.is_not_null() & (value != lit(0))


@RULES("bool", Kind.FLOAT)
def float_truth(value: Expr) -> Expr:
    return value.is_not_null() & (functions.isnan(value) | (value != lit(0.0)))


@RULES("bool", Kind.BOOL)
def bool_truth(value: Expr) -> Expr:
    return functions.coalesce(value, lit(False))


@RULES("bool", Kind.LIST)
def list_truth(value: Expr) -> Expr:
    return value.is_not_null() & (functions.cardinality(value) > lit(0))
