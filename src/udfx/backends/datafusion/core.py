from __future__ import annotations

from datafusion import Expr, functions, lit

from udfx.backends.datafusion.rules import RULES
from udfx.types import Kind


@RULES("not", Kind.BOOL)
def not_(value: Expr) -> Expr:
    return ~value


@RULES("and", Kind.BOOL, Kind.BOOL)
def and_(left: Expr, right: Expr) -> Expr:
    return left & right


@RULES("or", Kind.BOOL, Kind.BOOL)
def or_(left: Expr, right: Expr) -> Expr:
    return left | right


for _kind in (Kind.STR, Kind.INT, Kind.FLOAT, Kind.BOOL):
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
