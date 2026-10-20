from __future__ import annotations

from datafusion import Expr, functions, lit

from udfx.backends.datafusion.rules import RULES
from udfx.types import Kind


@RULES("not", Kind.BOOL)
def not_(a: Expr) -> Expr:
    return ~a


@RULES("and", Kind.BOOL, Kind.BOOL)
def and_(a: Expr, b: Expr) -> Expr:
    return a & b


@RULES("or", Kind.BOOL, Kind.BOOL)
def or_(a: Expr, b: Expr) -> Expr:
    return a | b


for _kind in (Kind.STR, Kind.INT, Kind.FLOAT, Kind.BOOL):
    RULES("is_none", _kind)(Expr.is_null)


@RULES("bool", Kind.STR)
def str_truth(a: Expr) -> Expr:
    return a.is_not_null() & (functions.octet_length(a) > lit(0))


@RULES("bool", Kind.INT)
def int_truth(a: Expr) -> Expr:
    return a.is_not_null() & (a != lit(0))


@RULES("bool", Kind.FLOAT)
def float_truth(a: Expr) -> Expr:
    return a.is_not_null() & (functions.isnan(a) | (a != lit(0.0)))


@RULES("bool", Kind.BOOL)
def bool_truth(a: Expr) -> Expr:
    return functions.coalesce(a, lit(False))
