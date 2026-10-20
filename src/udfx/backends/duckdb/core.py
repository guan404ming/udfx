from __future__ import annotations

import duckdb
from duckdb import Expression

from udfx.backends.duckdb.rules import RULES, constant, length
from udfx.types import Kind


@RULES("not", Kind.BOOL)
def not_(a: Expression) -> Expression:
    return ~a


@RULES("and", Kind.BOOL, Kind.BOOL)
def and_(a: Expression, b: Expression) -> Expression:
    return a & b


@RULES("or", Kind.BOOL, Kind.BOOL)
def or_(a: Expression, b: Expression) -> Expression:
    return a | b


for _kind in (Kind.STR, Kind.INT, Kind.FLOAT, Kind.BOOL):
    RULES("is_none", _kind)(Expression.isnull)


@RULES("bool", Kind.STR)
def str_truth(a: Expression) -> Expression:
    return a.isnotnull() & (length(a) > constant(0))


@RULES("bool", Kind.INT)
def int_truth(a: Expression) -> Expression:
    return a.isnotnull() & (a != constant(0))


@RULES("bool", Kind.FLOAT)
def float_truth(a: Expression) -> Expression:
    return a.isnotnull() & (a != constant(0.0))


@RULES("bool", Kind.BOOL)
def bool_truth(a: Expression) -> Expression:
    return duckdb.CoalesceOperator(a, constant(False))
