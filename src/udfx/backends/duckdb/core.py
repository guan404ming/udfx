from __future__ import annotations

import duckdb
from duckdb import Expression

from udfx.backends.duckdb.rules import RULES, constant, length
from udfx.types import Kind


@RULES("not", Kind.BOOL)
def not_(value: Expression) -> Expression:
    return ~value


@RULES("and", Kind.BOOL, Kind.BOOL)
def and_(left: Expression, right: Expression) -> Expression:
    return left & right


@RULES("or", Kind.BOOL, Kind.BOOL)
def or_(left: Expression, right: Expression) -> Expression:
    return left | right


for _kind in (Kind.STR, Kind.INT, Kind.FLOAT, Kind.BOOL):
    RULES("is_none", _kind)(Expression.isnull)


@RULES("bool", Kind.STR)
def str_truth(value: Expression) -> Expression:
    return value.isnotnull() & (length(value) > constant(0))


@RULES("bool", Kind.INT)
def int_truth(value: Expression) -> Expression:
    return value.isnotnull() & (value != constant(0))


@RULES("bool", Kind.FLOAT)
def float_truth(value: Expression) -> Expression:
    return value.isnotnull() & (value != constant(0.0))


@RULES("bool", Kind.BOOL)
def bool_truth(value: Expression) -> Expression:
    return duckdb.CoalesceOperator(value, constant(False))
