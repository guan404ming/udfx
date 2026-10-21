from __future__ import annotations

import duckdb
from duckdb import Expression

from udfx.backends.duckdb.rules import RULES, constant, function, length
from udfx.backends.registration import register_missing
from udfx.types import Kind

register_missing(RULES, Expression.isnull, lambda value: function("isnan", value))

for _kind in (Kind.STR, Kind.INT, Kind.FLOAT, Kind.BOOL, Kind.LIST, Kind.DATETIME):
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


@RULES("bool", Kind.LIST)
def list_truth(value: Expression) -> Expression:
    return value.isnotnull() & (length(value) > constant(0))
