from __future__ import annotations

import duckdb
from duckdb import Expression

from udfx.backends.engines.duckdb.rules import RULES, constant, function, length
from udfx.backends.registration.shared import register_missing
from udfx.types import Kind

register_missing(RULES, Expression.isnull, lambda value: function("isnan", value))

for _kind in (Kind.STR, Kind.INT, Kind.FLOAT, Kind.BOOL, Kind.LIST, Kind.DATE, Kind.DATETIME):
    RULES("is_none", _kind)(Expression.isnull)


@RULES("bool", Kind.STR)
def str_truth(value: Expression) -> Expression:
    return value.isnotnull() & (function("strlen", value) > constant(0))


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


for _method in ("max", "min"):
    for _kind in (Kind.INT, Kind.FLOAT):
        RULES(f"aggregate {_method}", _kind, Kind.BOOL)(
            lambda value, present, method=_method: function(
                method, duckdb.CaseExpression(present, value)
            )
        )
RULES("aggregate len", Kind.BOOL)(
    lambda present: function("count", duckdb.CaseExpression(present, constant(1)))
)
