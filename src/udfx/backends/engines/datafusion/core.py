from __future__ import annotations

from datafusion import Expr, functions, lit

from udfx.backends.engines.datafusion.rules import RULES
from udfx.backends.registration.shared import register_missing
from udfx.types import Kind

register_missing(RULES, Expr.is_null, functions.isnan)

for _kind in (Kind.STR, Kind.INT, Kind.FLOAT, Kind.BOOL, Kind.LIST, Kind.DATE, Kind.DATETIME):
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


for _method in ("max", "min"):
    for _kind in (Kind.INT, Kind.FLOAT):
        RULES(f"aggregate {_method}", _kind, Kind.BOOL)(
            lambda value, present, method=_method: getattr(functions, method)(
                functions.when(present, value).end()
            )
        )
RULES("aggregate len", Kind.BOOL)(
    lambda present: functions.count(functions.when(present, lit(1)).end())
)
