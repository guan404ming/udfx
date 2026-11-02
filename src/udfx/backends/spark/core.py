from __future__ import annotations

from pyspark.sql import Column, functions

from udfx.backends.registration import register_missing
from udfx.backends.spark.rules import RULES
from udfx.types import Kind

register_missing(RULES, Column.isNull, functions.isnan)

for _kind in (Kind.STR, Kind.INT, Kind.FLOAT, Kind.BOOL, Kind.LIST, Kind.DATE, Kind.DATETIME):
    RULES("is_none", _kind)(Column.isNull)


@RULES("bool", Kind.STR)
def str_truth(value: Column) -> Column:
    return value.isNotNull() & (functions.octet_length(value) > 0)


@RULES("bool", Kind.INT)
def int_truth(value: Column) -> Column:
    return value.isNotNull() & (value != 0)


@RULES("bool", Kind.FLOAT)
def float_truth(value: Column) -> Column:
    return value.isNotNull() & (value != 0.0)


@RULES("bool", Kind.BOOL)
def bool_truth(value: Column) -> Column:
    return functions.coalesce(value, functions.lit(False))


@RULES("bool", Kind.LIST)
def list_truth(value: Column) -> Column:
    return value.isNotNull() & (functions.size(value) > 0)


for _method in ("max", "min"):
    for _kind in (Kind.INT, Kind.FLOAT):
        RULES(f"aggregate {_method}", _kind, Kind.BOOL)(
            lambda value, present, method=_method: getattr(functions, method)(
                functions.when(present, value)
            )
        )
RULES("aggregate len", Kind.BOOL)(
    lambda present: functions.count(functions.when(present, functions.lit(1)))
)
