from __future__ import annotations

from pyspark.sql import Column, functions

from udfx.backends.spark.rules import RULES
from udfx.types import Kind

for _kind in (Kind.STR, Kind.INT, Kind.FLOAT, Kind.BOOL, Kind.LIST):
    RULES("is_none", _kind)(Column.isNull)


@RULES("bool", Kind.STR)
def str_truth(value: Column) -> Column:
    return value.isNotNull() & (functions.length(value) > 0)


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
