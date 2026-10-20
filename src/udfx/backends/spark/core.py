from __future__ import annotations

from pyspark.sql import Column, functions

from udfx.backends.spark.rules import RULES
from udfx.types import Kind


@RULES("not", Kind.BOOL)
def not_(value: Column) -> Column:
    return ~value


@RULES("and", Kind.BOOL, Kind.BOOL)
def and_(left: Column, right: Column) -> Column:
    return left & right


@RULES("or", Kind.BOOL, Kind.BOOL)
def or_(left: Column, right: Column) -> Column:
    return left | right


for _kind in (Kind.STR, Kind.INT, Kind.FLOAT, Kind.BOOL):
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
