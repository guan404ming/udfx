from __future__ import annotations

from pyspark.sql import Column, functions

from udfx.backends.spark.rules import RULES
from udfx.types import Kind


@RULES("not", Kind.BOOL)
def not_(a: Column) -> Column:
    return ~a


@RULES("and", Kind.BOOL, Kind.BOOL)
def and_(a: Column, b: Column) -> Column:
    return a & b


@RULES("or", Kind.BOOL, Kind.BOOL)
def or_(a: Column, b: Column) -> Column:
    return a | b


for _kind in (Kind.STR, Kind.INT, Kind.FLOAT, Kind.BOOL):
    RULES("is_none", _kind)(Column.isNull)


@RULES("bool", Kind.STR)
def str_truth(a: Column) -> Column:
    return a.isNotNull() & (functions.length(a) > 0)


@RULES("bool", Kind.INT)
def int_truth(a: Column) -> Column:
    return a.isNotNull() & (a != 0)


@RULES("bool", Kind.FLOAT)
def float_truth(a: Column) -> Column:
    return a.isNotNull() & (a != 0.0)


@RULES("bool", Kind.BOOL)
def bool_truth(a: Column) -> Column:
    return functions.coalesce(a, functions.lit(False))
