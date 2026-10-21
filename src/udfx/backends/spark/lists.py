from __future__ import annotations

from pyspark.sql import Column, functions

from udfx.backends.registration import Lowered
from udfx.backends.spark.rules import RULES
from udfx.backends.spark.strings import list_contains
from udfx.semantics.lists import SET_ORDER_MAX_SIZE
from udfx.types import Kind

for kind in (Kind.INT, Kind.BOOL):
    RULES("__contains__", Kind.LIST, kind)(list_contains)


@RULES("list", Kind.LIST)
def copy(value: Column) -> Column:
    return value


@RULES("__add__", Kind.LIST, Kind.LIST)
def concatenate(left: Column, right: Column) -> Column:
    return functions.concat(left, right)


@RULES("any", Kind.LIST)
def any_value(value: Column) -> Column:
    return functions.exists(value, lambda element: element)


@RULES("all", Kind.LIST)
def all_values(value: Column) -> Column:
    return functions.forall(value, lambda element: element)


@RULES("sum", Kind.LIST)
def total(value: Column) -> Lowered[Column]:
    result = functions.aggregate(
        value.cast("array<decimal(38,0)>"),
        functions.lit(0).cast("decimal(38,0)"),
        functions.try_add,
    ).try_cast("long")
    return Lowered(result, result.isNotNull())


@RULES("sum_float", Kind.LIST)
def float_total(value: Column) -> Lowered[Column]:
    return Lowered(functions.lit(0.0), functions.lit(False))


@RULES("sorted", Kind.LIST)
def ordered(value: Column) -> Column:
    return functions.sort_array(value)


@RULES("set_size", Kind.LIST)
def distinct_count(value: Column) -> Column:
    return functions.size(functions.array_distinct(value)).cast("long")


@RULES("set_list", Kind.LIST)
def distinct(value: Column) -> Lowered[Column]:
    result = functions.array_distinct(value)
    return Lowered(result, functions.size(result) <= SET_ORDER_MAX_SIZE)


@RULES("sorted_set", Kind.LIST)
def ordered_distinct(value: Column) -> Column:
    return functions.sort_array(functions.array_distinct(value))


@RULES("__contains__", Kind.LIST, Kind.FLOAT)
def float_contains(value: Column, item: Column) -> Lowered[Column]:
    return Lowered(list_contains(value, item), ~functions.isnan(item))
