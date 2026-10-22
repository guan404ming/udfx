from __future__ import annotations

from pyspark.sql import Column, functions

from udfx.backends.registration import Lowered, register_lists
from udfx.backends.spark.rules import RULES, when
from udfx.backends.spark.strings import list_contains
from udfx.semantics.lists import ListPrimitives, SetPrimitives
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
    return Lowered(result, overflow=result.isNull() & value.isNotNull())


@RULES("sum_float", Kind.LIST)
def float_total(value: Column) -> Lowered[Column]:
    return Lowered(functions.lit(0.0), functions.lit(False))


@RULES("sorted", Kind.LIST)
def ordered(value: Column) -> Column:
    return functions.sort_array(value)


@RULES("__contains__", Kind.LIST, Kind.FLOAT)
def float_contains(value: Column, item: Column) -> Lowered[Column]:
    return Lowered(list_contains(value, item), ~functions.isnan(item))


def range_values(start: Column, stop: Column, step: Column) -> Column:
    end = functions.when(start == stop, start).otherwise(
        stop - functions.when(step > 0, functions.lit(1)).otherwise(functions.lit(-1))
    )
    return functions.when(start == stop, functions.array().cast("array<long>")).otherwise(
        functions.sequence(start, end, step)
    )


PRIMITIVES = ListPrimitives(
    literal=functions.lit,
    when=when,
    length=lambda value: functions.size(value).cast("long"),
    repeat=lambda value, count: functions.flatten(functions.array_repeat(value, count.cast("int"))),
    range=range_values,
)


@RULES("list", Kind.STR)
def characters(value: Column) -> Column:
    return functions.when(value == "", functions.array().cast("array<string>")).otherwise(
        functions.split(value, "", -1)
    )


register_lists(
    RULES,
    PRIMITIVES,
    SetPrimitives(
        literal=functions.lit,
        length=lambda value: functions.size(value).cast("long"),
        distinct=functions.array_distinct,
        sort=functions.sort_array,
        contains=functions.array_contains,
    ),
)
