from __future__ import annotations

from pyspark.sql import Column, functions

from udfx.backends.lowering_context import bindings_enabled
from udfx.backends.registration import Lowered, register_lists
from udfx.backends.spark.bindings import bind
from udfx.backends.spark.rules import INTEGER_FITS, RULES, when
from udfx.backends.spark.runtime import call
from udfx.backends.spark.slicing import SEQUENCE
from udfx.backends.spark.strings import characters, list_contains
from udfx.semantics.facts import nullable_arguments, simple_arguments
from udfx.semantics.frequencies import (
    FrequencyOffsetPrimitives,
    FrequencyPrimitives,
    offset_frequencies,
    ordered_frequencies,
)
from udfx.semantics.lists import ListPrimitives, SetPrimitives
from udfx.semantics.sequences import take_prefix
from udfx.types import Kind

for kind in (Kind.INT, Kind.BOOL):
    RULES("__contains__", Kind.LIST, kind)(list_contains)


@RULES("__contains__", Kind.LIST, Kind.STR, literals=(0,), facts={"nullable": nullable_arguments})
def contains_strings(
    container: Column | list | tuple, item: Column, *, nullable: bool = True
) -> Column:
    if isinstance(container, list | tuple):
        found = item.isin([value for value in container if value is not None])
        if not nullable:
            return found
        return functions.coalesce(found, functions.lit(False)) | (
            item.isNull() & functions.lit(None in container)
        )
    return list_contains(container, item)


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


@RULES("sum", Kind.LIST, facts={"bounded": INTEGER_FITS, "simple": simple_arguments})
def total(value: Column, *, bounded: bool = False, simple: bool = False) -> Lowered[Column]:
    if bounded:
        return Lowered(
            functions.aggregate(
                value.cast("array<long>"), functions.lit(0).cast("long"), functions.try_add
            )
        )
    result = functions.aggregate(
        value.cast("array<decimal(38,0)>"),
        functions.lit(0).cast("decimal(38,0)"),
        functions.try_add,
    )
    if simple:
        items = [
            functions.coalesce(functions.get(value, index), functions.lit(0)).cast("decimal(38,0)")
            for index in range(4)
        ]
        small = (items[0] + items[1]) + (items[2] + items[3])
        result = functions.when(functions.size(value) <= 4, small).otherwise(result)
    result = result.try_cast("long")
    return Lowered(result, overflow=result.isNull() & value.isNotNull())


@RULES("sum_float", Kind.LIST)
def float_total(value: Column) -> Lowered[Column]:
    return Lowered(functions.lit(0.0), functions.lit(False))


@RULES("sorted", Kind.LIST)
def ordered(value: Column) -> Column:
    return functions.sort_array(value)


@RULES("frequency_order", Kind.LIST, Kind.INT)
def frequency_order(value: Column, count: Column) -> Column:
    if (runtime := call("frequency_order", value, count)) is not None:
        return runtime
    if bindings_enabled(False):
        return bind(value, lambda source: offset_frequencies(FREQUENCY_OFFSETS, source, count))
    return take_prefix(SEQUENCE, ordered_frequencies_value(value), count)


FREQUENCY_OFFSETS = FrequencyOffsetPrimitives(
    literal=functions.lit,
    length=functions.size,
    indices=lambda count: functions.slice(
        functions.sequence(functions.lit(0), functions.greatest(count - 1, functions.lit(0))),
        1,
        count,
    ),
    field=lambda value, index: value.getField(str(index)),
    record=lambda *values: functions.struct(
        *(value.alias(str(index)) for index, value in enumerate(values))
    ),
    sort=functions.sort_array,
    transform=functions.transform,
    filter=functions.filter,
    bind=bind,
    compact=functions.array_compact,
    get=functions.get,
    coalesce=functions.coalesce,
    position=functions.array_position,
    maximum=functions.array_max,
    prefix=lambda values, count: take_prefix(SEQUENCE, values, count),
    distinct=functions.array_distinct,
    indexed=lambda values, transform: functions.transform(
        values,
        lambda item, index: transform(item, index),  # noqa: PLW0108
    ),
    logarithm=functions.log2,
    select=when,
)


def ordered_frequencies_value(value: Column) -> Column:
    return ordered_frequencies(
        FrequencyPrimitives(
            literal=functions.lit,
            entries=functions.array_distinct,
            entry_key=lambda value: value,
            transform=functions.transform,
            count=lambda values, key: functions.size(
                functions.filter(values, lambda item: item == key)
            ),
            position=functions.array_position,
            record=lambda count, position, key: functions.struct(
                count.alias("count"), position.alias("position"), key.alias("key")
            ),
            sort=functions.sort_array,
            key=lambda record: record.getField("key"),
        ),
        value,
    )


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


RULES("list", Kind.STR)(characters)


register_lists(
    RULES,
    PRIMITIVES,
    SetPrimitives(
        literal=functions.lit,
        length=lambda value: functions.size(value).cast("long"),
        distinct=functions.array_distinct,
        sort=functions.sort_array,
        contains=functions.array_contains,
        intersection=functions.array_intersect,
        union=functions.array_union,
        concatenate=concatenate,
        difference=functions.array_except,
    ),
)
