from __future__ import annotations

import pyarrow as pa
from datafusion import Expr, functions, lit

from udfx.backends.datafusion.rules import RULES, array_filter, array_transform, when
from udfx.backends.datafusion.slicing import SEQUENCE
from udfx.backends.datafusion.strings import characters, list_contains, list_len
from udfx.backends.registration import Lowered, register_lists
from udfx.semantics.frequencies import FrequencyPrimitives, ordered_frequencies
from udfx.semantics.lists import ListPrimitives, SetPrimitives, SumPrimitives, integer_sum_exact
from udfx.types import BOOL, INT, Kind

for kind in (Kind.INT, Kind.BOOL):
    RULES("__contains__", Kind.LIST, kind)(list_contains)


@RULES("list", Kind.LIST)
def copy(value: Expr) -> Expr:
    return value


@RULES("__add__", Kind.LIST, Kind.LIST)
def concatenate(left: Expr, right: Expr) -> Expr:
    return functions.array_concat(left, right)


@RULES("any", Kind.LIST)
def any_value(value: Expr) -> Expr:
    return functions.array_has(value, lit(True))


@RULES("all", Kind.LIST)
def all_values(value: Expr) -> Expr:
    flags = functions.array_replace_all(value, lit(None).cast(pa.bool_()), lit(False))
    return ~functions.array_has(flags, lit(False))


SUM = SumPrimitives(
    literal=lit,
    when=when,
    length=list_len,
    minimum=functions.array_min,
    maximum=functions.array_max,
    coalesce=functions.coalesce,
)


def checked_total(value: Expr) -> Expr:
    value = value.cast(pa.list_(pa.int64()))
    exact = integer_sum_exact(SUM, value)
    result = functions.coalesce(functions.array_sum(value), lit(0))
    return when(exact, result, lit(0)).cast(pa.int64())


@RULES("sum", Kind.LIST)
def total(value: Expr) -> Lowered[Expr]:
    integers = value.cast(pa.list_(pa.int64()))
    return Lowered(
        SEQUENCE.bind(integers, checked_total, INT),
        SEQUENCE.bind(
            integers, lambda items: integer_sum_exact(SUM, items.cast(pa.list_(pa.int64()))), BOOL
        ),
    )


@RULES("sum_float", Kind.LIST)
def float_total(value: Expr) -> Lowered[Expr]:
    return Lowered(lit(0.0), lit(False))


@RULES("sorted", Kind.LIST)
def ordered(value: Expr) -> Expr:
    return functions.array_sort(value)


@RULES("frequency_order", Kind.LIST)
def frequency_order(value: Expr) -> Expr:
    return ordered_frequencies(
        FrequencyPrimitives(
            literal=lit,
            distinct=functions.array_distinct,
            transform=array_transform,
            count=lambda values, key: functions.cardinality(
                array_filter(values, lambda item: item == key)
            ),
            position=functions.array_position,
            record=lambda count, position, key: functions.named_struct(
                [("count", count), ("position", position), ("key", key)]
            ),
            sort=functions.array_sort,
            key=lambda record: functions.get_field(record, "key"),
        ),
        value,
    )


@RULES("__contains__", Kind.LIST, Kind.FLOAT)
def float_contains(value: Expr, item: Expr) -> Lowered[Expr]:
    return Lowered(list_contains(value, item), ~functions.isnan(item) & (item != lit(0.0)))


PRIMITIVES = ListPrimitives(
    literal=lit,
    when=when,
    length=lambda value: functions.cardinality(value).cast(pa.int64()),
    repeat=lambda value, count: functions.flatten(functions.array_repeat(value, count)),
    range=functions.range,
)


RULES("list", Kind.STR)(characters)


register_lists(
    RULES,
    PRIMITIVES,
    SetPrimitives(
        literal=lit,
        length=lambda value: functions.cardinality(value).cast(pa.int64()),
        distinct=functions.array_distinct,
        sort=functions.array_sort,
        contains=lambda value, item: functions.array_any_match(
            value,
            lambda candidate: (
                (candidate == item) | (functions.isnan(candidate) & functions.isnan(item))
            ),
        ),
        intersection=functions.array_intersect,
        union=functions.array_union,
        concatenate=concatenate,
        difference=functions.array_except,
    ),
)
