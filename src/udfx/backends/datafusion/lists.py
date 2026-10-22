from __future__ import annotations

import pyarrow as pa
from datafusion import Expr, functions, lit

from udfx.backends.datafusion.rules import RULES, when
from udfx.backends.datafusion.strings import list_contains
from udfx.backends.registration import Lowered, register_lists
from udfx.semantics.lists import ListPrimitives, SetPrimitives
from udfx.types import Kind

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
    return ~functions.array_has(value, lit(False))


@RULES("sum", Kind.LIST)
def total(value: Expr) -> Lowered[Expr]:
    return Lowered(lit(0).cast(pa.int64()), lit(False))


@RULES("sum_float", Kind.LIST)
def float_total(value: Expr) -> Lowered[Expr]:
    return Lowered(lit(0.0), lit(False))


@RULES("sorted", Kind.LIST)
def ordered(value: Expr) -> Expr:
    return functions.array_sort(value)


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


@RULES("list", Kind.STR)
def characters(value: Expr) -> Expr:
    return functions.string_to_array(value, lit(None).cast(pa.string()))


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
    ),
)
