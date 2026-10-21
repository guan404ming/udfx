from __future__ import annotations

import pyarrow as pa
from datafusion import Expr, functions, lit

from udfx.backends.datafusion.rules import RULES
from udfx.backends.datafusion.strings import list_contains
from udfx.backends.registration import Lowered
from udfx.semantics.lists import SET_ORDER_MAX_SIZE
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


@RULES("set_size", Kind.LIST)
def distinct_count(value: Expr) -> Expr:
    return functions.cardinality(functions.array_distinct(value)).cast(pa.int64())


@RULES("set_list", Kind.LIST)
def distinct(value: Expr) -> Lowered[Expr]:
    result = functions.array_distinct(value)
    return Lowered(result, functions.cardinality(result) <= lit(SET_ORDER_MAX_SIZE))


@RULES("sorted_set", Kind.LIST)
def ordered_distinct(value: Expr) -> Expr:
    return functions.array_sort(functions.array_distinct(value))


@RULES("__contains__", Kind.LIST, Kind.FLOAT)
def float_contains(value: Expr, item: Expr) -> Lowered[Expr]:
    return Lowered(list_contains(value, item), ~functions.isnan(item) & (item != lit(0.0)))
