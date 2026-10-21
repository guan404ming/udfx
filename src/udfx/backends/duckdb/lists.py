from __future__ import annotations

import duckdb
from duckdb import Expression

from udfx.backends.duckdb.numbers import checked_int64
from udfx.backends.duckdb.rules import RULES, constant, function, length
from udfx.backends.duckdb.strings import list_contains
from udfx.backends.registration import Lowered
from udfx.semantics.lists import SET_ORDER_MAX_SIZE
from udfx.types import Kind

for kind in (Kind.INT, Kind.BOOL):
    RULES("__contains__", Kind.LIST, kind)(list_contains)


@RULES("list", Kind.LIST)
def copy(value: Expression) -> Expression:
    return value


@RULES("__add__", Kind.LIST, Kind.LIST)
def concatenate(left: Expression, right: Expression) -> Expression:
    return function("list_concat", left, right)


@RULES("any", Kind.LIST)
def any_value(value: Expression) -> Expression:
    return duckdb.CoalesceOperator(function("list_bool_or", value), constant(False))


@RULES("all", Kind.LIST)
def all_values(value: Expression) -> Expression:
    return duckdb.CoalesceOperator(function("list_bool_and", value), constant(True))


@RULES("sum", Kind.LIST)
def total(value: Expression) -> Lowered[Expression]:
    result = duckdb.CoalesceOperator(function("list_sum", value.cast("BIGINT[]")), constant(0))
    return checked_int64(result)


@RULES("sum_float", Kind.LIST)
def float_total(value: Expression) -> Lowered[Expression]:
    return Lowered(constant(0.0), constant(False))


@RULES("sorted", Kind.LIST)
def ordered(value: Expression) -> Expression:
    return function("list_sort", value, constant("ASC"), constant("NULLS LAST"))


@RULES("set_size", Kind.LIST)
def distinct_count(value: Expression) -> Expression:
    return length(function("list_distinct", value))


@RULES("set_list", Kind.LIST)
def distinct(value: Expression) -> Lowered[Expression]:
    result = function("list_distinct", value)
    return Lowered(result, length(result) <= constant(SET_ORDER_MAX_SIZE))


@RULES("sorted_set", Kind.LIST)
def ordered_distinct(value: Expression) -> Expression:
    return function(
        "list_sort", function("list_distinct", value), constant("ASC"), constant("NULLS LAST")
    )


@RULES("__contains__", Kind.LIST, Kind.FLOAT)
def float_contains(value: Expression, item: Expression) -> Lowered[Expression]:
    return Lowered(list_contains(value, item), ~function("isnan", item))
