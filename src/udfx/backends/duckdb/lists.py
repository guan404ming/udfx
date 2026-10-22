from __future__ import annotations

import duckdb
from duckdb import Expression

from udfx.backends.duckdb.numbers import checked_int64
from udfx.backends.duckdb.rules import RULES, column, constant, function, length, when
from udfx.backends.duckdb.strings import list_contains
from udfx.backends.registration import Lowered, register_lists
from udfx.semantics.lists import ListPrimitives, SetPrimitives
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


@RULES("__contains__", Kind.LIST, Kind.FLOAT)
def float_contains(value: Expression, item: Expression) -> Lowered[Expression]:
    return Lowered(list_contains(value, item), ~function("isnan", item))


PRIMITIVES = ListPrimitives(
    literal=constant,
    when=when,
    length=length,
    repeat=lambda value, count: function(
        "flatten", function("list_resize", function("list_value", value), count, value)
    ),
    range=lambda start, stop, step: function("range", start, stop, step),
)


@RULES("list", Kind.STR)
def characters(value: Expression) -> Expression:
    return when(
        value == constant(""),
        function("list_value").cast("VARCHAR[]"),
        function("string_split", value, constant("")),
    )


def difference(left: Expression, right: Expression) -> Expression:
    return function(
        "list_distinct",
        function(
            "list_filter",
            left,
            duckdb.LambdaExpression(
                "udfx set item", ~function("list_contains", right, column("udfx set item"))
            ),
        ),
    )


register_lists(
    RULES,
    PRIMITIVES,
    SetPrimitives(
        literal=constant,
        length=length,
        distinct=lambda value: function("list_distinct", value),
        sort=lambda value: function("list_sort", value, constant("ASC"), constant("NULLS LAST")),
        contains=list_contains,
        intersection=lambda left, right: function("list_intersect", left, right),
        union=lambda left, right: function("list_distinct", function("list_concat", left, right)),
        difference=difference,
    ),
)
