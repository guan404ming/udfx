from __future__ import annotations

from collections.abc import Callable

import duckdb
from duckdb import Expression

from udfx.backends.duckdb.numbers import checked_int64
from udfx.backends.duckdb.rules import RULES, bind, column, constant, function, length, when
from udfx.backends.duckdb.slicing import SEQUENCE
from udfx.backends.duckdb.strings import list_contains
from udfx.backends.registration import Lowered, register_lists
from udfx.frontend.core.comprehensions import lowercase_words
from udfx.semantics.frequencies import FrequencyPrimitives, ordered_frequencies
from udfx.semantics.lists import ListPrimitives, SetPrimitives
from udfx.semantics.regex_patterns import literal_union
from udfx.semantics.sequences import take_prefix
from udfx.types import Kind

for kind in (Kind.INT, Kind.BOOL):
    RULES("__contains__", Kind.LIST, kind)(list_contains)


@RULES("__contains__", Kind.LIST, Kind.STR, literals=(0,))
def contains_strings(container: Expression | list | tuple, item: Expression) -> Expression:
    if isinstance(container, list | tuple):
        pattern = literal_union(tuple(value for value in container if value is not None))
        found = (
            function("regexp_full_match", item, constant(pattern))
            if pattern is not None
            else constant(False)
        )
        return duckdb.CoalesceOperator(found, constant(False)) | (
            item.isnull() & constant(None in container)
        )
    return list_contains(container, item)


@RULES("list", Kind.LIST)
def copy(value: Expression) -> Expression:
    return value


def concatenate(left: Expression, right: Expression) -> Expression:
    return function("list_concat", left, right)


RULES("__add__", Kind.LIST, Kind.LIST)(concatenate)


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


def frequency_entries(counts: Expression, count: Expression) -> Expression:
    sizes = function("list_sort", function("map_values", counts), constant("DESC"))
    cutoff = duckdb.CoalesceOperator(
        function("list_extract", sizes, function("greatest", count, constant(1))), constant(0)
    )
    entries = duckdb.CoalesceOperator(function("map_entries", counts), constant([]))
    return bind(
        cutoff,
        lambda threshold: function(
            "list_filter",
            entries,
            duckdb.LambdaExpression(
                "udfx frequency candidate",
                function("struct_extract", column("udfx frequency candidate"), constant("value"))
                >= threshold,
            ),
        ),
        "udfx frequency cutoff",
    )


@RULES("frequency_order", Kind.LIST, Kind.INT)
def frequency_order(value: Expression, count: Expression) -> Expression:
    def counted(values: Expression) -> Expression:
        def ranked(counts: Expression) -> Expression:
            return ordered_frequencies(
                FrequencyPrimitives(
                    literal=constant,
                    entries=lambda values: frequency_entries(counts, count),
                    entry_key=lambda entry: function("struct_extract", entry, constant("key")),
                    transform=lambda values, transform: function(
                        "list_transform",
                        values,
                        duckdb.LambdaExpression(
                            "udfx frequency key", transform(column("udfx frequency key"))
                        ),
                    ),
                    count=lambda values, entry: function(
                        "struct_extract", entry, constant("value")
                    ).cast("BIGINT"),
                    position=lambda values, key: function("list_position", values, key),
                    record=lambda count, position, key: function(
                        "struct_pack",
                        count.alias("count"),
                        position.alias("position"),
                        key.alias("key"),
                    ),
                    sort=ordered,
                    key=lambda record: function("struct_extract", record, constant("key")),
                ),
                values,
            )

        return bind(function("list_histogram", values), ranked, "udfx frequency counts")

    return take_prefix(SEQUENCE, bind(value, counted, "udfx frequency source"), count)


RULES.rewrite("frequency_order", Kind.LIST, Kind.INT)(lowercase_words)


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


def bind_sets(
    left: Expression, right: Expression, transform: Callable[[Expression, Expression], Expression]
) -> Expression:
    return bind(
        function("list_value", left, right),
        lambda operands: transform(
            function("list_extract", operands, constant(1)),
            function("list_extract", operands, constant(2)),
        ),
        "udfx sets",
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


def union(left: Expression, right: Expression) -> Expression:
    return function("list_distinct", function("list_concat", left, right))


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
        union=union,
        concatenate=concatenate,
        difference=lambda left, right: bind_sets(left, right, difference),
        size=lambda value: function("list_unique", value).cast("BIGINT"),
        symmetric_difference=lambda left, right: bind_sets(
            left,
            right,
            lambda first, second: concatenate(difference(first, second), difference(second, first)),
        ),
    ),
)
