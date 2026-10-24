from __future__ import annotations

from functools import partial

import duckdb
from duckdb import Expression

from udfx.backends.duckdb.rules import (
    ITEM,
    RULES,
    bind,
    column,
    constant,
    function,
    length,
    when,
)
from udfx.backends.duckdb.slicing import SEQUENCE
from udfx.backends.duckdb.strings import join, single_code_point_clusters, strip_text
from udfx.backends.registration import Lowered
from udfx.semantics.facts import separator_is_none, split_partitioned
from udfx.semantics.splitting import SplitPrimitives, split_limited, split_right
from udfx.semantics.text import (
    non_whitespace_pattern,
    word_prefix_pattern,
)
from udfx.types import Kind


def split_words(text: Expression) -> Expression:
    return function("regexp_extract_all", text, constant(non_whitespace_pattern()))


RULES(".split", Kind.STR)(split_words)


RULES(".rsplit", Kind.STR)(split_words)


def split(text: Expression, separator: Expression, *, optional: bool | None = None) -> Expression:
    pieces = function("string_split", text, separator)
    if optional is not None:
        return split_words(text) if optional else pieces
    return when(separator.isnull(), split_words(text), pieces)


RULES(".split", Kind.STR, Kind.STR, facts={"optional": separator_is_none})(split)


@RULES("last_split", Kind.STR, Kind.STR)
def last_split(text: Expression, separator: Expression) -> Expression:
    return function("split_part", text, separator, constant(-1))


SPLIT = SplitPrimitives(
    sequence=SEQUENCE,
    bind=lambda values, transform: bind(
        function("struct_pack", *(value.alias(str(index)) for index, value in enumerate(values))),
        lambda value: transform(
            *(
                function("struct_extract", value, constant(str(index)))
                for index in range(len(values))
            )
        ),
        f"udfx split {len(values)}",
    ),
    split=split,
    words=split_words,
    join=join,
    append=lambda values, value: function("list_append", values, value),
    length=length,
    suffix=lambda value, start: function("substring", value, start + constant(1)),
    word_prefix=lambda value, count: function(
        "array_to_string",
        function(
            "list_slice",
            function("regexp_extract_all", value, constant(word_prefix_pattern())),
            constant(1),
            count,
        ),
        constant(""),
    ),
    lstrip=partial(strip_text, method="lstrip"),
    is_null=Expression.isnull,
    reverse=lambda value: function("reverse", value),
    restore=lambda values: function(
        "list_transform",
        function("list_reverse", values),
        duckdb.LambdaExpression(ITEM, function("reverse", column(ITEM))),
    ),
)


@RULES(".split", Kind.STR, Kind.STR, Kind.INT, facts={"partitioned": split_partitioned})
def split_limit(
    text: Expression, separator: Expression, maximum: Expression, *, partitioned: bool = False
) -> Expression:
    return split_limited(SPLIT, text, separator, maximum, partitioned)


@RULES(".rsplit", Kind.STR, Kind.STR, facts={"partitioned": split_partitioned})
@RULES(".rsplit", Kind.STR, Kind.STR, Kind.INT, facts={"partitioned": split_partitioned})
def rsplit(
    text: Expression,
    separator: Expression,
    maximum: Expression | None = None,
    *,
    partitioned: bool = False,
) -> Lowered[Expression]:
    exact = (
        None
        if partitioned
        else single_code_point_clusters(text)
        & duckdb.CoalesceOperator(single_code_point_clusters(separator), constant(True))
    )
    return Lowered(split_right(SPLIT, text, separator, maximum, partitioned), exact)
