from __future__ import annotations

import pyarrow as pa
from datafusion import Expr, functions, lit

from udfx.backends.datafusion.rules import RULES, array_transform, length
from udfx.backends.datafusion.slicing import SEQUENCE
from udfx.backends.datafusion.strings import join, lstrip
from udfx.frontend.core.json import nonempty_array
from udfx.frontend.core.splitting import split_length_greater
from udfx.semantics.facts import separator_is_none, split_non_overlapping, split_partitioned
from udfx.semantics.splitting import SplitPrimitives, split_limited, split_right
from udfx.semantics.text import (
    non_whitespace_pattern,
    whitespace_normalization_pattern,
)
from udfx.types import INT64_MAX, INT64_MIN, STR, Kind, list_of


def split_words(text: Expr) -> Expr:
    spaced = functions.regexp_replace(
        text, lit(whitespace_normalization_pattern()), lit(" "), lit("g")
    )
    return functions.string_to_array(functions.btrim(spaced, lit(" ")), lit(" "))


RULES(".split", Kind.STR)(split_words)


RULES(".rsplit", Kind.STR)(split_words)


def split(text: Expr, separator: Expr, *, optional: bool | None = None) -> Expr:
    pieces = functions.when(text == lit(""), functions.make_array(lit(""))).otherwise(
        functions.string_to_array(text, separator)
    )
    if optional is not None:
        return split_words(text) if optional else pieces
    return functions.when(separator.is_null(), split_words(text)).otherwise(pieces)


RULES(".split", Kind.STR, Kind.STR, facts={"optional": separator_is_none})(split)


@RULES("last_split", Kind.STR, Kind.STR, facts={"non_overlapping": split_non_overlapping})
def last_split(text: Expr, separator: Expr, *, non_overlapping: bool = False) -> Expr:
    if non_overlapping:
        return functions.split_part(text, separator, lit(-1))
    return functions.array_element(split(text, separator), lit(-1))


@RULES("split_item", Kind.STR, Kind.STR, Kind.INT)
def split_item(text: Expr, separator: Expr, index: Expr) -> Expr:
    position = functions.when(index < lit(0), functions.nullif(index, lit(INT64_MIN))).otherwise(
        functions.nullif(index, lit(INT64_MAX)) + lit(1)
    )
    return functions.split_part(text, separator, position)


RULES.rewrite("__gt__", Kind.INT, Kind.INT)(
    lambda node: nonempty_array(node) or split_length_greater(node)
)


SPLIT = SplitPrimitives(
    sequence=SEQUENCE,
    bind=lambda values, transform: SEQUENCE.bind(
        functions.named_struct([(str(index), value) for index, value in enumerate(values)]),
        lambda value: transform(
            *(functions.get_field(value, str(index)) for index in range(len(values)))
        ),
        list_of(STR),
    ),
    split=split,
    words=split_words,
    join=join,
    append=functions.array_append,
    length=length,
    suffix=lambda value, start: functions.substr(value, start + lit(1)),
    word_prefix=lambda value, count: functions.substr(
        value,
        lit(1),
        functions.greatest(
            functions.regexp_instr(value, lit(non_whitespace_pattern()), lit(1), count + lit(1))
            - lit(1),
            lit(0),
        ),
    ),
    lstrip=lstrip,
    is_null=Expr.is_null,
    reverse=functions.reverse,
    restore=lambda values: array_transform(
        functions.array_reverse(values), functions.reverse, pa.string()
    ),
)


@RULES(".split", Kind.STR, Kind.STR, Kind.INT, facts={"partitioned": split_partitioned})
def split_limit(text: Expr, separator: Expr, maximum: Expr, *, partitioned: bool = False) -> Expr:
    return split_limited(SPLIT, text, separator, maximum, partitioned)


@RULES(".rsplit", Kind.STR, Kind.STR, facts={"partitioned": split_partitioned})
@RULES(".rsplit", Kind.STR, Kind.STR, Kind.INT, facts={"partitioned": split_partitioned})
def rsplit(
    text: Expr, separator: Expr, maximum: Expr | None = None, *, partitioned: bool = False
) -> Expr:
    return split_right(SPLIT, text, separator, maximum, partitioned)
