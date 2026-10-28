from __future__ import annotations

from collections.abc import Callable, Sequence

from pyspark.sql import Column, functions

from udfx.backends.spark.regex_categories import category
from udfx.backends.spark.rules import INT32_MAX, RULES, split_literal
from udfx.backends.spark.slicing import SEQUENCE
from udfx.backends.spark.strings import array_join, join, length, lstrip
from udfx.frontend.core import splitting
from udfx.semantics.facts import (
    separator_is_none,
    simple_arguments,
    split_non_overlapping,
    split_partitioned,
)
from udfx.semantics.splitting import (
    RightSplitPrimitives,
    SplitPrimitives,
    split_limited,
    split_right,
    split_right_native,
)
from udfx.semantics.text import word_prefix_pattern
from udfx.types import Kind


def split_words(text: Column) -> Column:
    return functions.regexp_extract_all(
        text, functions.lit(category("CATEGORY_NOT_SPACE") + "+"), 0
    )


RULES(".split", Kind.STR)(split_words)
RULES(".rsplit", Kind.STR)(split_words)


def split(text: Column, separator: Column, *, optional: bool | None = None) -> Column:
    pieces = split_literal(text, separator)
    if optional is not None:
        return split_words(text) if optional else pieces
    return functions.when(separator.isNull(), split_words(text)).otherwise(pieces)


RULES(".split", Kind.STR, Kind.STR, facts={"optional": separator_is_none})(split)


@RULES("last_split", Kind.STR, Kind.STR, facts={"non_overlapping": split_non_overlapping})
def last_split(text: Column, separator: Column, *, non_overlapping: bool = False) -> Column:
    if non_overlapping:
        return functions.call_function("substring_index", text, separator, functions.lit(-1))
    return functions.split_part(text, separator, functions.lit(-1))


@RULES("split_item", Kind.STR, Kind.STR, Kind.INT)
def split_item(text: Column, separator: Column, index: Column) -> Column:
    position = functions.when(index < 0, index).otherwise(
        functions.try_add(index, functions.lit(1))
    )
    return functions.split_part(text, separator, position.try_cast("int"))


RULES.rewrite("__getitem__", Kind.LIST, Kind.INT)(splitting.split_item)
RULES.rewrite("__gt__", Kind.INT, Kind.INT)(splitting.split_length_greater)


def bind(values: Sequence[Column], body: Callable[..., Column], simple: bool) -> Column:
    if simple:
        return body(*values)
    packed = functions.struct(*(value.alias(str(index)) for index, value in enumerate(values)))
    return functions.transform(
        functions.array(packed),
        lambda value: body(*(value[str(index)] for index in range(len(values)))),
    )[0]


SPLIT = SplitPrimitives(
    sequence=SEQUENCE,
    bind=lambda values, transform: transform(*values),
    split=split,
    words=split_words,
    join=join,
    append=functions.array_append,
    length=length,
    suffix=lambda value, start: functions.substring(value, (start + 1).cast("int"), INT32_MAX),
    word_prefix=lambda value, count: array_join(
        functions.slice(
            functions.regexp_extract_all(
                value, functions.lit(word_prefix_pattern()), functions.lit(0)
            ),
            functions.lit(1),
            count.cast("int"),
        ),
        functions.lit(""),
    ),
    lstrip=lstrip,
    is_null=Column.isNull,
    reverse=functions.reverse,
    restore=lambda values: functions.transform(functions.reverse(values), functions.reverse),
    limited=lambda text, separator, limit: split_literal(text, separator, limit.cast("int")),
)

RIGHT_SPLIT = RightSplitPrimitives(
    split=SPLIT,
    head=lambda value, size: functions.substring(value, functions.lit(1), size.cast("int")),
    contains=functions.contains,
    tail=lambda text, separator, count: functions.call_function(
        "substring_index", text, separator, -count.cast("int")
    ),
    singleton=functions.array,
    concatenate=functions.concat,
)


@RULES(
    ".split",
    Kind.STR,
    Kind.STR,
    Kind.INT,
    facts={"optional": separator_is_none, "simple": simple_arguments},
)
def split_limit(
    text: Column,
    separator: Column,
    maximum: Column,
    *,
    optional: bool | None = None,
    simple: bool = False,
) -> Column:
    return bind(
        (text, separator, maximum),
        lambda text, separator, maximum: split_limited(
            SPLIT, text, separator, maximum, optional=optional
        ),
        simple,
    )


@RULES(
    ".rsplit",
    Kind.STR,
    Kind.STR,
    facts={"partitioned": split_partitioned, "simple": simple_arguments},
)
@RULES(
    ".rsplit",
    Kind.STR,
    Kind.STR,
    Kind.INT,
    facts={"optional": separator_is_none, "simple": simple_arguments},
)
def rsplit(
    text: Column,
    separator: Column,
    maximum: Column | None = None,
    *,
    partitioned: bool = False,
    optional: bool | None = None,
    simple: bool = False,
) -> Column:
    return bind(
        (text, separator) if maximum is None else (text, separator, maximum),
        lambda text, separator, maximum=None: split_right_native(
            RIGHT_SPLIT,
            text,
            separator,
            maximum,
            split_right(SPLIT, text, separator, maximum, partitioned, optional=optional),
        ),
        simple,
    )
