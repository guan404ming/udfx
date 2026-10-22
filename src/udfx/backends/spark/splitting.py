from __future__ import annotations

from pyspark.sql import Column, functions

from udfx.backends.spark.rules import RULES, split_literal
from udfx.backends.spark.slicing import SEQUENCE
from udfx.backends.spark.strings import INT32_MAX, array_join, join, length, lstrip
from udfx.semantics.facts import split_partitioned
from udfx.semantics.splitting import SplitPrimitives, split_limited, split_right
from udfx.semantics.text import non_whitespace_pattern, word_prefix_pattern
from udfx.types import STR, Kind, list_of


def split_words(text: Column) -> Column:
    return functions.regexp_extract_all(
        text, functions.lit(non_whitespace_pattern()), functions.lit(0)
    )


RULES(".split", Kind.STR)(split_words)
RULES(".rsplit", Kind.STR)(split_words)


def split(text: Column, separator: Column) -> Column:
    pieces = split_literal(text, separator)
    return functions.when(separator.isNull(), split_words(text)).otherwise(pieces)


RULES(".split", Kind.STR, Kind.STR)(split)


@RULES("last_split", Kind.STR, Kind.STR)
def last_split(text: Column, separator: Column) -> Column:
    return functions.split_part(text, separator, functions.lit(-1))


SPLIT = SplitPrimitives(
    sequence=SEQUENCE,
    bind=lambda values, transform: SEQUENCE.bind(
        functions.struct(*(value.alias(str(index)) for index, value in enumerate(values))),
        lambda value: transform(*(value.getField(str(index)) for index in range(len(values)))),
        list_of(STR),
    ),
    split=split,
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
)


@RULES(".split", Kind.STR, Kind.STR, Kind.INT, facts={"partitioned": split_partitioned})
def split_limit(
    text: Column, separator: Column, maximum: Column, *, partitioned: bool = False
) -> Column:
    return split_limited(SPLIT, text, separator, maximum, partitioned)


@RULES(".rsplit", Kind.STR, Kind.STR, facts={"partitioned": split_partitioned})
@RULES(".rsplit", Kind.STR, Kind.STR, Kind.INT, facts={"partitioned": split_partitioned})
def rsplit(
    text: Column, separator: Column, maximum: Column | None = None, *, partitioned: bool = False
) -> Column:
    return split_right(SPLIT, text, separator, maximum, partitioned)
