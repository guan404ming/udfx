from __future__ import annotations

import pyarrow as pa
from datafusion import Expr, functions, lit

from udfx.backends.datafusion.rules import RULES, array_transform, length
from udfx.backends.datafusion.slicing import SEQUENCE
from udfx.backends.datafusion.strings import join, lstrip
from udfx.semantics.facts import split_non_overlapping, split_partitioned
from udfx.semantics.splitting import SplitPrimitives, split_limited, split_right
from udfx.semantics.text import (
    char_class,
    non_whitespace_pattern,
    whitespace,
)
from udfx.types import STR, Kind, list_of


def split_words(text: Expr) -> Expr:
    spaced = functions.regexp_replace(text, lit(char_class(whitespace()) + "+"), lit(" "), lit("g"))
    return functions.string_to_array(functions.btrim(spaced, lit(" ")), lit(" "))


RULES(".split", Kind.STR)(split_words)


RULES(".rsplit", Kind.STR)(split_words)


def split(text: Expr, separator: Expr) -> Expr:
    return (
        functions.when(separator.is_null(), split_words(text))
        .when(text == lit(""), functions.make_array(lit("")))
        .otherwise(functions.string_to_array(text, separator))
    )


RULES(".split", Kind.STR, Kind.STR)(split)


@RULES("last_split", Kind.STR, Kind.STR, facts={"non_overlapping": split_non_overlapping})
def last_split(text: Expr, separator: Expr, *, non_overlapping: bool = False) -> Expr:
    if non_overlapping:
        return functions.split_part(text, separator, lit(-1))
    return functions.array_element(split(text, separator), lit(-1))


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
