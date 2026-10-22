from __future__ import annotations

import polars as pl

from udfx.backends.polars.rules import RULES, when
from udfx.backends.polars.slicing import SEQUENCE
from udfx.backends.polars.strings import join, length, lstrip, single_code_point_clusters
from udfx.backends.registration import Lowered
from udfx.semantics.facts import split_partitioned
from udfx.semantics.splitting import SplitPrimitives, split_limited, split_right
from udfx.semantics.text import (
    non_whitespace_pattern,
    word_prefix_pattern,
)
from udfx.types import STR, Kind, list_of


def split_words(text: pl.Expr) -> pl.Expr:
    return text.str.extract_all(non_whitespace_pattern())


RULES(".split", Kind.STR)(split_words)


RULES(".rsplit", Kind.STR)(split_words)


def split(text: pl.Expr, separator: pl.Expr) -> pl.Expr:
    return when(separator.is_null(), split_words(text), text.str.split(separator))


RULES(".split", Kind.STR, Kind.STR)(split)


@RULES("last_split", Kind.STR, Kind.STR)
def last_split(text: pl.Expr, separator: pl.Expr) -> pl.Expr:
    return text.str.split(separator).list.last()


SPLIT = SplitPrimitives(
    sequence=SEQUENCE,
    bind=lambda values, transform: SEQUENCE.bind(
        pl.struct(value.alias(str(index)) for index, value in enumerate(values)),
        lambda value: transform(*(value.struct.field(str(index)) for index in range(len(values)))),
        list_of(STR),
    ),
    split=split,
    join=join,
    append=lambda values, value: values.list.concat(value),
    length=length,
    suffix=lambda value, start: value.str.slice(start),
    word_prefix=lambda value, count: (
        value.str.extract_all(word_prefix_pattern()).list.slice(0, count).list.join("")
    ),
    lstrip=lstrip,
    is_null=pl.Expr.is_null,
    reverse=lambda value: value.str.reverse(),
    restore=lambda values: values.list.reverse().list.eval(pl.element().str.reverse()),
)


@RULES(".split", Kind.STR, Kind.STR, Kind.INT, facts={"partitioned": split_partitioned})
def split_limit(
    text: pl.Expr, separator: pl.Expr, maximum: pl.Expr, *, partitioned: bool = False
) -> pl.Expr:
    return split_limited(SPLIT, text, separator, maximum, partitioned)


@RULES(".rsplit", Kind.STR, Kind.STR, facts={"partitioned": split_partitioned})
@RULES(".rsplit", Kind.STR, Kind.STR, Kind.INT, facts={"partitioned": split_partitioned})
def rsplit(
    text: pl.Expr, separator: pl.Expr, maximum: pl.Expr | None = None, *, partitioned: bool = False
) -> Lowered[pl.Expr]:
    exact = (
        None
        if partitioned
        else single_code_point_clusters(text)
        & single_code_point_clusters(separator).fill_null(True)
    )
    return Lowered(split_right(SPLIT, text, separator, maximum, partitioned), exact)
