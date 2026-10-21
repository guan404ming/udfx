from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace as replace_record
from functools import cache, partial

import polars as pl

from udfx.backends.polars.rules import RULES, broadcast, literal_value, when
from udfx.backends.registration import (
    Lowered,
    RuleResult,
    register_padding,
    register_string_predicates,
)
from udfx.semantics import text
from udfx.semantics.text import (
    CASE_METHODS,
    CLUSTER_JOINERS,
    SIGMA,
    CaseDrift,
    CasePrimitives,
    PaddingPrimitives,
    SearchPrimitives,
    SequencePrimitives,
    SlicePrimitives,
    SplitPrimitives,
    case_expression,
    case_guard,
    has_prefix_suffix_overlap,
    non_whitespace_pattern,
    reverse_find,
    slice_sequence,
    split_limited,
    split_right,
    whitespace,
    word_prefix_pattern,
)
from udfx.types import Kind


def native_recase(method: str, samples: Sequence[str]) -> list[str]:
    return (
        pl.DataFrame({"udfx point": samples})
        .select(case_expression(CASE, pl.col("udfx point"), method))
        .to_series()
        .to_list()
    )


@cache
def case_drift(method: str) -> CaseDrift:
    return text.case_drift(method, partial(native_recase, method), f"polars {pl.__version__}")


def recase(value: pl.Expr, method: str) -> RuleResult[pl.Expr]:
    expr = case_expression(CASE, value, method)
    sigma = value.str.contains(SIGMA, literal=True)
    exact = case_guard(case_drift(method), value.str.contains, sigma)
    return expr if exact is None else Lowered(expr, exact)


@RULES(".strip", Kind.STR)
@RULES(".strip", Kind.STR, Kind.STR)
def strip(text: pl.Expr, characters: pl.Expr | None = None) -> pl.Expr:
    characters = (
        pl.coalesce(characters, pl.lit(whitespace()))
        if characters is not None
        else pl.lit(whitespace())
    )
    return text.str.strip_chars(characters)


@RULES(".lstrip", Kind.STR)
@RULES(".lstrip", Kind.STR, Kind.STR)
def lstrip(text: pl.Expr, characters: pl.Expr | None = None) -> pl.Expr:
    characters = (
        pl.coalesce(characters, pl.lit(whitespace()))
        if characters is not None
        else pl.lit(whitespace())
    )
    return text.str.strip_chars_start(characters)


@RULES(".rstrip", Kind.STR)
@RULES(".rstrip", Kind.STR, Kind.STR)
def rstrip(text: pl.Expr, characters: pl.Expr | None = None) -> pl.Expr:
    characters = (
        pl.coalesce(characters, pl.lit(whitespace()))
        if characters is not None
        else pl.lit(whitespace())
    )
    return text.str.strip_chars_end(characters)


def split_words(text: pl.Expr) -> pl.Expr:
    return text.str.extract_all(non_whitespace_pattern())


RULES(".split", Kind.STR)(split_words)
RULES(".rsplit", Kind.STR)(split_words)


def split(text: pl.Expr, separator: pl.Expr) -> pl.Expr:
    return when(separator.is_null(), split_words(text), text.str.split(separator))


RULES(".split", Kind.STR, Kind.STR)(split)


def length(text: pl.Expr) -> pl.Expr:
    return text.str.len_chars().cast(pl.Int64)


def find(text: pl.Expr, pattern: pl.Expr) -> pl.Expr:
    prefix = length(text.str.split(pattern).list.first())
    return (
        pl.when(pattern == "")
        .then(0)
        .when(text.str.contains(pattern, literal=True))
        .then(prefix)
        .otherwise(-1)
    )


RULES(".find", Kind.STR, Kind.STR)(find)


def single_code_point_clusters(text: pl.Expr) -> pl.Expr:
    return ~text.str.contains(CLUSTER_JOINERS)


SEARCH = SearchPrimitives(pl.lit, when, length, lambda text: text.str.reverse(), find)


@RULES(".rfind", Kind.STR, Kind.STR)
def rfind(text: pl.Expr, pattern: pl.Expr) -> RuleResult[pl.Expr]:
    known = literal_value(pattern)
    if isinstance(known, str) and known and not has_prefix_suffix_overlap(known):
        suffix = length(text.str.split(pattern).list.last())
        found = length(text) - length(pattern) - suffix
        return when(text.str.contains(pattern, literal=True), found, pl.lit(-1))
    found = reverse_find(SEARCH, text, pattern)
    return Lowered(found, single_code_point_clusters(text) & single_code_point_clusters(pattern))


@RULES(".count", Kind.STR, Kind.STR)
def count(text: pl.Expr, pattern: pl.Expr) -> pl.Expr:
    return broadcast(text).str.count_matches(pattern, literal=True).cast(pl.Int64)


@RULES(".startswith", Kind.STR, Kind.STR)
def startswith(text: pl.Expr, prefix: pl.Expr) -> pl.Expr:
    return text.str.starts_with(prefix)


@RULES(".endswith", Kind.STR, Kind.STR)
def endswith(text: pl.Expr, suffix: pl.Expr) -> pl.Expr:
    return text.str.ends_with(suffix)


@RULES(".replace", Kind.STR, Kind.STR, Kind.STR)
def replace(text: pl.Expr, old: pl.Expr, new: pl.Expr) -> pl.Expr:
    text = broadcast(text)
    padded = pl.concat_list([pl.lit(""), text.str.split(""), pl.lit("")])
    return (
        pl.when(old == "").then(padded.list.join(new)).otherwise(text.str.split(old).list.join(new))
    )


def join(separator: pl.Expr, words: pl.Expr) -> pl.Expr:
    return broadcast(words).list.join(separator)


RULES(".join", Kind.STR, Kind.LIST)(join)


@RULES(".join", Kind.STR, Kind.STR)
def join_characters(separator: pl.Expr, text: pl.Expr) -> pl.Expr:
    return join(separator, text.str.split(""))


@RULES("__add__", Kind.STR, Kind.STR)
def str_add(left: pl.Expr, right: pl.Expr) -> pl.Expr:
    return left + right


@RULES("__eq__", Kind.STR, Kind.STR)
def str_eq(left: pl.Expr, right: pl.Expr) -> pl.Expr:
    return left.eq_missing(right)


@RULES("__ne__", Kind.STR, Kind.STR)
def str_ne(left: pl.Expr, right: pl.Expr) -> pl.Expr:
    return left.ne_missing(right)


@RULES("__contains__", Kind.STR, Kind.STR)
def str_contains(text: pl.Expr, item: pl.Expr) -> pl.Expr:
    return text.str.contains(item, literal=True)


def list_contains(container: pl.Expr, item: pl.Expr) -> pl.Expr:
    return container.list.contains(item, nulls_equal=True)


RULES("__contains__", Kind.LIST, Kind.STR)(list_contains)


@RULES("len", Kind.STR)
def str_len(text: pl.Expr) -> pl.Expr:
    return length(text)


@RULES("len", Kind.LIST)
def list_len(container: pl.Expr) -> pl.Expr:
    return container.list.len().cast(pl.Int64)


@RULES("__getitem__", Kind.STR, Kind.INT)
def str_getitem(text: pl.Expr, index: pl.Expr) -> pl.Expr:
    return text.str.slice(index, 1)


@RULES("__getitem__", Kind.LIST, Kind.INT)
def list_getitem(container: pl.Expr, index: pl.Expr) -> pl.Expr:
    return container.list.get(index.clip(-(2**62), None), null_on_oob=True)


SLICE = SlicePrimitives(
    maximum=2**63 - 1,
    literal=pl.lit,
    when=when,
    coalesce=pl.Expr.fill_null,
    clip=lambda value, lower, upper: pl.min_horizontal(pl.max_horizontal(value, lower), upper),
)


SEQUENCE = SequencePrimitives(
    bounds=replace_record(SLICE, maximum=2**32 - 1),
    length=list_len,
    slice=lambda value, start, length: value.list.slice(start.fill_null(0), length.fill_null(0)),
    head=lambda value, size: (
        value.list.head(size)
        if size >= 0
        else value.list.reverse().list.slice(-size).list.reverse()
    ),
    tail=lambda value, size: value.list.slice(-size),
    reverse=lambda value: value.list.reverse(),
    stride=lambda value, step: value.list.gather_every(step),
    bind=lambda value, transform: (
        pl.concat_list(pl.struct(value.alias("udfx value")))
        .list.eval(transform(pl.element().struct.field("udfx value")))
        .list.first()
    ),
)


TEXT = SequencePrimitives(
    bounds=SLICE,
    length=length,
    slice=lambda value, start, size: value.str.slice(start, size),
    head=lambda value, size: value.str.head(size),
    tail=lambda value, size: value.str.tail(size),
    reverse=lambda value: value.str.split("").list.reverse().list.join(""),
    stride=lambda value, step: SEQUENCE.stride(value.str.split(""), min(step, 2**32 - 1)).list.join(
        ""
    ),
    bind=SEQUENCE.bind,
)


@RULES("slice", Kind.STR, Kind.INT, Kind.INT)
@RULES("slice", Kind.STR, Kind.INT, Kind.INT, Kind.INT)
def str_slice(
    text: pl.Expr, start: pl.Expr | int | None, stop: pl.Expr | int | None, step: int | None = None
) -> pl.Expr:
    return slice_sequence(TEXT, text, start, stop, step)


@RULES("slice", Kind.LIST, Kind.INT, Kind.INT)
@RULES("slice", Kind.LIST, Kind.INT, Kind.INT, Kind.INT)
def list_slice(
    container: pl.Expr,
    start: pl.Expr | int | None,
    stop: pl.Expr | int | None,
    step: int | None = None,
) -> pl.Expr:
    return slice_sequence(SEQUENCE, container, start, stop, step)


SPLIT = SplitPrimitives(
    sequence=SEQUENCE,
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


@RULES(".split", Kind.STR, Kind.STR, Kind.INT)
def split_limit(text: pl.Expr, separator: pl.Expr, maximum: pl.Expr) -> pl.Expr:
    return split_limited(SPLIT, text, separator, maximum)


@RULES(".rsplit", Kind.STR, Kind.STR)
@RULES(".rsplit", Kind.STR, Kind.STR, Kind.INT)
def rsplit(text: pl.Expr, separator: pl.Expr, maximum: pl.Expr | None = None) -> Lowered[pl.Expr]:
    exact = single_code_point_clusters(text) & single_code_point_clusters(separator).fill_null(True)
    return Lowered(split_right(SPLIT, text, separator, maximum), exact)


register_string_predicates(RULES, lambda value, pattern: value.str.contains(pattern))


for method in CASE_METHODS:
    RULES(f".{method}", Kind.STR)(partial(recase, method=method))


CASE = CasePrimitives(
    lower=lambda value: value.str.to_lowercase(),
    upper=lambda value: value.str.to_uppercase(),
    title=lambda value: value.str.to_titlecase(),
    head=lambda value: value.str.slice(0, 1),
    tail=lambda value: value.str.slice(1),
    concat=lambda *values: pl.concat_str(values),
    when=when,
    matches=lambda value, pattern: value.str.contains(pattern),
    characters=lambda value, transform: (
        value.str.split("").list.eval(transform(pl.element())).list.join("")
    ),
)


register_padding(
    RULES,
    PaddingPrimitives(
        sequence=TEXT,
        literal=pl.lit,
        concat=lambda *values: pl.concat_str(values),
        repeat=lambda value, count: value.repeat_by(count).list.join(""),
        half=lambda value: value // 2,
    ),
)
