from __future__ import annotations

from collections.abc import Sequence
from functools import cache, partial
from typing import cast

import pyspark
from pyspark.sql import Column, functions

from udfx.backends.registration import (
    Lowered,
    RuleResult,
    register_padding,
    register_string_predicates,
    register_translation,
)
from udfx.backends.spark.backend import SparkBackend
from udfx.backends.spark.conversions import TEXT as TRANSLATION
from udfx.backends.spark.rules import RULES, when
from udfx.ir import Col
from udfx.semantics import text
from udfx.semantics.text import (
    CASE_METHODS,
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
    count_occurrences,
    non_whitespace_pattern,
    reverse_find,
    slice_sequence,
    split_limited,
    split_right,
    whitespace,
    word_prefix_pattern,
)
from udfx.types import STR, Kind, Type, list_of

INT32_MAX = 2**31 - 1
POINT = "point"


def native_recase(method: str, samples: Sequence[str]) -> list[str]:
    backend = SparkBackend()
    expr = case_expression(CASE, backend.column(Col(POINT, STR)), method)
    (recased,) = backend.evaluate([expr], {POINT: STR}, [(sample,) for sample in samples])
    return cast("list[str]", recased)


@cache
def case_drift(method: str) -> CaseDrift:
    engine = f"spark {pyspark.__version__}"
    return text.case_drift(method, partial(native_recase, method), engine)


def recase(value: Column, method: str) -> RuleResult[Column]:
    expr = case_expression(CASE, value, method)
    exact = case_guard(case_drift(method), value.rlike, value.contains(SIGMA))
    return expr if exact is None else Lowered(expr, exact)


def length(text: Column) -> Column:
    return functions.char_length(text).cast("bigint")


def bounded_native_index(index: Column) -> Column:
    bounded = functions.least(
        functions.greatest(index, functions.lit(-INT32_MAX)), functions.lit(INT32_MAX - 1)
    )
    return functions.when(bounded >= 0, bounded + 1).otherwise(bounded).cast("int")


def characters(text: Column) -> Column:
    return functions.regexp_extract_all(text, functions.lit("(?s)."), functions.lit(0))


def quote_java_regex(pattern: Column) -> Column:
    escaped = functions.replace(pattern, functions.lit("\\E"), functions.lit("\\E\\\\E\\Q"))
    return functions.concat(functions.lit("\\Q"), escaped, functions.lit("\\E"))


def array_join(words: Column, separator: Column) -> Column:
    return functions.call_function("array_join", words, separator)


@RULES(".strip", Kind.STR)
@RULES(".strip", Kind.STR, Kind.STR)
def strip(text: Column, characters: Column | None = None) -> Column:
    characters = (
        functions.coalesce(characters, functions.lit(whitespace()))
        if characters is not None
        else functions.lit(whitespace())
    )
    return functions.btrim(text, characters)


@RULES(".lstrip", Kind.STR)
@RULES(".lstrip", Kind.STR, Kind.STR)
def lstrip(text: Column, characters: Column | None = None) -> Column:
    characters = (
        functions.coalesce(characters, functions.lit(whitespace()))
        if characters is not None
        else functions.lit(whitespace())
    )
    return functions.ltrim(text, characters)


@RULES(".rstrip", Kind.STR)
@RULES(".rstrip", Kind.STR, Kind.STR)
def rstrip(text: Column, characters: Column | None = None) -> Column:
    characters = (
        functions.coalesce(characters, functions.lit(whitespace()))
        if characters is not None
        else functions.lit(whitespace())
    )
    return functions.rtrim(text, characters)


def split_words(text: Column) -> Column:
    return functions.regexp_extract_all(
        text, functions.lit(non_whitespace_pattern()), functions.lit(0)
    )


RULES(".split", Kind.STR)(split_words)
RULES(".rsplit", Kind.STR)(split_words)


def split(text: Column, separator: Column) -> Column:
    pieces = functions.split(text, quote_java_regex(separator), -1)
    return functions.when(separator.isNull(), split_words(text)).otherwise(pieces)


RULES(".split", Kind.STR, Kind.STR)(split)


@RULES("last_split", Kind.STR, Kind.STR)
def last_split(text: Column, separator: Column) -> Column:
    return functions.split_part(text, separator, functions.lit(-1))


def find(text: Column, pattern: Column) -> Column:
    return (functions.instr(text, pattern) - 1).cast("bigint")


RULES(".find", Kind.STR, Kind.STR)(find)


SEARCH = SearchPrimitives(functions.lit, when, length, functions.reverse, find)


@RULES(".rfind", Kind.STR, Kind.STR)
def rfind(text: Column, pattern: Column) -> Column:
    return reverse_find(SEARCH, text, pattern)


@RULES(".count", Kind.STR, Kind.STR)
def count(text: Column, pattern: Column) -> Column:
    removed = length(text) - length(functions.replace(text, pattern, functions.lit("")))
    occurrences = functions.call_function(
        "div", removed, functions.greatest(length(pattern), functions.lit(1))
    )
    return count_occurrences(SEARCH, text, pattern, occurrences)


@RULES(".startswith", Kind.STR, Kind.STR)
def startswith(text: Column, prefix: Column) -> Column:
    return functions.startswith(text, prefix)


@RULES(".endswith", Kind.STR, Kind.STR)
def endswith(text: Column, suffix: Column) -> Column:
    return functions.endswith(text, suffix)


@RULES(".replace", Kind.STR, Kind.STR, Kind.STR)
def replace(text: Column, old: Column, new: Column) -> Column:
    empty = functions.array(functions.lit(""))
    spread = array_join(functions.concat(empty, characters(text), empty), new)
    return functions.when(old == "", spread).otherwise(functions.replace(text, old, new))


def join(separator: Column, words: Column) -> Column:
    return array_join(words, separator)


RULES(".join", Kind.STR, Kind.LIST)(join)


@RULES(".join", Kind.STR, Kind.STR)
def join_characters(separator: Column, text: Column) -> Column:
    return join(separator, characters(text))


@RULES("__add__", Kind.STR, Kind.STR)
def str_add(left: Column, right: Column) -> Column:
    return functions.concat(left, right)


@RULES("__eq__", Kind.STR, Kind.STR)
def str_eq(left: Column, right: Column) -> Column:
    return left.eqNullSafe(right)


@RULES("__ne__", Kind.STR, Kind.STR)
def str_ne(left: Column, right: Column) -> Column:
    return ~left.eqNullSafe(right)


@RULES("__contains__", Kind.STR, Kind.STR)
def str_contains(text: Column, item: Column) -> Column:
    return text.contains(item)


def list_contains(container: Column, item: Column) -> Column:
    return functions.size(functions.array_intersect(container, functions.array(item))) > 0


RULES("__contains__", Kind.LIST, Kind.STR)(list_contains)


@RULES("len", Kind.STR)
def str_len(text: Column) -> Column:
    return length(text)


@RULES("len", Kind.LIST)
def list_len(container: Column) -> Column:
    return functions.size(container).cast("bigint")


@RULES("__getitem__", Kind.STR, Kind.INT)
def str_getitem(text: Column, index: Column) -> Column:
    return functions.substring(text, bounded_native_index(index), functions.lit(1))


@RULES("__getitem__", Kind.LIST, Kind.INT)
def list_getitem(container: Column, index: Column) -> Column:
    return functions.try_element_at(container, bounded_native_index(index))


SLICE = SlicePrimitives(
    maximum=INT32_MAX,
    literal=functions.lit,
    when=when,
    coalesce=functions.coalesce,
    clip=lambda value, lower, upper: functions.least(functions.greatest(value, lower), upper),
)


SEQUENCE = SequencePrimitives(
    bounds=SLICE,
    length=list_len,
    slice=lambda value, start, size: functions.slice(
        value, (start + 1).cast("int"), size.cast("int")
    ),
    head=lambda value, size: (
        functions.slice(value, 1, size)
        if size >= 0
        else functions.reverse(
            functions.slice(functions.reverse(value), 1 - size, INT32_MAX + size)
        )
    ),
    tail=lambda value, size: (
        functions.reverse(functions.slice(functions.reverse(value), 1, size))
        if size > 0
        else functions.slice(value, 1 - size, INT32_MAX + size)
    ),
    reverse=functions.reverse,
    stride=lambda value, step: functions.filter(value, lambda item, index: index % step == 0),
    bind=lambda value, transform, result: functions.transform(functions.array(value), transform)[0],
)


TEXT = SequencePrimitives(
    bounds=SLICE,
    length=length,
    slice=lambda value, start, size: functions.substring(
        value, (start + 1).cast("int"), size.cast("int")
    ),
    head=lambda value, size: (
        functions.substring(value, 1, size)
        if size >= 0
        else functions.reverse(
            functions.substring(functions.reverse(value), min(1 - size, INT32_MAX), INT32_MAX)
        )
    ),
    tail=lambda value, size: functions.substring(
        value, -size if size > 0 else min(1 - size, INT32_MAX), INT32_MAX
    ),
    reverse=functions.reverse,
    stride=lambda value, step: join(functions.lit(""), SEQUENCE.stride(characters(value), step)),
    bind=SEQUENCE.bind,
)


@RULES("slice", Kind.STR, Kind.INT, Kind.INT)
@RULES("slice", Kind.STR, Kind.INT, Kind.INT, Kind.INT)
def str_slice(
    text: Column,
    start: Column | int | None,
    stop: Column | int | None,
    step: int | None = None,
    *,
    result: Type,
) -> Column:
    return slice_sequence(TEXT, text, start, stop, step, result=result)


@RULES("slice", Kind.LIST, Kind.INT, Kind.INT)
@RULES("slice", Kind.LIST, Kind.INT, Kind.INT, Kind.INT)
def list_slice(
    container: Column,
    start: Column | int | None,
    stop: Column | int | None,
    step: int | None = None,
    *,
    result: Type,
) -> Column:
    return slice_sequence(SEQUENCE, container, start, stop, step, result=result)


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


@RULES(".split", Kind.STR, Kind.STR, Kind.INT)
def split_limit(text: Column, separator: Column, maximum: Column) -> Column:
    return split_limited(SPLIT, text, separator, maximum)


@RULES(".rsplit", Kind.STR, Kind.STR)
@RULES(".rsplit", Kind.STR, Kind.STR, Kind.INT)
def rsplit(text: Column, separator: Column, maximum: Column | None = None) -> Column:
    return split_right(SPLIT, text, separator, maximum)


register_string_predicates(RULES, lambda value, pattern: value.rlike(pattern))


for method in CASE_METHODS:
    RULES(f".{method}", Kind.STR)(partial(recase, method=method))


CASE = CasePrimitives(
    lower=functions.lower,
    upper=functions.upper,
    title=functions.initcap,
    head=lambda value: functions.substring(value, 1, 1),
    tail=lambda value: functions.substring(value, 2, INT32_MAX),
    concat=functions.concat,
    when=when,
    matches=lambda value, pattern: value.rlike(pattern),
    characters=lambda value, transform: array_join(
        functions.transform(characters(value), transform), functions.lit("")
    ),
)


register_padding(
    RULES,
    PaddingPrimitives(
        sequence=TEXT,
        literal=functions.lit,
        concat=functions.concat,
        repeat=lambda value, count: functions.repeat(value, count.cast("int")),
        half=lambda value: functions.call_function("div", value, functions.lit(2)),
    ),
)


register_translation(RULES, TRANSLATION)
