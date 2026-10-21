from __future__ import annotations

from collections.abc import Sequence
from functools import cache, partial
from typing import cast

import pyspark
from pyspark.sql import Column, functions

from udfx.backends.registration import Lowered, RuleResult
from udfx.backends.spark.backend import SparkBackend
from udfx.backends.spark.rules import RULES, when
from udfx.ir import Col
from udfx.semantics import text
from udfx.semantics.text import (
    SIGMA,
    CaseDrift,
    SearchPrimitives,
    SequencePrimitives,
    SlicePrimitives,
    SplitPrimitives,
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
from udfx.types import STR, Kind

INT32_MAX = 2**31 - 1
POINT = "point"


def native_recase(method: str, samples: Sequence[str]) -> list[str]:
    backend = SparkBackend()
    expr = getattr(functions, method)(backend.column(Col(POINT, STR)))
    (recased,) = backend.evaluate([expr], {POINT: STR}, [(sample,) for sample in samples])
    return cast("list[str]", recased)


@cache
def case_drift(method: str) -> CaseDrift:
    engine = f"spark {pyspark.__version__}"
    return text.case_drift(method, partial(native_recase, method), engine)


def recase(value: Column, method: str) -> RuleResult[Column]:
    expr = getattr(functions, method)(value)
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


@RULES(".lower", Kind.STR)
def lower(value: Column) -> RuleResult[Column]:
    return recase(value, "lower")


@RULES(".upper", Kind.STR)
def upper(value: Column) -> RuleResult[Column]:
    return recase(value, "upper")


@RULES(".strip", Kind.STR)
def strip(text: Column) -> Column:
    return functions.btrim(text, functions.lit(whitespace()))


@RULES(".lstrip", Kind.STR)
def lstrip(text: Column) -> Column:
    return functions.ltrim(text, functions.lit(whitespace()))


@RULES(".rstrip", Kind.STR)
def rstrip(text: Column) -> Column:
    return functions.rtrim(text, functions.lit(whitespace()))


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
    return functions.exists(container, lambda element: element.eqNullSafe(item))


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
    bind=lambda value, transform: functions.transform(functions.array(value), transform)[0],
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
    text: Column, start: Column | int | None, stop: Column | int | None, step: int | None = None
) -> Column:
    return slice_sequence(TEXT, text, start, stop, step)


@RULES("slice", Kind.LIST, Kind.INT, Kind.INT)
@RULES("slice", Kind.LIST, Kind.INT, Kind.INT, Kind.INT)
def list_slice(
    container: Column,
    start: Column | int | None,
    stop: Column | int | None,
    step: int | None = None,
) -> Column:
    return slice_sequence(SEQUENCE, container, start, stop, step)


SPLIT = SplitPrimitives(
    sequence=SEQUENCE,
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
