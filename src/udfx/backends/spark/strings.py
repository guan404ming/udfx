from __future__ import annotations

from collections.abc import Sequence
from functools import cache
from typing import cast

import pyspark
from pyspark.sql import Column, functions

from udfx.backends.base import Lowered, RuleResult
from udfx.backends.spark.backend import SparkBackend
from udfx.backends.spark.rules import RULES, when
from udfx.ir import Col
from udfx.semantics import text
from udfx.semantics.text import (
    SIGMA,
    CaseDrift,
    SlicePrimitives,
    case_guard,
    slice_bounds,
    whitespace,
    word,
)
from udfx.types import STR, Kind

INT32_MAX = 2**31 - 1
POINT = "udfx point"


@cache
def case_drift(method: str) -> CaseDrift:
    backend = SparkBackend()

    def native(samples: Sequence[str]) -> list[str]:
        expr = getattr(functions, method)(backend.column(Col(POINT, STR)))
        (recased,) = backend.evaluate([expr], {POINT: STR}, [(sample,) for sample in samples])
        return cast("list[str]", recased)

    return text.case_drift(method, native, f"spark {pyspark.__version__}")


def recase(value: Column, method: str) -> RuleResult[Column]:
    expr = getattr(functions, method)(value)
    exact = case_guard(case_drift(method), value.rlike, value.contains(SIGMA))
    return expr if exact is None else Lowered(expr, exact)


def length(text: Column) -> Column:
    return functions.char_length(text).cast("bigint")


def position(index: Column) -> Column:
    bounded = functions.least(
        functions.greatest(index, functions.lit(-INT32_MAX)), functions.lit(INT32_MAX - 1)
    )
    return functions.when(bounded >= 0, bounded + 1).otherwise(bounded).cast("int")


def characters(text: Column) -> Column:
    return functions.regexp_extract_all(text, functions.lit("(?s)."), functions.lit(0))


def quoted(pattern: Column) -> Column:
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
    return functions.regexp_extract_all(text, functions.lit(word()), functions.lit(0))


RULES(".split", Kind.STR)(split_words)


@RULES(".split", Kind.STR, Kind.STR)
def split(text: Column, separator: Column) -> Column:
    pieces = functions.split(text, quoted(separator), -1)
    return functions.when(separator.isNull(), split_words(text)).otherwise(pieces)


def find(text: Column, pattern: Column) -> Column:
    return (functions.instr(text, pattern) - 1).cast("bigint")


RULES(".find", Kind.STR, Kind.STR)(find)


@RULES(".rfind", Kind.STR, Kind.STR)
def rfind(text: Column, pattern: Column) -> Column:
    found = find(functions.reverse(text), functions.reverse(pattern))
    return functions.when(found < 0, -1).otherwise(length(text) - length(pattern) - found)


@RULES(".count", Kind.STR, Kind.STR)
def count(text: Column, pattern: Column) -> Column:
    removed = length(text) - length(functions.replace(text, pattern, functions.lit("")))
    occurrences = functions.call_function(
        "div", removed, functions.greatest(length(pattern), functions.lit(1))
    )
    return functions.when(length(pattern) == 0, length(text) + 1).otherwise(occurrences)


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


@RULES(".join", Kind.STR, Kind.LIST)
def join(separator: Column, words: Column) -> Column:
    return array_join(words, separator)


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


@RULES("__contains__", Kind.LIST, Kind.STR)
def list_contains(container: Column, item: Column) -> Column:
    return functions.exists(container, lambda element: element.eqNullSafe(item))


@RULES("len", Kind.STR)
def str_len(text: Column) -> Column:
    return length(text)


@RULES("len", Kind.LIST)
def list_len(container: Column) -> Column:
    return functions.size(container).cast("bigint")


@RULES("__getitem__", Kind.STR, Kind.INT)
def str_getitem(text: Column, index: Column) -> Column:
    return functions.substring(text, position(index), functions.lit(1))


@RULES("__getitem__", Kind.LIST, Kind.INT)
def list_getitem(container: Column, index: Column) -> Column:
    return functions.try_element_at(container, position(index))


SLICE = SlicePrimitives(
    literal=functions.lit,
    when=when,
    coalesce=functions.coalesce,
    clip=lambda value, lower, upper: functions.least(functions.greatest(value, lower), upper),
)


@RULES("slice", Kind.STR, Kind.INT, Kind.INT)
def str_slice(text: Column, start: Column, stop: Column) -> Column:
    bounds = slice_bounds(SLICE, length(text), start, stop)
    return functions.substring(text, (bounds.start + 1).cast("int"), bounds.length.cast("int"))
