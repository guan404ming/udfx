from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from pyspark.sql import Column, functions

from udfx.backends.base import Lowered
from udfx.backends.spark.backend import SparkBackend
from udfx.backends.spark.rules import RULES
from udfx.ir import Col
from udfx.semantics.text import char_class, drift, whitespace, word
from udfx.types import STR, Kind

CASE_DRIFT: dict[str, str | None] = {}

INT32_MAX = 2**31 - 1


def case_drift(method: str) -> str | None:
    if method not in CASE_DRIFT:
        backend = SparkBackend()
        point = Col("point", STR)

        def native(points: Sequence[str]) -> list[str]:
            expr = getattr(functions, method)(backend.column(point))
            rows = [(item,) for item in points]
            return cast("list[str]", backend.evaluate([expr], {point.name: STR}, rows)[0])

        drifted = drift(getattr(str, method), native)
        CASE_DRIFT[method] = char_class(drifted) if drifted else None
    return CASE_DRIFT[method]


def recase(value: Column, method: str) -> Column | Lowered[Column]:
    expr = getattr(functions, method)(value)
    drift = case_drift(method)
    return expr if drift is None else Lowered(expr, ~value.rlike(drift))


def length(a: Column) -> Column:
    return functions.char_length(a).cast("bigint")


def position(index: Column) -> Column:
    bounded = functions.least(
        functions.greatest(index, functions.lit(-INT32_MAX)), functions.lit(INT32_MAX - 1)
    )
    return functions.when(bounded >= 0, bounded + 1).otherwise(bounded).cast("int")


def characters(a: Column) -> Column:
    return functions.regexp_extract_all(a, functions.lit("(?s)."), functions.lit(0))


def quoted(pattern: Column) -> Column:
    escaped = functions.replace(pattern, functions.lit("\\E"), functions.lit("\\E\\\\E\\Q"))
    return functions.concat(functions.lit("\\Q"), escaped, functions.lit("\\E"))


def join(words: Column, separator: Column) -> Column:
    return functions.call_function("array_join", words, separator)


@RULES(".lower", Kind.STR)
def lower(value: Column) -> Column | Lowered[Column]:
    return recase(value, "lower")


@RULES(".upper", Kind.STR)
def upper(value: Column) -> Column | Lowered[Column]:
    return recase(value, "upper")


@RULES(".strip", Kind.STR)
def strip(a: Column) -> Column:
    return functions.btrim(a, functions.lit(whitespace()))


@RULES(".lstrip", Kind.STR)
def lstrip(a: Column) -> Column:
    return functions.ltrim(a, functions.lit(whitespace()))


@RULES(".rstrip", Kind.STR)
def rstrip(a: Column) -> Column:
    return functions.rtrim(a, functions.lit(whitespace()))


@RULES(".split", Kind.STR)
def split_words(a: Column) -> Column:
    return functions.regexp_extract_all(a, functions.lit(word()), functions.lit(0))


@RULES(".split", Kind.STR, Kind.STR)
def split(a: Column, b: Column) -> Column:
    return functions.split(a, quoted(b), -1)


def find(a: Column, b: Column) -> Column:
    return (functions.instr(a, b) - 1).cast("bigint")


RULES(".find", Kind.STR, Kind.STR)(find)


@RULES(".rfind", Kind.STR, Kind.STR)
def rfind(a: Column, b: Column) -> Column:
    found = find(functions.reverse(a), functions.reverse(b))
    return functions.when(found < 0, -1).otherwise(length(a) - length(b) - found)


@RULES(".count", Kind.STR, Kind.STR)
def count(a: Column, b: Column) -> Column:
    removed = length(a) - length(functions.replace(a, b, functions.lit("")))
    occurrences = functions.call_function(
        "div", removed, functions.greatest(length(b), functions.lit(1))
    )
    return functions.when(length(b) == 0, length(a) + 1).otherwise(occurrences)


@RULES(".startswith", Kind.STR, Kind.STR)
def startswith(a: Column, b: Column) -> Column:
    return functions.startswith(a, b)


@RULES(".endswith", Kind.STR, Kind.STR)
def endswith(a: Column, b: Column) -> Column:
    return functions.endswith(a, b)


@RULES(".replace", Kind.STR, Kind.STR, Kind.STR)
def replace(a: Column, old: Column, new: Column) -> Column:
    empty = functions.array(functions.lit(""))
    spread = join(functions.concat(empty, characters(a), empty), new)
    return functions.when(old == "", spread).otherwise(functions.replace(a, old, new))


@RULES(".join", Kind.STR, Kind.LIST)
def join_words(separator: Column, words: Column) -> Column:
    return join(words, separator)


@RULES("__add__", Kind.STR, Kind.STR)
def str_add(a: Column, b: Column) -> Column:
    return functions.concat(a, b)


@RULES("__eq__", Kind.STR, Kind.STR)
def str_eq(a: Column, b: Column) -> Column:
    return a.eqNullSafe(b)


@RULES("__ne__", Kind.STR, Kind.STR)
def str_ne(a: Column, b: Column) -> Column:
    return ~a.eqNullSafe(b)


@RULES("__contains__", Kind.STR, Kind.STR)
def str_contains(a: Column, b: Column) -> Column:
    return a.contains(b)


@RULES("__contains__", Kind.LIST, Kind.STR)
def list_contains(a: Column, b: Column) -> Column:
    return functions.array_contains(a, b)


@RULES("len", Kind.STR)
def str_len(a: Column) -> Column:
    return length(a)


@RULES("len", Kind.LIST)
def list_len(a: Column) -> Column:
    return functions.size(a).cast("bigint")


@RULES("__getitem__", Kind.STR, Kind.INT)
def str_getitem(a: Column, b: Column) -> Column:
    return functions.substring(a, position(b), functions.lit(1))


@RULES("__getitem__", Kind.LIST, Kind.INT)
def list_getitem(container: Column, index: Column) -> Column:
    return functions.try_element_at(container, position(index))


@RULES("slice", Kind.STR, Kind.INT, Kind.INT)
def str_slice(a: Column, b: Column, c: Column) -> Column:
    size = length(a)

    def clamp(bound: Column, default: Column) -> Column:
        bound = functions.coalesce(bound, default)
        shifted = functions.when(bound < 0, bound + size).otherwise(bound)
        return functions.least(functions.greatest(shifted, functions.lit(0)), size)

    start = clamp(b, functions.lit(0))
    stop = clamp(c, size)
    span = functions.greatest(stop - start, functions.lit(0))
    return functions.substring(a, (start + 1).cast("int"), span.cast("int"))
