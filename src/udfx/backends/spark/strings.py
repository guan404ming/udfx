from __future__ import annotations

from collections.abc import Sequence
from functools import cache, partial
from typing import cast

import pyspark
from pyspark.sql import Column, functions

from udfx.backends.registration import (
    Lowered,
    RuleResult,
    register_string_predicates,
    register_translation,
)
from udfx.backends.spark.backend import SparkBackend
from udfx.backends.spark.conversions import TEXT as TRANSLATION
from udfx.backends.spark.rules import RULES, when
from udfx.ir import Col
from udfx.semantics import text
from udfx.semantics.sequences import (
    SearchPrimitives,
    count_occurrences,
    reverse_find,
)
from udfx.semantics.text import (
    CASE_METHODS,
    SIGMA,
    CaseDrift,
    CasePrimitives,
    case_expression,
    case_guard,
    whitespace,
)
from udfx.types import STR, Kind

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


def characters(text: Column) -> Column:
    return functions.array_remove(functions.split(text, "", -1), "")


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


RULES("coalesce", Kind.STR, Kind.STR)(functions.coalesce)


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


register_translation(RULES, TRANSLATION)
