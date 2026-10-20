from __future__ import annotations

import operator
from collections.abc import Sequence
from functools import cache, reduce

import duckdb
import pyarrow as pa
from duckdb import Expression

from udfx.backends.base import Lowered
from udfx.backends.duckdb.rules import (
    RULES,
    column,
    constant,
    function,
    length,
    nullable_equal,
    when,
)
from udfx.semantics import text
from udfx.semantics.text import SIGMA, CaseDrift, whitespace, word
from udfx.types import Kind

POINT = "udfx point"
ITEM = "udfx item"
INDEX = "udfx index"


@cache
def case_drift(method: str) -> CaseDrift:
    def native(samples: Sequence[str]) -> list[str]:
        table = pa.table({POINT: pa.array(list(samples))})
        rows = duckdb.from_arrow(table).select(function(method, column(POINT))).fetchall()
        return [row[0] for row in rows]

    return text.case_drift(method, native, f"duckdb {duckdb.__version__}")


def recase(value: Expression, method: str) -> Expression | Lowered[Expression]:
    expr = function(method, value)
    found = case_drift(method)
    guards = [~function("regexp_matches", value, constant(found.alone))] if found.alone else []
    if found.nearby:
        sigma = function("contains", value, constant(SIGMA))
        guards.append(~(sigma & function("regexp_matches", value, constant(found.nearby))))
    return Lowered(expr, reduce(operator.and_, guards)) if guards else expr


@RULES(".lower", Kind.STR)
def lower(value: Expression) -> Expression | Lowered[Expression]:
    return recase(value, "lower")


@RULES(".upper", Kind.STR)
def upper(value: Expression) -> Expression | Lowered[Expression]:
    return recase(value, "upper")


@RULES(".strip", Kind.STR)
def strip(text: Expression) -> Expression:
    return function("trim", text, constant(whitespace()))


@RULES(".lstrip", Kind.STR)
def lstrip(text: Expression) -> Expression:
    return function("ltrim", text, constant(whitespace()))


@RULES(".rstrip", Kind.STR)
def rstrip(text: Expression) -> Expression:
    return function("rtrim", text, constant(whitespace()))


@RULES(".split", Kind.STR)
def split_words(text: Expression) -> Expression:
    return function("regexp_extract_all", text, constant(word()))


@RULES(".split", Kind.STR, Kind.STR)
def split(text: Expression, separator: Expression) -> Expression:
    return function("string_split", text, separator)


def find(text: Expression, pattern: Expression) -> Expression:
    return function("instr", text, pattern) - constant(1)


RULES(".find", Kind.STR, Kind.STR)(find)


def reverse(text: Expression) -> Expression:
    parts = function("list_reverse", function("string_split", text, constant("")))
    return function("array_to_string", parts, constant(""))


@RULES(".rfind", Kind.STR, Kind.STR)
def rfind(text: Expression, pattern: Expression) -> Expression:
    position = find(reverse(text), reverse(pattern))
    return when(position < constant(0), constant(-1), length(text) - length(pattern) - position)


@RULES(".count", Kind.STR, Kind.STR)
def count(text: Expression, pattern: Expression) -> Expression:
    pieces = length(function("string_split", text, pattern)) - constant(1)
    return when(pattern == constant(""), length(text) + constant(1), pieces)


@RULES(".startswith", Kind.STR, Kind.STR)
def startswith(text: Expression, prefix: Expression) -> Expression:
    return function("starts_with", text, prefix)


@RULES(".endswith", Kind.STR, Kind.STR)
def endswith(text: Expression, suffix: Expression) -> Expression:
    return function("ends_with", text, suffix)


@RULES(".replace", Kind.STR, Kind.STR, Kind.STR)
def replace(text: Expression, old: Expression, new: Expression) -> Expression:
    escaped = function("replace", new, constant("\\"), constant("\\\\"))
    between = function("regexp_replace", text, constant(""), escaped, constant("g"))
    return when(old == constant(""), between, function("replace", text, old, new))


@RULES(".join", Kind.STR, Kind.LIST)
def join(separator: Expression, words: Expression) -> Expression:
    first = column(INDEX) == constant(1)
    prefixed = when(first, column(ITEM), function("concat", separator, column(ITEM)))
    pieces = function("list_transform", words, duckdb.LambdaExpression((ITEM, INDEX), prefixed))
    return function("array_to_string", pieces, constant(""))


@RULES("__add__", Kind.STR, Kind.STR)
def str_add(left: Expression, right: Expression) -> Expression:
    return function("concat", left, right)


RULES("__eq__", Kind.STR, Kind.STR)(nullable_equal)


@RULES("__ne__", Kind.STR, Kind.STR)
def str_ne(left: Expression, right: Expression) -> Expression:
    return ~nullable_equal(left, right)


@RULES("__contains__", Kind.STR, Kind.STR)
def str_contains(text: Expression, item: Expression) -> Expression:
    return function("contains", text, item)


@RULES("__contains__", Kind.LIST, Kind.STR)
def list_contains(container: Expression, item: Expression) -> Expression:
    return item.isnotnull() & function("list_contains", container, item)


RULES("len", Kind.STR)(length)
RULES("len", Kind.LIST)(length)


def position(container: Expression, index: Expression) -> Expression:
    size = length(container)
    bounded = function("greatest", function("least", index, size), -size - constant(1))
    return when(bounded < constant(0), bounded, bounded + constant(1))


@RULES("__getitem__", Kind.STR, Kind.INT)
def str_getitem(text: Expression, index: Expression) -> Expression:
    return function("array_extract", text, position(text, index))


@RULES("__getitem__", Kind.LIST, Kind.INT)
def list_getitem(container: Expression, index: Expression) -> Expression:
    return function("list_extract", container, position(container, index))


@RULES("slice", Kind.STR, Kind.INT, Kind.INT)
def str_slice(text: Expression, start: Expression, stop: Expression) -> Expression:
    size = length(text)

    def clamp(bound: Expression, default: Expression) -> Expression:
        bound = duckdb.CoalesceOperator(bound, default)
        shifted = when(bound < constant(0), bound + size, bound)
        return function("least", function("greatest", shifted, constant(0)), size)

    begin = clamp(start, constant(0))
    end = clamp(stop, size)
    width = function("greatest", end - begin, constant(0))
    return function("substring", text, begin + constant(1), width)
