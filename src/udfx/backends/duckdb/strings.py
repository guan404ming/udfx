from __future__ import annotations

from collections.abc import Sequence

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
from udfx.semantics.text import char_class, drift, whitespace, word
from udfx.types import Kind

CASE_DRIFT: dict[str, str | None] = {}
POINT = "udfx point"
ITEM = "udfx item"
INDEX = "udfx index"


def case_drift(method: str) -> str | None:
    if method not in CASE_DRIFT:

        def native(points: Sequence[str]) -> list[str]:
            table = pa.table({POINT: pa.array(list(points))})
            rows = duckdb.from_arrow(table).select(function(method, column(POINT))).fetchall()
            return [row[0] for row in rows]

        drifted = drift(getattr(str, method), native)
        CASE_DRIFT[method] = char_class(drifted) if drifted else None
    return CASE_DRIFT[method]


def recase(value: Expression, method: str) -> Expression | Lowered[Expression]:
    expr = function(method, value)
    drift = case_drift(method)
    if drift is None:
        return expr
    return Lowered(expr, ~function("regexp_matches", value, constant(drift)))


@RULES(".lower", Kind.STR)
def lower(value: Expression) -> Expression | Lowered[Expression]:
    return recase(value, "lower")


@RULES(".upper", Kind.STR)
def upper(value: Expression) -> Expression | Lowered[Expression]:
    return recase(value, "upper")


@RULES(".strip", Kind.STR)
def strip(a: Expression) -> Expression:
    return function("trim", a, constant(whitespace()))


@RULES(".lstrip", Kind.STR)
def lstrip(a: Expression) -> Expression:
    return function("ltrim", a, constant(whitespace()))


@RULES(".rstrip", Kind.STR)
def rstrip(a: Expression) -> Expression:
    return function("rtrim", a, constant(whitespace()))


@RULES(".split", Kind.STR)
def split_words(a: Expression) -> Expression:
    return function("regexp_extract_all", a, constant(word()))


@RULES(".split", Kind.STR, Kind.STR)
def split(a: Expression, b: Expression) -> Expression:
    return function("string_split", a, b)


def find(a: Expression, b: Expression) -> Expression:
    return function("instr", a, b) - constant(1)


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
def startswith(a: Expression, b: Expression) -> Expression:
    return function("starts_with", a, b)


@RULES(".endswith", Kind.STR, Kind.STR)
def endswith(a: Expression, b: Expression) -> Expression:
    return function("ends_with", a, b)


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
def str_add(a: Expression, b: Expression) -> Expression:
    return function("concat", a, b)


RULES("__eq__", Kind.STR, Kind.STR)(nullable_equal)


@RULES("__ne__", Kind.STR, Kind.STR)
def str_ne(a: Expression, b: Expression) -> Expression:
    return ~nullable_equal(a, b)


@RULES("__contains__", Kind.STR, Kind.STR)
def str_contains(a: Expression, b: Expression) -> Expression:
    return function("contains", a, b)


@RULES("__contains__", Kind.LIST, Kind.STR)
def list_contains(a: Expression, b: Expression) -> Expression:
    return function("list_contains", a, b)


RULES("len", Kind.STR)(length)
RULES("len", Kind.LIST)(length)


def position(container: Expression, index: Expression) -> Expression:
    size = length(container)
    bounded = function("greatest", function("least", index, size), -size - constant(1))
    return when(bounded < constant(0), bounded, bounded + constant(1))


@RULES("__getitem__", Kind.STR, Kind.INT)
def str_getitem(a: Expression, b: Expression) -> Expression:
    return function("array_extract", a, position(a, b))


@RULES("__getitem__", Kind.LIST, Kind.INT)
def list_getitem(container: Expression, index: Expression) -> Expression:
    return function("list_extract", container, position(container, index))


@RULES("slice", Kind.STR, Kind.INT, Kind.INT)
def str_slice(a: Expression, b: Expression, c: Expression) -> Expression:
    size = length(a)

    def clamp(bound: Expression, default: Expression) -> Expression:
        bound = duckdb.CoalesceOperator(bound, default)
        shifted = when(bound < constant(0), bound + size, bound)
        return function("least", function("greatest", shifted, constant(0)), size)

    start = clamp(b, constant(0))
    stop = clamp(c, size)
    width = function("greatest", stop - start, constant(0))
    return function("substring", a, start + constant(1), width)
