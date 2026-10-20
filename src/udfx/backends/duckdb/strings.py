from __future__ import annotations

from collections.abc import Sequence
from functools import cache

import duckdb
import pyarrow as pa
from duckdb import Expression

from udfx.backends.base import Lowered, RuleResult
from udfx.backends.duckdb.rules import (
    RULES,
    column,
    constant,
    function,
    length,
    null_safe_equal,
    when,
)
from udfx.semantics import text
from udfx.semantics.text import (
    CLUSTER_JOINERS,
    SIGMA,
    CaseDrift,
    SlicePrimitives,
    case_guard,
    slice_bounds,
    whitespace,
    word,
)
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


def recase(value: Expression, method: str) -> RuleResult[Expression]:
    expr = function(method, value)
    sigma = function("contains", value, constant(SIGMA))
    exact = case_guard(
        case_drift(method),
        lambda pattern: function("regexp_matches", value, constant(pattern)),
        sigma,
    )
    return expr if exact is None else Lowered(expr, exact)


@RULES(".lower", Kind.STR)
def lower(value: Expression) -> RuleResult[Expression]:
    return recase(value, "lower")


@RULES(".upper", Kind.STR)
def upper(value: Expression) -> RuleResult[Expression]:
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


def split_words(text: Expression) -> Expression:
    return function("regexp_extract_all", text, constant(word()))


RULES(".split", Kind.STR)(split_words)


@RULES(".split", Kind.STR, Kind.STR)
def split(text: Expression, separator: Expression) -> Expression:
    return when(separator.isnull(), split_words(text), function("string_split", text, separator))


def find(text: Expression, pattern: Expression) -> Expression:
    return function("instr", text, pattern) - constant(1)


RULES(".find", Kind.STR, Kind.STR)(find)


def single_code_point_clusters(text: Expression) -> Expression:
    return ~function("regexp_matches", text, constant(CLUSTER_JOINERS))


@RULES(".rfind", Kind.STR, Kind.STR)
def rfind(text: Expression, pattern: Expression) -> Lowered[Expression]:
    position = find(function("reverse", text), function("reverse", pattern))
    found = when(position < constant(0), constant(-1), length(text) - length(pattern) - position)
    return Lowered(found, single_code_point_clusters(text) & single_code_point_clusters(pattern))


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


RULES("__eq__", Kind.STR, Kind.STR)(null_safe_equal)


@RULES("__ne__", Kind.STR, Kind.STR)
def str_ne(left: Expression, right: Expression) -> Expression:
    return ~null_safe_equal(left, right)


@RULES("__contains__", Kind.STR, Kind.STR)
def str_contains(text: Expression, item: Expression) -> Expression:
    return function("contains", text, item)


@RULES("__contains__", Kind.LIST, Kind.STR)
def list_contains(container: Expression, item: Expression) -> Expression:
    return function("list_position", container, item).isnotnull()


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


SLICE = SlicePrimitives(
    literal=constant,
    when=when,
    coalesce=duckdb.CoalesceOperator,
    clip=lambda value, lower, upper: function("least", function("greatest", value, lower), upper),
)


@RULES("slice", Kind.STR, Kind.INT, Kind.INT)
def str_slice(text: Expression, start: Expression, stop: Expression) -> Expression:
    bounds = slice_bounds(SLICE, length(text), start, stop)
    return function("substring", text, bounds.start + constant(1), bounds.length)
