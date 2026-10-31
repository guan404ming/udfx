from __future__ import annotations

from collections.abc import Sequence
from functools import cache, partial

import duckdb
import pyarrow as pa
from duckdb import Expression

from udfx.backends.duckdb.conversions import TEXT as TRANSLATION
from udfx.backends.duckdb.rules import (
    ASCII,
    ITEM,
    RULES,
    column,
    concatenate,
    constant,
    function,
    length,
    null_safe_equal,
    trim,
    when,
)
from udfx.backends.registration import (
    Lowered,
    RuleResult,
    register_string_predicates,
    register_translation,
)
from udfx.frontend.core.sequences import join_words, word_item
from udfx.frontend.core.splitting import joined_tokens
from udfx.frontend.core.text import folded_case_chain
from udfx.semantics import casing, text
from udfx.semantics.clusters import CLUSTER_JOINERS
from udfx.semantics.facts import pattern_is_empty
from udfx.semantics.sequences import SearchPrimitives, count_occurrences, reverse_find
from udfx.semantics.text import (
    CASE_METHODS,
    SIGMA,
    CaseDrift,
    CasePrimitives,
    case_expression,
    case_guard,
    whitespace,
)
from udfx.semantics.text_predicates import unicode_match
from udfx.types import INT64_MAX, Kind

POINT = "udfx point"
INDEX = "udfx index"


def native_recase(method: str, samples: Sequence[str]) -> list[str]:
    table = pa.table({POINT: pa.array(list(samples))})
    rows = duckdb.from_arrow(table).select(case_expression(CASE, column(POINT), method)).fetchall()
    return [row[0] for row in rows]


@cache
def case_drift(method: str) -> CaseDrift:
    engine = f"duckdb {duckdb.__version__}"
    return text.case_drift(method, partial(native_recase, method), engine)


def recase(value: Expression, method: str) -> RuleResult[Expression]:
    expr = case_expression(CASE, value, method)
    exact = case_guard(
        case_drift(method),
        partial(unicode_match, ASCII, CASE.matches, value),
        function("contains", value, constant(SIGMA)),
    )
    return expr if exact is None else Lowered(expr, exact)


NATIVE_TRIMS = {"strip": "trim", "lstrip": "ltrim", "rstrip": "rtrim"}


def strip_text(
    text: Expression, characters: Expression | str | None = None, *, method: str
) -> Expression:
    if not isinstance(characters, Expression):
        return text if characters == "" else trim(text, characters or whitespace(), method)
    return function(
        NATIVE_TRIMS[method], text, duckdb.CoalesceOperator(characters, constant(whitespace()))
    )


for _method in NATIVE_TRIMS:
    RULES(f".{_method}", Kind.STR)(partial(strip_text, method=_method))
    RULES(f".{_method}", Kind.STR, Kind.STR, literals=(1,))(partial(strip_text, method=_method))


def find(text: Expression, pattern: Expression) -> Expression:
    return function("instr", text, pattern) - constant(1)


RULES(".find", Kind.STR, Kind.STR)(find)


def single_code_point_clusters(text: Expression) -> Expression:
    return ~function("regexp_matches", text, constant(CLUSTER_JOINERS))


SEARCH = SearchPrimitives(constant, when, length, lambda text: function("reverse", text), find)


@RULES(".rfind", Kind.STR, Kind.STR)
def rfind(text: Expression, pattern: Expression) -> Lowered[Expression]:
    found = reverse_find(SEARCH, text, pattern)
    return Lowered(found, single_code_point_clusters(text) & single_code_point_clusters(pattern))


@RULES(".count", Kind.STR, Kind.STR)
def count(text: Expression, pattern: Expression) -> Expression:
    pieces = length(function("string_split", text, pattern)) - constant(1)
    return count_occurrences(SEARCH, text, pattern, pieces)


@RULES(".startswith", Kind.STR, Kind.STR)
def startswith(text: Expression, prefix: Expression) -> Expression:
    return function("starts_with", text, prefix)


@RULES(".endswith", Kind.STR, Kind.STR)
def endswith(text: Expression, suffix: Expression) -> Expression:
    return function("ends_with", text, suffix)


@RULES(".replace", Kind.STR, Kind.STR, Kind.STR, facts={"empty": pattern_is_empty})
def replace(
    text: Expression, old: Expression, new: Expression, *, empty: bool | None = None
) -> Expression:
    if empty is False:
        return function("replace", text, old, new)
    escaped = function("replace", new, constant("\\"), constant("\\\\"))
    between = function("regexp_replace", text, constant(""), escaped, constant("g"))
    if empty is True:
        return between
    return when(old == constant(""), between, function("replace", text, old, new))


def join(separator: Expression | str | None, words: Expression) -> Expression:
    if not isinstance(separator, Expression):
        return duckdb.CoalesceOperator(
            function("array_to_string", words, constant(separator)), constant("")
        )
    first = column(INDEX) == constant(1)
    prefixed = when(first, column(ITEM), function("concat", separator, column(ITEM)))
    pieces = function("list_transform", words, duckdb.LambdaExpression((ITEM, INDEX), prefixed))
    return concatenate(pieces)


RULES(".join", Kind.STR, Kind.LIST, literals=(0,))(join)


@RULES(".join", Kind.STR, Kind.STR)
def join_characters(separator: Expression, text: Expression) -> Expression:
    return join(separator, function("string_split", text, constant("")))


RULES("coalesce", Kind.STR, Kind.STR)(duckdb.CoalesceOperator)


@RULES("__add__", Kind.STR, Kind.STR, linear=True)
def str_add(left: Expression, right: Expression) -> Expression:
    return function("concat", left, right)


RULES("__eq__", Kind.STR, Kind.STR)(null_safe_equal)


@RULES("__ne__", Kind.STR, Kind.STR)
def str_ne(left: Expression, right: Expression) -> Expression:
    return ~null_safe_equal(left, right)


@RULES("__contains__", Kind.STR, Kind.STR)
def str_contains(text: Expression, item: Expression) -> Expression:
    return function("contains", text, item)


def list_contains(container: Expression, item: Expression) -> Expression:
    return function("list_position", container, item).isnotnull()


RULES("len", Kind.STR, linear=True)(length)
RULES("len", Kind.LIST)(length)


def bounded_native_index(container: Expression, index: Expression) -> Expression:
    size = length(container)
    bounded = function("greatest", function("least", index, size), -size - constant(1))
    return when(bounded < constant(0), bounded, bounded + constant(1))


@RULES("__getitem__", Kind.STR, Kind.INT)
def str_getitem(text: Expression, index: Expression) -> Expression:
    return function("array_extract", text, bounded_native_index(text, index))


@RULES("__getitem__", Kind.LIST, Kind.INT)
@RULES("optional_item", Kind.LIST, Kind.INT)
def list_getitem(container: Expression, index: Expression) -> Expression:
    following = function("nullif", index, constant(INT64_MAX)) + constant(1)
    position = when(index < constant(0), index, following)
    return function("list_extract", container, position)


register_string_predicates(
    RULES, lambda value, pattern: function("regexp_matches", value, constant(pattern))
)
RULES.rewrite(".join", Kind.STR, Kind.LIST)(lambda node: joined_tokens(node) or join_words(node))
RULES.rewrite("__getitem__", Kind.LIST, Kind.INT)(word_item)
RULES.rewrite(".casefold", Kind.STR)(folded_case_chain)


for method in CASE_METHODS:
    RULES(f".{method}", Kind.STR)(partial(recase, method=method))


CASE = CasePrimitives(
    lower=lambda value: function("lower", value),
    upper=lambda value: function("upper", value),
    title=lambda value: casing.title_words(
        CASE,
        value,
        lambda value, pattern, transform: concatenate(
            function(
                "list_transform",
                function("regexp_extract_all", value, constant(pattern)),
                duckdb.LambdaExpression(ITEM, transform(column(ITEM))),
            ),
        ),
    ),
    head=lambda value: function("substring", value, constant(1), constant(1)),
    tail=lambda value: function("substring", value, constant(2)),
    concat=lambda *values: function("concat", *values),
    when=when,
    matches=lambda value, pattern: function("regexp_matches", value, constant(pattern)),
    characters=lambda value, transform: concatenate(
        function(
            "list_transform",
            function("string_split", value, constant("")),
            duckdb.LambdaExpression(ITEM, transform(column(ITEM))),
        ),
    ),
)


register_translation(RULES, TRANSLATION)
