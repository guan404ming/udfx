from __future__ import annotations

from collections.abc import Sequence
from functools import cache, partial
from typing import cast

import datafusion
import pyarrow as pa
from datafusion import Expr, col, functions, lit

from udfx.backends.datafusion.backend import DataFusionBackend
from udfx.backends.datafusion.conversions import TEXT as TRANSLATION
from udfx.backends.datafusion.rules import (
    RULES,
    array_transform,
    length,
    null_safe_equal,
    typed_literal,
    when,
)
from udfx.backends.registration import (
    Lowered,
    RuleResult,
    register_string_predicates,
    register_translation,
)
from udfx.frontend.core.sequences import join_words, word_item
from udfx.frontend.core.splitting import join_filtered_words, split_item
from udfx.frontend.core.text import folded_case_chain
from udfx.semantics import text
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
from udfx.types import INT64_MAX, STR, Kind

POINT = "point"


def native_recase(method: str, samples: Sequence[str]) -> list[str]:
    expr = case_expression(CASE, col(POINT), method)
    (recased,) = DataFusionBackend().evaluate(
        [expr], {POINT: STR}, [(sample,) for sample in samples]
    )
    return cast("list[str]", recased)


@cache
def case_drift(method: str) -> CaseDrift:
    engine = f"datafusion {datafusion.__version__}"
    return text.case_drift(method, partial(native_recase, method), engine)


def recase(value: Expr, method: str) -> RuleResult[Expr]:
    expr = case_expression(CASE, value, method)
    sigma = functions.strpos(value, lit(SIGMA)) > lit(0)
    exact = case_guard(
        case_drift(method), lambda pattern: functions.regexp_like(value, lit(pattern)), sigma
    )
    return expr if exact is None else Lowered(expr, exact)


@RULES(".strip", Kind.STR)
@RULES(".strip", Kind.STR, Kind.STR)
def strip(text: Expr, characters: Expr | None = None) -> Expr:
    characters = (
        functions.coalesce(characters, lit(whitespace()))
        if characters is not None
        else lit(whitespace())
    )
    return functions.btrim(text, characters)


@RULES(".lstrip", Kind.STR)
@RULES(".lstrip", Kind.STR, Kind.STR)
def lstrip(text: Expr, characters: Expr | None = None) -> Expr:
    characters = (
        functions.coalesce(characters, lit(whitespace()))
        if characters is not None
        else lit(whitespace())
    )
    return functions.ltrim(text, characters)


@RULES(".rstrip", Kind.STR)
@RULES(".rstrip", Kind.STR, Kind.STR)
def rstrip(text: Expr, characters: Expr | None = None) -> Expr:
    characters = (
        functions.coalesce(characters, lit(whitespace()))
        if characters is not None
        else lit(whitespace())
    )
    return functions.rtrim(text, characters)


def find(text: Expr, pattern: Expr) -> Expr:
    return functions.strpos(text, pattern).cast(pa.int64()) - lit(1)


RULES(".find", Kind.STR, Kind.STR)(find)


SEARCH = SearchPrimitives(lit, when, length, functions.reverse, find)


@RULES(".rfind", Kind.STR, Kind.STR)
def rfind(text: Expr, pattern: Expr) -> Expr:
    return reverse_find(SEARCH, text, pattern)


@RULES(".count", Kind.STR, Kind.STR)
def count(text: Expr, pattern: Expr) -> Expr:
    removed = length(text) - length(functions.replace(text, pattern, lit("")))
    occurrences = removed / functions.greatest(length(pattern), lit(1))
    return count_occurrences(SEARCH, text, pattern, occurrences)


@RULES(".startswith", Kind.STR, Kind.STR)
def startswith(text: Expr, prefix: Expr) -> Expr:
    return functions.starts_with(text, prefix)


@RULES(".endswith", Kind.STR, Kind.STR)
def endswith(text: Expr, suffix: Expr) -> Expr:
    return functions.ends_with(text, suffix)


def characters(text: Expr) -> Expr:
    return functions.string_to_array(text, typed_literal(None, STR.or_none()))


@RULES(".replace", Kind.STR, Kind.STR, Kind.STR, facts={"empty": pattern_is_empty})
def replace(text: Expr, old: Expr, new: Expr, *, empty: bool | None = None) -> Expr:
    if empty is False:
        return functions.replace(text, old, new)
    padded = functions.concat(new, functions.array_to_string(characters(text), new), new)
    if empty is True:
        return when(text == lit(""), new, padded)
    return (
        functions.when((old == lit("")) & (text == lit("")), new)
        .when(old == lit(""), padded)
        .otherwise(functions.replace(text, old, new))
    )


def join(separator: Expr, words: Expr) -> Expr:
    return functions.array_to_string(words, separator)


RULES(".join", Kind.STR, Kind.LIST)(join)


@RULES(".join", Kind.STR, Kind.STR)
def join_characters(separator: Expr, text: Expr) -> Expr:
    return join(separator, characters(text))


RULES("coalesce", Kind.STR, Kind.STR)(functions.coalesce)


@RULES("__add__", Kind.STR, Kind.STR, linear=True)
def str_add(left: Expr, right: Expr) -> Expr:
    return functions.concat(left, right)


RULES("__eq__", Kind.STR, Kind.STR)(null_safe_equal)


@RULES("__ne__", Kind.STR, Kind.STR)
def str_ne(left: Expr, right: Expr) -> Expr:
    return ~null_safe_equal(left, right)


@RULES("__contains__", Kind.STR, Kind.STR)
def str_contains(text: Expr, item: Expr) -> Expr:
    return functions.strpos(text, item) > lit(0)


def list_contains(container: Expr, item: Expr) -> Expr:
    return functions.array_position(container, item).is_not_null()


RULES("len", Kind.STR, linear=True)(length)


def list_len(container: Expr) -> Expr:
    return functions.cardinality(container).cast(pa.int64())


RULES("len", Kind.LIST)(list_len)


@RULES("__getitem__", Kind.STR, Kind.INT)
def str_getitem(text: Expr, index: Expr) -> Expr:
    total = length(text)
    position = functions.when(index < lit(0), index + total).otherwise(index)
    clamped = functions.greatest(functions.least(position, total), lit(-1))
    return functions.substr(text, clamped + lit(1), lit(1))


@RULES("__getitem__", Kind.LIST, Kind.INT)
def list_getitem(container: Expr, index: Expr) -> Expr:
    position = functions.when(index < lit(0), index).otherwise(
        functions.nullif(index, lit(INT64_MAX)) + lit(1)
    )
    return functions.array_element(container, position.try_cast(pa.int32()))


register_string_predicates(RULES, lambda value, pattern: functions.regexp_like(value, lit(pattern)))
RULES.rewrite(".join", Kind.STR, Kind.LIST)(
    lambda node: join_filtered_words(node) or join_words(node)
)
RULES.rewrite("__getitem__", Kind.LIST, Kind.INT)(lambda node: split_item(node) or word_item(node))
RULES.rewrite(".casefold", Kind.STR)(folded_case_chain)


for method in CASE_METHODS:
    RULES(f".{method}", Kind.STR)(partial(recase, method=method))


CASE = CasePrimitives(
    lower=functions.lower,
    upper=functions.upper,
    title=functions.initcap,
    head=lambda value: functions.substr(value, lit(1), lit(1)),
    tail=lambda value: functions.substr(value, lit(2)),
    concat=functions.concat,
    when=when,
    matches=lambda value, pattern: functions.regexp_like(value, lit(pattern)),
    characters=lambda value, transform: functions.array_to_string(
        array_transform(characters(value), transform, pa.string()), lit("")
    ),
)


register_translation(RULES, TRANSLATION)
