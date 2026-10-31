from __future__ import annotations

from collections.abc import Sequence
from functools import cache, partial

import polars as pl

from udfx.backends.polars.conversions import TEXT as TRANSLATION
from udfx.backends.polars.rules import RULES, broadcast, literal_value, when
from udfx.backends.registration import (
    Lowered,
    RuleResult,
    register_string_predicates,
    register_translation,
)
from udfx.frontend.core.sequences import join_words, word_item
from udfx.frontend.core.splitting import join_filtered_words
from udfx.frontend.core.text import folded_case_chain
from udfx.semantics import text
from udfx.semantics.clusters import CLUSTER_JOINERS
from udfx.semantics.facts import join_source_is_column, pattern_is_empty
from udfx.semantics.sequences import (
    SearchPrimitives,
    reverse_find,
)
from udfx.semantics.text import (
    CASE_METHODS,
    SIGMA,
    CaseDrift,
    CasePrimitives,
    case_expression,
    case_guard,
    has_prefix_suffix_overlap,
    whitespace,
)
from udfx.types import Kind


def native_recase(method: str, samples: Sequence[str]) -> list[str]:
    return (
        pl.DataFrame({"udfx point": samples})
        .select(case_expression(CASE, pl.col("udfx point"), method))
        .to_series()
        .to_list()
    )


@cache
def case_drift(method: str) -> CaseDrift:
    return text.case_drift(method, partial(native_recase, method), f"polars {pl.__version__}")


def recase(value: pl.Expr, method: str) -> RuleResult[pl.Expr]:
    expr = case_expression(CASE, value, method)
    sigma = value.str.contains(SIGMA, literal=True)
    exact = case_guard(case_drift(method), value.str.contains, sigma)
    return expr if exact is None else Lowered(expr, exact)


@RULES(".strip", Kind.STR)
@RULES(".strip", Kind.STR, Kind.STR)
def strip(text: pl.Expr, characters: pl.Expr | None = None) -> pl.Expr:
    characters = (
        pl.coalesce(characters, pl.lit(whitespace()))
        if characters is not None
        else pl.lit(whitespace())
    )
    return text.str.strip_chars(characters)


@RULES(".lstrip", Kind.STR)
@RULES(".lstrip", Kind.STR, Kind.STR)
def lstrip(text: pl.Expr, characters: pl.Expr | None = None) -> pl.Expr:
    characters = (
        pl.coalesce(characters, pl.lit(whitespace()))
        if characters is not None
        else pl.lit(whitespace())
    )
    return text.str.strip_chars_start(characters)


@RULES(".rstrip", Kind.STR)
@RULES(".rstrip", Kind.STR, Kind.STR)
def rstrip(text: pl.Expr, characters: pl.Expr | None = None) -> pl.Expr:
    characters = (
        pl.coalesce(characters, pl.lit(whitespace()))
        if characters is not None
        else pl.lit(whitespace())
    )
    return text.str.strip_chars_end(characters)


def length(text: pl.Expr) -> pl.Expr:
    return text.str.len_chars().cast(pl.Int64)


def find(text: pl.Expr, pattern: pl.Expr) -> pl.Expr:
    prefix = length(text.str.split(pattern).list.first())
    return (
        pl.when(pattern == "")
        .then(0)
        .when(text.str.contains(pattern, literal=True))
        .then(prefix)
        .otherwise(-1)
    )


RULES(".find", Kind.STR, Kind.STR)(find)


def single_code_point_clusters(text: pl.Expr) -> pl.Expr:
    return ~text.str.contains(CLUSTER_JOINERS)


SEARCH = SearchPrimitives(pl.lit, when, length, lambda text: text.str.reverse(), find)


@RULES(".rfind", Kind.STR, Kind.STR)
def rfind(text: pl.Expr, pattern: pl.Expr) -> RuleResult[pl.Expr]:
    known = literal_value(pattern)
    if isinstance(known, str) and known and not has_prefix_suffix_overlap(known):
        suffix = length(text.str.split(pattern).list.last())
        found = length(text) - length(pattern) - suffix
        return when(text.str.contains(pattern, literal=True), found, pl.lit(-1))
    found = reverse_find(SEARCH, text, pattern)
    return Lowered(found, single_code_point_clusters(text) & single_code_point_clusters(pattern))


@RULES(".count", Kind.STR, Kind.STR)
def count(text: pl.Expr, pattern: pl.Expr) -> pl.Expr:
    return broadcast(text).str.count_matches(pattern, literal=True).cast(pl.Int64)


@RULES(".startswith", Kind.STR, Kind.STR)
def startswith(text: pl.Expr, prefix: pl.Expr) -> pl.Expr:
    return text.str.starts_with(prefix)


@RULES(".endswith", Kind.STR, Kind.STR)
def endswith(text: pl.Expr, suffix: pl.Expr) -> pl.Expr:
    return text.str.ends_with(suffix)


@RULES(".replace", Kind.STR, Kind.STR, Kind.STR, facts={"empty": pattern_is_empty})
def replace(text: pl.Expr, old: pl.Expr, new: pl.Expr, *, empty: bool | None = None) -> pl.Expr:
    if empty is not None:
        return broadcast(text).str.replace_all(old, new.fill_null(""), literal=True)
    text = broadcast(text)
    padded = pl.concat_list([pl.lit(""), text.str.split(""), pl.lit("")])
    return (
        pl.when(old == "").then(padded.list.join(new)).otherwise(text.str.split(old).list.join(new))
    )


def join(separator: pl.Expr, words: pl.Expr, *, column: bool = False) -> pl.Expr:
    words = broadcast(words)
    if not column:
        words = words.list.concat(pl.lit([], dtype=pl.List(pl.String)))
    return words.list.join(separator)


RULES(".join", Kind.STR, Kind.LIST, facts={"column": join_source_is_column})(join)


@RULES(".join", Kind.STR, Kind.STR)
def join_characters(separator: pl.Expr, text: pl.Expr) -> pl.Expr:
    return join(separator, text.str.split(""), column=True)


@RULES("__add__", Kind.STR, Kind.STR, linear=True)
def str_add(left: pl.Expr, right: pl.Expr) -> pl.Expr:
    return left + right


RULES("coalesce", Kind.STR, Kind.STR)(pl.Expr.fill_null)


@RULES("__eq__", Kind.STR, Kind.STR)
def str_eq(left: pl.Expr, right: pl.Expr) -> pl.Expr:
    return left.eq_missing(right)


@RULES("__ne__", Kind.STR, Kind.STR)
def str_ne(left: pl.Expr, right: pl.Expr) -> pl.Expr:
    return left.ne_missing(right)


@RULES("__contains__", Kind.STR, Kind.STR)
def str_contains(text: pl.Expr, item: pl.Expr) -> pl.Expr:
    return text.str.contains(item, literal=True)


def list_contains(container: pl.Expr, item: pl.Expr) -> pl.Expr:
    return container.list.contains(item, nulls_equal=True)


RULES("__contains__", Kind.LIST, Kind.STR)(list_contains)


@RULES("len", Kind.STR, linear=True)
def str_len(text: pl.Expr) -> pl.Expr:
    return length(text)


@RULES("len", Kind.LIST)
def list_len(container: pl.Expr) -> pl.Expr:
    return container.list.len().cast(pl.Int64)


@RULES("__getitem__", Kind.STR, Kind.INT)
def str_getitem(text: pl.Expr, index: pl.Expr) -> pl.Expr:
    return text.str.slice(index, 1)


@RULES("__getitem__", Kind.LIST, Kind.INT)
@RULES("optional_item", Kind.LIST, Kind.INT)
def list_getitem(container: pl.Expr, index: pl.Expr) -> pl.Expr:
    return container.list.get(index.clip(-(2**62), None), null_on_oob=True)


register_string_predicates(RULES, lambda value, pattern: value.str.contains(pattern))
RULES.rewrite(".join", Kind.STR, Kind.LIST)(
    lambda node: join_filtered_words(node) or join_words(node)
)
RULES.rewrite("__getitem__", Kind.LIST, Kind.INT)(word_item)
RULES.rewrite(".casefold", Kind.STR)(folded_case_chain)


for method in CASE_METHODS:
    RULES(f".{method}", Kind.STR)(partial(recase, method=method))


CASE = CasePrimitives(
    lower=lambda value: value.str.to_lowercase(),
    upper=lambda value: value.str.to_uppercase(),
    title=lambda value: value.str.to_titlecase(),
    head=lambda value: value.str.slice(0, 1),
    tail=lambda value: value.str.slice(1),
    concat=lambda *values: pl.concat_str(values),
    when=when,
    matches=lambda value, pattern: value.str.contains(pattern),
    characters=lambda value, transform: (
        value.str.split("").list.eval(transform(pl.element())).list.join("")
    ),
)


register_translation(RULES, TRANSLATION)
