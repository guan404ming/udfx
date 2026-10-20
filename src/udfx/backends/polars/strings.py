from __future__ import annotations

import operator
from collections.abc import Sequence
from functools import cache, reduce

import polars as pl

from udfx.backends.base import Lowered
from udfx.backends.polars.rules import RULES, broadcast, literal_value, when
from udfx.semantics import text
from udfx.semantics.text import CLUSTER_JOINERS, SIGMA, CaseDrift, has_border, whitespace, word
from udfx.types import Kind


@cache
def case_drift(method: str) -> CaseDrift:
    def native(samples: Sequence[str]) -> list[str]:
        return getattr(pl.Series(samples).str, f"to_{method}case")().to_list()

    return text.case_drift(method, native, f"polars {pl.__version__}")


def recase(value: pl.Expr, method: str) -> pl.Expr | Lowered[pl.Expr]:
    expr = getattr(value.str, f"to_{method}case")()
    found = case_drift(method)
    guards = [~value.str.contains(found.alone)] if found.alone else []
    if found.nearby:
        guards.append(~(value.str.contains(SIGMA, literal=True) & value.str.contains(found.nearby)))
    return Lowered(expr, reduce(operator.and_, guards)) if guards else expr


@RULES(".lower", Kind.STR)
def lower(value: pl.Expr) -> pl.Expr | Lowered[pl.Expr]:
    return recase(value, "lower")


@RULES(".upper", Kind.STR)
def upper(value: pl.Expr) -> pl.Expr | Lowered[pl.Expr]:
    return recase(value, "upper")


@RULES(".strip", Kind.STR)
def strip(text: pl.Expr) -> pl.Expr:
    return text.str.strip_chars(whitespace())


@RULES(".lstrip", Kind.STR)
def lstrip(text: pl.Expr) -> pl.Expr:
    return text.str.strip_chars_start(whitespace())


@RULES(".rstrip", Kind.STR)
def rstrip(text: pl.Expr) -> pl.Expr:
    return text.str.strip_chars_end(whitespace())


@RULES(".split", Kind.STR)
def split_words(text: pl.Expr) -> pl.Expr:
    return text.str.extract_all(word())


@RULES(".split", Kind.STR, Kind.STR)
def split(text: pl.Expr, separator: pl.Expr) -> pl.Expr:
    return text.str.split(separator)


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


@RULES(".rfind", Kind.STR, Kind.STR)
def rfind(text: pl.Expr, pattern: pl.Expr) -> pl.Expr | Lowered[pl.Expr]:
    known = literal_value(pattern)
    if isinstance(known, str) and known and not has_border(known):
        suffix = length(text.str.split(pattern).list.last())
        found = length(text) - length(pattern) - suffix
        return when(text.str.contains(pattern, literal=True), found, pl.lit(-1))
    position = find(text.str.reverse(), pattern.str.reverse())
    found = when(position < 0, pl.lit(-1), length(text) - length(pattern) - position)
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


@RULES(".replace", Kind.STR, Kind.STR, Kind.STR)
def replace(text: pl.Expr, old: pl.Expr, new: pl.Expr) -> pl.Expr:
    text = broadcast(text)
    padded = pl.concat_list([pl.lit(""), text.str.split(""), pl.lit("")])
    return (
        pl.when(old == "").then(padded.list.join(new)).otherwise(text.str.split(old).list.join(new))
    )


@RULES(".join", Kind.STR, Kind.LIST)
def join(separator: pl.Expr, words: pl.Expr) -> pl.Expr:
    return broadcast(words).list.join(separator)


@RULES("__add__", Kind.STR, Kind.STR)
def str_add(left: pl.Expr, right: pl.Expr) -> pl.Expr:
    return left + right


@RULES("__eq__", Kind.STR, Kind.STR)
def str_eq(left: pl.Expr, right: pl.Expr) -> pl.Expr:
    return left.eq_missing(right)


@RULES("__ne__", Kind.STR, Kind.STR)
def str_ne(left: pl.Expr, right: pl.Expr) -> pl.Expr:
    return left.ne_missing(right)


@RULES("__contains__", Kind.STR, Kind.STR)
def str_contains(text: pl.Expr, item: pl.Expr) -> pl.Expr:
    return text.str.contains(item, literal=True)


@RULES("__contains__", Kind.LIST, Kind.STR)
def list_contains(container: pl.Expr, item: pl.Expr) -> pl.Expr:
    return container.list.contains(item)


@RULES("len", Kind.STR)
def str_len(text: pl.Expr) -> pl.Expr:
    return length(text)


@RULES("len", Kind.LIST)
def list_len(container: pl.Expr) -> pl.Expr:
    return container.list.len().cast(pl.Int64)


@RULES("__getitem__", Kind.STR, Kind.INT)
def str_getitem(text: pl.Expr, index: pl.Expr) -> pl.Expr:
    return text.str.slice(index, 1)


@RULES("__getitem__", Kind.LIST, Kind.INT)
def list_getitem(container: pl.Expr, index: pl.Expr) -> pl.Expr:
    return container.list.get(index.clip(-(2**62), None), null_on_oob=True)


@RULES("slice", Kind.STR, Kind.INT, Kind.INT)
def str_slice(text: pl.Expr, start: pl.Expr, stop: pl.Expr) -> pl.Expr:
    size = length(text)

    def clamp(bound: pl.Expr, default: pl.Expr) -> pl.Expr:
        bound = bound.fill_null(default)
        return pl.when(bound < 0).then(bound + size).otherwise(bound).clip(0, size)

    begin = clamp(start, pl.lit(0))
    end = clamp(stop, size)
    return text.str.slice(begin, (end - begin).clip(lower_bound=0))
