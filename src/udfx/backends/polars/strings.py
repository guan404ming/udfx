from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from udfx.backends.base import Lowered
from udfx.backends.polars.rules import RULES, broadcast
from udfx.semantics.text import char_class, drift, whitespace, word
from udfx.types import Kind

CASE_DRIFT: dict[str, str | None] = {}


def case_drift(method: str) -> str | None:
    if method not in CASE_DRIFT:

        def native(points: Sequence[str]) -> list[str]:
            return getattr(pl.Series(points).str, f"to_{method}case")().to_list()

        drifted = drift(getattr(str, method), native)
        CASE_DRIFT[method] = char_class(drifted) if drifted else None
    return CASE_DRIFT[method]


def recase(value: pl.Expr, method: str) -> pl.Expr | Lowered[pl.Expr]:
    expr = getattr(value.str, f"to_{method}case")()
    drift = case_drift(method)
    return expr if drift is None else Lowered(expr, ~value.str.contains(drift))


@RULES(".lower", Kind.STR)
def lower(value: pl.Expr) -> pl.Expr | Lowered[pl.Expr]:
    return recase(value, "lower")


@RULES(".upper", Kind.STR)
def upper(value: pl.Expr) -> pl.Expr | Lowered[pl.Expr]:
    return recase(value, "upper")


@RULES(".strip", Kind.STR)
def strip(a: pl.Expr) -> pl.Expr:
    return a.str.strip_chars(whitespace())


@RULES(".lstrip", Kind.STR)
def lstrip(a: pl.Expr) -> pl.Expr:
    return a.str.strip_chars_start(whitespace())


@RULES(".rstrip", Kind.STR)
def rstrip(a: pl.Expr) -> pl.Expr:
    return a.str.strip_chars_end(whitespace())


@RULES(".split", Kind.STR)
def split_words(a: pl.Expr) -> pl.Expr:
    return a.str.extract_all(word())


@RULES(".split", Kind.STR, Kind.STR)
def split(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    return a.str.split(b)


def length(a: pl.Expr) -> pl.Expr:
    return a.str.len_chars().cast(pl.Int64)


def find(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    prefix = length(a.str.split(b).list.first())
    return pl.when(b == "").then(0).when(a.str.contains(b, literal=True)).then(prefix).otherwise(-1)


RULES(".find", Kind.STR, Kind.STR)(find)


@RULES(".rfind", Kind.STR, Kind.STR)
def rfind(text: pl.Expr, pattern: pl.Expr) -> pl.Expr:
    suffix = length(text.str.split(pattern).list.last())
    found = text.str.contains(pattern, literal=True)
    position = length(text) - length(pattern) - suffix
    return pl.when(pattern == "").then(length(text)).when(found).then(position).otherwise(-1)


@RULES(".count", Kind.STR, Kind.STR)
def count(text: pl.Expr, pattern: pl.Expr) -> pl.Expr:
    return broadcast(text).str.count_matches(pattern, literal=True).cast(pl.Int64)


@RULES(".startswith", Kind.STR, Kind.STR)
def startswith(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    return a.str.starts_with(b)


@RULES(".endswith", Kind.STR, Kind.STR)
def endswith(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    return a.str.ends_with(b)


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
def str_add(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    return a + b


@RULES("__eq__", Kind.STR, Kind.STR)
def str_eq(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    return a.eq_missing(b)


@RULES("__ne__", Kind.STR, Kind.STR)
def str_ne(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    return a.ne_missing(b)


@RULES("__contains__", Kind.STR, Kind.STR)
def str_contains(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    return a.str.contains(b, literal=True)


@RULES("__contains__", Kind.LIST, Kind.STR)
def list_contains(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    return a.list.contains(b)


@RULES("len", Kind.STR)
def str_len(a: pl.Expr) -> pl.Expr:
    return length(a)


@RULES("len", Kind.LIST)
def list_len(a: pl.Expr) -> pl.Expr:
    return a.list.len().cast(pl.Int64)


@RULES("__getitem__", Kind.STR, Kind.INT)
def str_getitem(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    return a.str.slice(b, 1)


@RULES("__getitem__", Kind.LIST, Kind.INT)
def list_getitem(container: pl.Expr, index: pl.Expr) -> pl.Expr:
    return container.list.get(index.clip(-(2**62), None), null_on_oob=True)


@RULES("slice", Kind.STR, Kind.INT, Kind.INT)
def str_slice(a: pl.Expr, b: pl.Expr, c: pl.Expr) -> pl.Expr:
    size = length(a)

    def clamp(bound: pl.Expr, default: pl.Expr) -> pl.Expr:
        bound = bound.fill_null(default)
        return pl.when(bound < 0).then(bound + size).otherwise(bound).clip(0, size)

    start = clamp(b, pl.lit(0))
    stop = clamp(c, size)
    return a.str.slice(start, (stop - start).clip(lower_bound=0))
