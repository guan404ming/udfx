from __future__ import annotations

import polars as pl

from udfx.backends.polars.rules import RULES
from udfx.ops.strings import WHITESPACE
from udfx.types import Kind

WORDS = "[^" + "".join(f"\\x{{{ord(char):x}}}" for char in WHITESPACE) + "]+"


@RULES(".lower", Kind.STR)
def lower(a: pl.Expr) -> pl.Expr:
    return a.str.to_lowercase()


@RULES(".upper", Kind.STR)
def upper(a: pl.Expr) -> pl.Expr:
    return a.str.to_uppercase()


@RULES(".strip", Kind.STR)
def strip(a: pl.Expr) -> pl.Expr:
    return a.str.strip_chars(WHITESPACE)


@RULES(".lstrip", Kind.STR)
def lstrip(a: pl.Expr) -> pl.Expr:
    return a.str.strip_chars_start(WHITESPACE)


@RULES(".rstrip", Kind.STR)
def rstrip(a: pl.Expr) -> pl.Expr:
    return a.str.strip_chars_end(WHITESPACE)


@RULES(".split", Kind.STR)
def split_words(a: pl.Expr) -> pl.Expr:
    return a.str.extract_all(WORDS)


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
def rfind(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    reversed_find = find(a.str.reverse(), b.str.reverse())
    return pl.when(reversed_find < 0).then(-1).otherwise(length(a) - length(b) - reversed_find)


@RULES(".count", Kind.STR, Kind.STR)
def count(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    return a.str.count_matches(b, literal=True).cast(pl.Int64)


@RULES(".startswith", Kind.STR, Kind.STR)
def startswith(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    return a.str.starts_with(b)


@RULES(".endswith", Kind.STR, Kind.STR)
def endswith(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    return a.str.ends_with(b)


@RULES(".replace", Kind.STR, Kind.STR, Kind.STR)
def replace(a: pl.Expr, b: pl.Expr, c: pl.Expr) -> pl.Expr:
    padded = pl.concat_list([pl.lit(""), a.str.split(""), pl.lit("")])
    return pl.when(b == "").then(padded.list.join(c)).otherwise(a.str.split(b).list.join(c))


@RULES(".join", Kind.STR, Kind.LIST)
def join(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    return b.list.join(a)


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
