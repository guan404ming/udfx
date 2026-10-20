from __future__ import annotations

from collections.abc import Sequence
from functools import cache
from typing import cast

import pyarrow as pa
from datafusion import Expr, col, functions, lit

from udfx.backends.base import Lowered
from udfx.backends.datafusion.backend import DataFusionBackend
from udfx.backends.datafusion.rules import RULES, length, not_distinct, typed
from udfx.semantics.text import char_class, drift, whitespace
from udfx.types import STR, Kind

POINT = "udfx point"


@cache
def case_drift(method: str) -> str | None:
    def native(points: Sequence[str]) -> list[str]:
        expr = getattr(functions, method)(col(POINT))
        (recased,) = DataFusionBackend().evaluate(
            [expr], {POINT: STR}, [(point,) for point in points]
        )
        return cast("list[str]", recased)

    drifted = drift(getattr(str, method), native)
    return char_class(drifted) if drifted else None


def recase(value: Expr, method: str) -> Expr | Lowered[Expr]:
    expr = getattr(functions, method)(value)
    drifted = case_drift(method)
    return expr if drifted is None else Lowered(expr, ~functions.regexp_like(value, lit(drifted)))


@RULES(".lower", Kind.STR)
def lower(value: Expr) -> Expr | Lowered[Expr]:
    return recase(value, "lower")


@RULES(".upper", Kind.STR)
def upper(value: Expr) -> Expr | Lowered[Expr]:
    return recase(value, "upper")


@RULES(".strip", Kind.STR)
def strip(a: Expr) -> Expr:
    return functions.btrim(a, lit(whitespace()))


@RULES(".lstrip", Kind.STR)
def lstrip(a: Expr) -> Expr:
    return functions.ltrim(a, lit(whitespace()))


@RULES(".rstrip", Kind.STR)
def rstrip(a: Expr) -> Expr:
    return functions.rtrim(a, lit(whitespace()))


@RULES(".split", Kind.STR)
def split_words(a: Expr) -> Expr:
    spaced = functions.regexp_replace(a, lit(char_class(whitespace()) + "+"), lit(" "), lit("g"))
    return functions.string_to_array(functions.btrim(spaced, lit(" ")), lit(" "))


@RULES(".split", Kind.STR, Kind.STR)
def split(a: Expr, b: Expr) -> Expr:
    return functions.when(a == lit(""), functions.make_array(lit(""))).otherwise(
        functions.string_to_array(a, b)
    )


def find(a: Expr, b: Expr) -> Expr:
    return functions.strpos(a, b).cast(pa.int64()) - lit(1)


RULES(".find", Kind.STR, Kind.STR)(find)


@RULES(".rfind", Kind.STR, Kind.STR)
def rfind(text: Expr, pattern: Expr) -> Expr:
    position = find(functions.reverse(text), functions.reverse(pattern))
    return functions.when(position < lit(0), lit(-1)).otherwise(
        length(text) - length(pattern) - position
    )


@RULES(".count", Kind.STR, Kind.STR)
def count(text: Expr, pattern: Expr) -> Expr:
    removed = length(text) - length(functions.replace(text, pattern, lit("")))
    return functions.when(pattern == lit(""), length(text) + lit(1)).otherwise(
        removed / functions.greatest(length(pattern), lit(1))
    )


@RULES(".startswith", Kind.STR, Kind.STR)
def startswith(a: Expr, b: Expr) -> Expr:
    return functions.starts_with(a, b)


@RULES(".endswith", Kind.STR, Kind.STR)
def endswith(a: Expr, b: Expr) -> Expr:
    return functions.ends_with(a, b)


@RULES(".replace", Kind.STR, Kind.STR, Kind.STR)
def replace(text: Expr, old: Expr, new: Expr) -> Expr:
    characters = functions.string_to_array(text, typed(None, STR.or_none()))
    padded = functions.concat(new, functions.array_to_string(characters, new), new)
    return (
        functions.when((old == lit("")) & (text == lit("")), new)
        .when(old == lit(""), padded)
        .otherwise(functions.replace(text, old, new))
    )


@RULES(".join", Kind.STR, Kind.LIST)
def join(separator: Expr, words: Expr) -> Expr:
    return functions.array_to_string(words, separator)


@RULES("__add__", Kind.STR, Kind.STR)
def str_add(a: Expr, b: Expr) -> Expr:
    return functions.concat(a, b)


@RULES("__eq__", Kind.STR, Kind.STR)
def str_eq(a: Expr, b: Expr) -> Expr:
    return not_distinct(a, b)


@RULES("__ne__", Kind.STR, Kind.STR)
def str_ne(a: Expr, b: Expr) -> Expr:
    return ~not_distinct(a, b)


@RULES("__contains__", Kind.STR, Kind.STR)
def str_contains(a: Expr, b: Expr) -> Expr:
    return functions.strpos(a, b) > lit(0)


@RULES("__contains__", Kind.LIST, Kind.STR)
def list_contains(a: Expr, b: Expr) -> Expr:
    return functions.array_has(a, b)


@RULES("len", Kind.STR)
def str_len(a: Expr) -> Expr:
    return length(a)


def size(a: Expr) -> Expr:
    return functions.cardinality(a).cast(pa.int64())


RULES("len", Kind.LIST)(size)


@RULES("__getitem__", Kind.STR, Kind.INT)
def str_getitem(a: Expr, b: Expr) -> Expr:
    total = length(a)
    position = functions.when(b < lit(0), b + total).otherwise(b)
    clamped = functions.greatest(functions.least(position, total), lit(-1))
    return functions.substr(a, clamped + lit(1), lit(1))


@RULES("__getitem__", Kind.LIST, Kind.INT)
def list_getitem(container: Expr, index: Expr) -> Expr:
    total = size(container)
    position = functions.when(index < lit(0), functions.greatest(index, lit(-1) - total)).otherwise(
        functions.least(index, total) + lit(1)
    )
    return functions.array_element(container, position)


@RULES("slice", Kind.STR, Kind.INT, Kind.INT)
def str_slice(a: Expr, b: Expr, c: Expr) -> Expr:
    total = length(a)

    def clamp(bound: Expr, default: Expr) -> Expr:
        bound = functions.coalesce(bound, default)
        shifted = functions.when(bound < lit(0), bound + total).otherwise(bound)
        return functions.greatest(functions.least(shifted, total), lit(0))

    start = clamp(b, lit(0))
    stop = clamp(c, total)
    return functions.substr(a, start + lit(1), functions.greatest(stop - start, lit(0)))
