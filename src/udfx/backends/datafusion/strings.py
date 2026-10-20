from __future__ import annotations

import operator
from collections.abc import Sequence
from functools import cache, reduce
from typing import cast

import datafusion
import pyarrow as pa
from datafusion import Expr, col, functions, lit

from udfx.backends.base import Lowered
from udfx.backends.datafusion.backend import DataFusionBackend
from udfx.backends.datafusion.rules import RULES, length, not_distinct, typed
from udfx.semantics import text
from udfx.semantics.text import SIGMA, CaseDrift, char_class, whitespace
from udfx.types import STR, Kind

POINT = "udfx point"


@cache
def case_drift(method: str) -> CaseDrift:
    def native(samples: Sequence[str]) -> list[str]:
        expr = getattr(functions, method)(col(POINT))
        (recased,) = DataFusionBackend().evaluate(
            [expr], {POINT: STR}, [(sample,) for sample in samples]
        )
        return cast("list[str]", recased)

    return text.case_drift(method, native, f"datafusion {datafusion.__version__}")


def recase(value: Expr, method: str) -> Expr | Lowered[Expr]:
    expr = getattr(functions, method)(value)
    found = case_drift(method)
    guards = [~functions.regexp_like(value, lit(found.alone))] if found.alone else []
    if found.nearby:
        sigma = functions.strpos(value, lit(SIGMA)) > lit(0)
        guards.append(~(sigma & functions.regexp_like(value, lit(found.nearby))))
    return Lowered(expr, reduce(operator.and_, guards)) if guards else expr


@RULES(".lower", Kind.STR)
def lower(value: Expr) -> Expr | Lowered[Expr]:
    return recase(value, "lower")


@RULES(".upper", Kind.STR)
def upper(value: Expr) -> Expr | Lowered[Expr]:
    return recase(value, "upper")


@RULES(".strip", Kind.STR)
def strip(text: Expr) -> Expr:
    return functions.btrim(text, lit(whitespace()))


@RULES(".lstrip", Kind.STR)
def lstrip(text: Expr) -> Expr:
    return functions.ltrim(text, lit(whitespace()))


@RULES(".rstrip", Kind.STR)
def rstrip(text: Expr) -> Expr:
    return functions.rtrim(text, lit(whitespace()))


@RULES(".split", Kind.STR)
def split_words(text: Expr) -> Expr:
    spaced = functions.regexp_replace(text, lit(char_class(whitespace()) + "+"), lit(" "), lit("g"))
    return functions.string_to_array(functions.btrim(spaced, lit(" ")), lit(" "))


@RULES(".split", Kind.STR, Kind.STR)
def split(text: Expr, separator: Expr) -> Expr:
    return functions.when(text == lit(""), functions.make_array(lit(""))).otherwise(
        functions.string_to_array(text, separator)
    )


def find(text: Expr, pattern: Expr) -> Expr:
    return functions.strpos(text, pattern).cast(pa.int64()) - lit(1)


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
def startswith(text: Expr, prefix: Expr) -> Expr:
    return functions.starts_with(text, prefix)


@RULES(".endswith", Kind.STR, Kind.STR)
def endswith(text: Expr, suffix: Expr) -> Expr:
    return functions.ends_with(text, suffix)


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
def str_add(left: Expr, right: Expr) -> Expr:
    return functions.concat(left, right)


@RULES("__eq__", Kind.STR, Kind.STR)
def str_eq(left: Expr, right: Expr) -> Expr:
    return not_distinct(left, right)


@RULES("__ne__", Kind.STR, Kind.STR)
def str_ne(left: Expr, right: Expr) -> Expr:
    return ~not_distinct(left, right)


@RULES("__contains__", Kind.STR, Kind.STR)
def str_contains(text: Expr, item: Expr) -> Expr:
    return functions.strpos(text, item) > lit(0)


@RULES("__contains__", Kind.LIST, Kind.STR)
def list_contains(container: Expr, item: Expr) -> Expr:
    return functions.array_has(container, item)


@RULES("len", Kind.STR)
def str_len(text: Expr) -> Expr:
    return length(text)


def size(container: Expr) -> Expr:
    return functions.cardinality(container).cast(pa.int64())


RULES("len", Kind.LIST)(size)


@RULES("__getitem__", Kind.STR, Kind.INT)
def str_getitem(text: Expr, index: Expr) -> Expr:
    total = length(text)
    position = functions.when(index < lit(0), index + total).otherwise(index)
    clamped = functions.greatest(functions.least(position, total), lit(-1))
    return functions.substr(text, clamped + lit(1), lit(1))


@RULES("__getitem__", Kind.LIST, Kind.INT)
def list_getitem(container: Expr, index: Expr) -> Expr:
    total = size(container)
    position = functions.when(index < lit(0), functions.greatest(index, lit(-1) - total)).otherwise(
        functions.least(index, total) + lit(1)
    )
    return functions.array_element(container, position)


@RULES("slice", Kind.STR, Kind.INT, Kind.INT)
def str_slice(text: Expr, start: Expr, stop: Expr) -> Expr:
    total = length(text)

    def clamp(bound: Expr, default: Expr) -> Expr:
        bound = functions.coalesce(bound, default)
        shifted = functions.when(bound < lit(0), bound + total).otherwise(bound)
        return functions.greatest(functions.least(shifted, total), lit(0))

    begin = clamp(start, lit(0))
    end = clamp(stop, total)
    return functions.substr(text, begin + lit(1), functions.greatest(end - begin, lit(0)))
