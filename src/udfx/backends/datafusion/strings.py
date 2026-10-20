from __future__ import annotations

from collections.abc import Sequence
from functools import cache
from typing import cast

import datafusion
import pyarrow as pa
from datafusion import Expr, col, functions, lit

from udfx.backends.base import Lowered, RuleResult
from udfx.backends.datafusion.backend import DataFusionBackend
from udfx.backends.datafusion.rules import RULES, length, null_safe_equal, typed, when
from udfx.semantics import text
from udfx.semantics.text import (
    SIGMA,
    CaseDrift,
    SlicePrimitives,
    case_guard,
    char_class,
    slice_bounds,
    whitespace,
)
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


def recase(value: Expr, method: str) -> RuleResult[Expr]:
    expr = getattr(functions, method)(value)
    sigma = functions.strpos(value, lit(SIGMA)) > lit(0)
    exact = case_guard(
        case_drift(method), lambda pattern: functions.regexp_like(value, lit(pattern)), sigma
    )
    return expr if exact is None else Lowered(expr, exact)


@RULES(".lower", Kind.STR)
def lower(value: Expr) -> RuleResult[Expr]:
    return recase(value, "lower")


@RULES(".upper", Kind.STR)
def upper(value: Expr) -> RuleResult[Expr]:
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


def split_words(text: Expr) -> Expr:
    spaced = functions.regexp_replace(text, lit(char_class(whitespace()) + "+"), lit(" "), lit("g"))
    return functions.string_to_array(functions.btrim(spaced, lit(" ")), lit(" "))


RULES(".split", Kind.STR)(split_words)


@RULES(".split", Kind.STR, Kind.STR)
def split(text: Expr, separator: Expr) -> Expr:
    return (
        functions.when(separator.is_null(), split_words(text))
        .when(text == lit(""), functions.make_array(lit("")))
        .otherwise(functions.string_to_array(text, separator))
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
    return null_safe_equal(left, right)


@RULES("__ne__", Kind.STR, Kind.STR)
def str_ne(left: Expr, right: Expr) -> Expr:
    return ~null_safe_equal(left, right)


@RULES("__contains__", Kind.STR, Kind.STR)
def str_contains(text: Expr, item: Expr) -> Expr:
    return functions.strpos(text, item) > lit(0)


@RULES("__contains__", Kind.LIST, Kind.STR)
def list_contains(container: Expr, item: Expr) -> Expr:
    return functions.array_position(container, item).is_not_null()


@RULES("len", Kind.STR)
def str_len(text: Expr) -> Expr:
    return length(text)


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
    total = list_len(container)
    position = functions.when(index < lit(0), functions.greatest(index, lit(-1) - total)).otherwise(
        functions.least(index, total) + lit(1)
    )
    return functions.array_element(container, position)


SLICE = SlicePrimitives(
    literal=lit,
    when=when,
    coalesce=functions.coalesce,
    clip=lambda value, lower, upper: functions.greatest(functions.least(value, upper), lower),
)


@RULES("slice", Kind.STR, Kind.INT, Kind.INT)
def str_slice(text: Expr, start: Expr, stop: Expr) -> Expr:
    bounds = slice_bounds(SLICE, length(text), start, stop)
    return functions.substr(text, bounds.start + lit(1), bounds.length)
