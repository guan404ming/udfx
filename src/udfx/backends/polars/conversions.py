from __future__ import annotations

import polars as pl

from udfx.backends.base import Lowered
from udfx.backends.polars.numbers import cast_float64
from udfx.backends.polars.rules import RULES, when
from udfx.semantics.literals import (
    SpellingPrimitives,
    TextPrimitives,
    bool_text,
    float_parses,
    int_parses,
    parse_float,
    parse_int,
    show_none,
)
from udfx.semantics.numbers import (
    REPR_FIXED_MAX,
    REPR_FIXED_MIN,
)
from udfx.types import Kind

TEXT = TextPrimitives(
    matches=lambda text, pattern: text.str.contains(pattern),
    strip=lambda text, chars: text.str.strip_chars(chars),
    replace=lambda text, old, new: text.str.replace_all(old, new, literal=True),
    byte_length=lambda text: text.str.len_bytes(),
    char_length=lambda text: text.str.len_chars(),
    not_null=pl.Expr.is_not_null,
    try_int=lambda text: text.str.to_integer(strict=False),
    try_float=lambda text: text.cast(pl.Float64, strict=False),
)

SPELLING = SpellingPrimitives(pl.lit, when, pl.Expr.is_null, pl.Expr.fill_null)


@RULES("int", Kind.FLOAT)
def float_int(value: pl.Expr) -> Lowered[pl.Expr]:
    whole = value.cast(pl.Int64, strict=False)
    return Lowered(whole, whole.is_not_null() | ~value.is_finite())


@RULES("float", Kind.INT)
def int_float(value: pl.Expr) -> pl.Expr:
    return cast_float64(value)


@RULES("float", Kind.FLOAT)
def float_float(value: pl.Expr) -> pl.Expr:
    return value


@RULES("str", Kind.INT)
def int_str(value: pl.Expr) -> pl.Expr:
    return show_none(SPELLING, value.cast(pl.String))


@RULES("str", Kind.FLOAT)
def float_str(value: pl.Expr) -> Lowered[pl.Expr]:
    fixed = value.abs().is_between(REPR_FIXED_MIN, REPR_FIXED_MAX, closed="left")
    return Lowered(
        show_none(SPELLING, value.cast(pl.String)), value.is_null() | (value == 0.0) | fixed
    )


@RULES("str", Kind.BOOL)
def bool_str(value: pl.Expr) -> pl.Expr:
    return bool_text(SPELLING, value)


RULES("str", Kind.STR)(lambda text: show_none(SPELLING, text))


@RULES("int_parses", Kind.STR)
def str_int_parses(text: pl.Expr) -> Lowered[pl.Expr]:
    return Lowered(*int_parses(TEXT, text))


@RULES("int", Kind.STR)
def str_int(text: pl.Expr) -> Lowered[pl.Expr]:
    return Lowered(*parse_int(TEXT, text))


@RULES("float_parses", Kind.STR)
def str_float_parses(text: pl.Expr) -> Lowered[pl.Expr]:
    return Lowered(*float_parses(TEXT, text))


@RULES("float", Kind.STR)
def str_float(text: pl.Expr) -> Lowered[pl.Expr]:
    return Lowered(*parse_float(TEXT, text))
