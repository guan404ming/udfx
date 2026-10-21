from __future__ import annotations

import polars as pl

from udfx.backends.polars.numbers import cast_float64
from udfx.backends.polars.rules import RULES, when
from udfx.backends.registration import Lowered, register_text_conversions
from udfx.semantics.literals import (
    SpellingPrimitives,
    TextPrimitives,
    bool_text,
    show_none,
)
from udfx.semantics.numbers import (
    REPR_FIXED_MAX,
    REPR_FIXED_MIN,
)
from udfx.types import Kind

TEXT = TextPrimitives(
    coalesce=pl.coalesce,
    translate=lambda text, old, new: text.str.replace_many(
        list(old), list(new) + [""] * (len(old) - len(new))
    ),
    matches=lambda text, pattern: text.str.contains(pattern),
    strip=lambda text, chars: text.str.strip_chars(chars),
    replace=lambda text, old, new: text.str.replace_all(old, new, literal=True),
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


RULES("float", Kind.INT)(cast_float64)


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

register_text_conversions(RULES, TEXT)
