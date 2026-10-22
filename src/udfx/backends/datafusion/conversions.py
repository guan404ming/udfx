from __future__ import annotations

import pyarrow as pa
from datafusion import Expr, functions, lit

from udfx.backends.datafusion.numbers import cast_float64, finite
from udfx.backends.datafusion.rules import RULES, length, when
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
    literal=lit,
    when=when,
    coalesce=functions.coalesce,
    translate=lambda text, old, new: functions.translate(text, lit(old), lit(new)),
    matches=lambda text, pattern: functions.regexp_like(text, lit(pattern)),
    strip=lambda text, chars: functions.btrim(text, lit(chars)),
    replace=lambda text, old, new: functions.replace(text, lit(old), lit(new)),
    char_length=length,
    not_null=Expr.is_not_null,
    try_int=lambda text: text.try_cast(pa.int64()),
    try_float=lambda text: text.try_cast(pa.float64()),
)

SPELLING = SpellingPrimitives(lit, when, Expr.is_null, functions.coalesce)


@RULES("int", Kind.FLOAT)
def float_int(value: Expr) -> Lowered[Expr]:
    whole = value.try_cast(pa.int64())
    return Lowered(whole, overflow=~(whole.is_not_null() | ~finite(value)))


RULES("float", Kind.INT)(cast_float64)


@RULES("float", Kind.FLOAT)
def float_float(value: Expr) -> Expr:
    return value


@RULES("str", Kind.INT)
def int_str(value: Expr) -> Expr:
    return show_none(SPELLING, value.cast(pa.string()))


@RULES("str", Kind.FLOAT)
def float_str(value: Expr) -> Lowered[Expr]:
    size = functions.abs(value)
    fixed = (size >= lit(REPR_FIXED_MIN)) & (size < lit(REPR_FIXED_MAX))
    exact = value.is_null() | (value == lit(0.0)) | fixed
    return Lowered(show_none(SPELLING, value.cast(pa.string())), exact)


@RULES("str", Kind.BOOL)
def bool_str(value: Expr) -> Expr:
    return bool_text(SPELLING, value)


RULES("str", Kind.STR)(lambda text: show_none(SPELLING, text))

register_text_conversions(RULES, TEXT)
