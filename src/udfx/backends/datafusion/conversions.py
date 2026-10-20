from __future__ import annotations

import pyarrow as pa
from datafusion import Expr, functions, lit

from udfx.backends.base import Lowered
from udfx.backends.datafusion.numbers import cast_float64, finite
from udfx.backends.datafusion.rules import RULES, length, when
from udfx.semantics.literals import (
    TextPrimitives,
    float_parses,
    int_parses,
    parse_float,
    parse_int,
)
from udfx.semantics.numbers import (
    REPR_FIXED_MAX,
    REPR_FIXED_MIN,
)
from udfx.types import Kind

TEXT = TextPrimitives(
    matches=lambda text, pattern: functions.regexp_like(text, lit(pattern)),
    strip=lambda text, chars: functions.btrim(text, lit(chars)),
    replace=lambda text, old, new: functions.replace(text, lit(old), lit(new)),
    byte_length=functions.octet_length,
    char_length=length,
    not_null=Expr.is_not_null,
    try_int=lambda text: text.try_cast(pa.int64()),
    try_float=lambda text: text.try_cast(pa.float64()),
)


@RULES("int", Kind.FLOAT)
def float_int(value: Expr) -> Lowered[Expr]:
    whole = value.try_cast(pa.int64())
    return Lowered(whole, whole.is_not_null() | ~finite(value))


@RULES("float", Kind.INT)
def int_float(value: Expr) -> Expr:
    return cast_float64(value)


@RULES("float", Kind.FLOAT)
def float_float(value: Expr) -> Expr:
    return value


def show_none(text: Expr) -> Expr:
    return functions.coalesce(text, lit("None"))


@RULES("str", Kind.INT)
def int_str(value: Expr) -> Expr:
    return show_none(value.cast(pa.string()))


@RULES("str", Kind.FLOAT)
def float_str(value: Expr) -> Lowered[Expr]:
    size = functions.abs(value)
    fixed = (size >= lit(REPR_FIXED_MIN)) & (size < lit(REPR_FIXED_MAX))
    exact = value.is_null() | (value == lit(0.0)) | fixed
    return Lowered(show_none(value.cast(pa.string())), exact)


@RULES("str", Kind.BOOL)
def bool_str(value: Expr) -> Expr:
    return when(value.is_null(), lit("None"), when(value, lit("True"), lit("False")))


RULES("str", Kind.STR)(show_none)


@RULES("int_parses", Kind.STR)
def str_int_parses(text: Expr) -> Lowered[Expr]:
    return Lowered(*int_parses(TEXT, text))


@RULES("int", Kind.STR)
def str_int(text: Expr) -> Lowered[Expr]:
    return Lowered(*parse_int(TEXT, text))


@RULES("float_parses", Kind.STR)
def str_float_parses(text: Expr) -> Lowered[Expr]:
    return Lowered(*float_parses(TEXT, text))


@RULES("float", Kind.STR)
def str_float(text: Expr) -> Lowered[Expr]:
    return Lowered(*parse_float(TEXT, text))
