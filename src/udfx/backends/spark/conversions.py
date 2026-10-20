from __future__ import annotations

from pyspark.sql import Column, functions

from udfx.backends.base import Lowered
from udfx.backends.spark.numbers import cast_float64, float_isinf
from udfx.backends.spark.rules import RULES, when
from udfx.semantics.literals import (
    TextPrimitives,
    float_parses,
    int_parses,
    parse_float,
    parse_int,
)
from udfx.types import INT64_MAX, INT64_MIN, Kind

JAVA_FIXED_MIN = 1e-3
JAVA_FIXED_MAX = 1e7


TEXT = TextPrimitives(
    matches=lambda text, pattern: text.rlike(pattern),
    strip=lambda text, chars: functions.btrim(text, functions.lit(chars)),
    replace=lambda text, old, new: functions.replace(text, functions.lit(old), functions.lit(new)),
    byte_length=functions.octet_length,
    char_length=functions.length,
    not_null=Column.isNotNull,
    try_int=lambda text: text.try_cast("bigint"),
    try_float=lambda text: text.try_cast("double"),
)


@RULES("int", Kind.FLOAT)
def float_int(value: Column) -> Lowered[Column]:
    in_range = (value >= float(INT64_MIN)) & (value < float(INT64_MAX))
    return Lowered(value.try_cast("bigint"), in_range | functions.isnan(value) | float_isinf(value))


@RULES("float", Kind.INT)
def int_float(value: Column) -> Column:
    return cast_float64(value)


@RULES("float", Kind.FLOAT)
def float_float(value: Column) -> Column:
    return value


def show_none(text: Column) -> Column:
    return functions.coalesce(text, functions.lit("None"))


@RULES("str", Kind.INT)
def int_str(value: Column) -> Column:
    return show_none(value.cast("string"))


@RULES("str", Kind.FLOAT)
def float_str(value: Column) -> Lowered[Column]:
    size = functions.abs(value)
    fixed = (size >= JAVA_FIXED_MIN) & (size < JAVA_FIXED_MAX)
    return Lowered(show_none(value.cast("string")), value.isNull() | (value == 0.0) | fixed)


@RULES("str", Kind.BOOL)
def bool_str(value: Column) -> Column:
    named = when(value, functions.lit("True"), functions.lit("False"))
    return when(value.isNull(), functions.lit("None"), named)


RULES("str", Kind.STR)(show_none)


@RULES("int_parses", Kind.STR)
def str_int_parses(text: Column) -> Lowered[Column]:
    return Lowered(*int_parses(TEXT, text))


@RULES("int", Kind.STR)
def str_int(text: Column) -> Lowered[Column]:
    return Lowered(*parse_int(TEXT, text))


@RULES("float_parses", Kind.STR)
def str_float_parses(text: Column) -> Lowered[Column]:
    return Lowered(*float_parses(TEXT, text))


@RULES("float", Kind.STR)
def str_float(text: Column) -> Lowered[Column]:
    return Lowered(*parse_float(TEXT, text))
