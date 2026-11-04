from __future__ import annotations

from functools import partial

from pyspark.sql import Column, functions

from udfx.backends.engines.spark.bindings import bind
from udfx.backends.engines.spark.numbers import cast_float64, float_isinf
from udfx.backends.engines.spark.regex_categories import category
from udfx.backends.engines.spark.rules import RULES, when
from udfx.backends.registration.shared import Lowered, register_json, register_text_conversions
from udfx.semantics.numbers.float_text import FloatSpelling, FloatTextPrimitives, float_repr
from udfx.semantics.text.literals import (
    ASCII_SPACE,
    SpellingPrimitives,
    TextPrimitives,
    bool_text,
    show_none,
)
from udfx.types import INT64_MAX, INT64_MIN, Kind

JAVA_FIXED_MIN = 1e-3
JAVA_FIXED_MAX = 1e7


TEXT = TextPrimitives(
    literal=functions.lit,
    when=when,
    coalesce=functions.coalesce,
    translate=functions.translate,
    matches=lambda text, pattern: text.rlike(pattern),
    strip=lambda text, chars: functions.btrim(text, functions.lit(chars)),
    replace=lambda text, old, new: functions.replace(text, functions.lit(old), functions.lit(new)),
    char_length=functions.length,
    not_null=Column.isNotNull,
    try_int=lambda text: functions.btrim(text, functions.lit(ASCII_SPACE)).try_cast("bigint"),
    try_float=lambda text: text.try_cast("double"),
    lazy_coalesce=True,
    possessive=True,
    decimal_pattern=partial(category, "CATEGORY_DIGIT"),
)

SPELLING = SpellingPrimitives(functions.lit, when, Column.isNull, functions.coalesce)

FLOAT_TEXT = FloatTextPrimitives(
    literal=functions.lit,
    when=when,
    concat=functions.concat,
    abs=functions.abs,
    is_null=Column.isNull,
    contains=Column.contains,
    floor=functions.floor,
    format=functions.format_string,
    one_digit=lambda value: functions.format_string("%.0e", value),
    trim_right=lambda text, characters: functions.rtrim(text, functions.lit(characters)),
    before=lambda text, marker: functions.substring_index(text, marker, 1),
    after=lambda text, marker: functions.substring_index(text, marker, -1),
    to_int=lambda text: text.try_cast("int"),
    to_float=lambda text: text.try_cast("double"),
)


@RULES("int", Kind.FLOAT)
def float_int(value: Column) -> Lowered[Column]:
    in_range = (value >= float(INT64_MIN)) & (value < float(INT64_MAX))
    return Lowered(
        value.try_cast("bigint"), overflow=~(in_range | functions.isnan(value) | float_isinf(value))
    )


RULES("float", Kind.INT)(cast_float64)


@RULES("float", Kind.FLOAT)
def float_float(value: Column) -> Column:
    return value


@RULES("str", Kind.INT)
def int_str(value: Column) -> Column:
    return show_none(SPELLING, value.cast("string"))


@RULES("str", Kind.FLOAT)
def float_str(value: Column) -> Column:
    size = functions.abs(value)
    plain = value.isNull() | (value == 0.0) | ((size >= JAVA_FIXED_MIN) & (size < JAVA_FIXED_MAX))
    rare = bind(value, float_text, tracked=True)
    return show_none(SPELLING, when(plain, value.cast("string"), rare))


def float_text(value: Column) -> Column:
    plain = value.cast("decimal(38,22)").cast("string")
    spelling = FloatSpelling(value.cast("string"), plain, "E", "NaN", "Infinity")
    return float_repr(FLOAT_TEXT, value, spelling)


@RULES("str", Kind.BOOL)
def bool_str(value: Column) -> Column:
    return bool_text(SPELLING, value)


RULES("str", Kind.STR)(lambda text: show_none(SPELLING, text))

register_text_conversions(RULES, TEXT)
register_json(
    RULES,
    TEXT,
    functions.concat,
    lambda text, pattern: functions.regexp_extract_all(text, functions.lit(pattern), 0),
    possessive=True,
)
