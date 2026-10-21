from __future__ import annotations

import duckdb
from duckdb import Expression, sqltypes

from udfx.backends.duckdb.numbers import cast_float64, cast_int128, is_nan, try_truncate_int64
from udfx.backends.duckdb.rules import (
    NULL,
    RULES,
    constant,
    function,
    length,
    when,
)
from udfx.backends.registration import Lowered, register_text_conversions
from udfx.semantics.literals import (
    FLOAT_LITERAL,
    INT_LITERAL,
    SpellingPrimitives,
    TextPrimitives,
    bool_text,
    show_none,
)
from udfx.types import INT64_MAX, INT64_MIN, Kind


def matches(text: Expression, pattern: str) -> Expression:
    return function("regexp_matches", text, constant(pattern))


def try_int(digits: Expression) -> Expression:
    parsable = matches(digits, INT_LITERAL) & (length(digits) <= constant(38))
    value = when(parsable, cast_int128(digits), NULL)
    fits = value.between(constant(INT64_MIN), constant(INT64_MAX))
    return when(fits, value, NULL).cast(sqltypes.BIGINT)


def try_float(digits: Expression) -> Expression:
    return when(matches(digits, FLOAT_LITERAL), cast_float64(digits), NULL)


TEXT = TextPrimitives(
    matches=matches,
    strip=lambda text, chars: function("trim", text, constant(chars)),
    replace=lambda text, old, new: function("replace", text, constant(old), constant(new)),
    byte_length=lambda text: function("strlen", text),
    char_length=length,
    not_null=Expression.isnotnull,
    try_int=try_int,
    try_float=try_float,
)

SPELLING = SpellingPrimitives(constant, when, Expression.isnull, duckdb.CoalesceOperator)


@RULES("int", Kind.FLOAT)
def float_int(value: Expression) -> Lowered[Expression]:
    value = try_truncate_int64(value)
    return Lowered(value, value.isnotnull() | ~function("isfinite", value))


RULES("float", Kind.INT)(cast_float64)


@RULES("float", Kind.FLOAT)
def float_float(value: Expression) -> Expression:
    return value


@RULES("str", Kind.INT)
def int_str(value: Expression) -> Expression:
    return show_none(SPELLING, value.cast(sqltypes.VARCHAR))


@RULES("str", Kind.FLOAT)
def float_str(value: Expression) -> Lowered[Expression]:
    return Lowered(
        show_none(SPELLING, value.cast(sqltypes.VARCHAR)),
        ~duckdb.CoalesceOperator(is_nan(value), constant(False)),
    )


@RULES("str", Kind.BOOL)
def bool_str(value: Expression) -> Expression:
    return bool_text(SPELLING, value)


RULES("str", Kind.STR)(lambda text: show_none(SPELLING, text))

register_text_conversions(RULES, TEXT)
