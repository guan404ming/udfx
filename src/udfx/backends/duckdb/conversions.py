from __future__ import annotations

import duckdb
from duckdb import Expression, sqltypes

from udfx.backends.base import Lowered
from udfx.backends.duckdb.numbers import cast_float64, cast_int128, is_nan, try_truncate_int64
from udfx.backends.duckdb.rules import (
    NULL,
    RULES,
    constant,
    function,
    length,
    when,
)
from udfx.semantics.literals import (
    FLOAT_LITERAL,
    INT_LITERAL,
    TextPrimitives,
    float_parses,
    int_parses,
    parse_float,
    parse_int,
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


@RULES("int", Kind.FLOAT)
def float_int(value: Expression) -> Lowered[Expression]:
    value = try_truncate_int64(value)
    return Lowered(value, value.isnotnull() | ~function("isfinite", value))


RULES("float", Kind.INT)(cast_float64)


@RULES("float", Kind.FLOAT)
def float_float(value: Expression) -> Expression:
    return value


def show_none(text: Expression) -> Expression:
    return duckdb.CoalesceOperator(text, constant("None"))


@RULES("str", Kind.INT)
def int_str(value: Expression) -> Expression:
    return show_none(value.cast(sqltypes.VARCHAR))


@RULES("str", Kind.FLOAT)
def float_str(value: Expression) -> Lowered[Expression]:
    return Lowered(
        show_none(value.cast(sqltypes.VARCHAR)),
        ~duckdb.CoalesceOperator(is_nan(value), constant(False)),
    )


@RULES("str", Kind.BOOL)
def bool_str(value: Expression) -> Expression:
    named = when(value, constant("True"), constant("False"))
    return when(value.isnull(), constant("None"), named)


RULES("str", Kind.STR)(show_none)


@RULES("int_parses", Kind.STR)
def str_int_parses(text: Expression) -> Lowered[Expression]:
    return Lowered(*int_parses(TEXT, text))


@RULES("int", Kind.STR)
def str_int(text: Expression) -> Lowered[Expression]:
    return Lowered(*parse_int(TEXT, text))


@RULES("float_parses", Kind.STR)
def str_float_parses(text: Expression) -> Lowered[Expression]:
    return Lowered(*float_parses(TEXT, text))


@RULES("float", Kind.STR)
def str_float(text: Expression) -> Lowered[Expression]:
    return Lowered(*parse_float(TEXT, text))
