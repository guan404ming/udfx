from __future__ import annotations

import operator

import polars as pl

from udfx.backends.base import BinaryRule, Lowered
from udfx.backends.polars.rules import RULES, broadcast, when
from udfx.ops.numbers import REPR_FIXED_MAX, REPR_FIXED_MIN
from udfx.semantics.literals import (
    TextPrimitives,
    float_parses,
    int_parses,
    parse_float,
    parse_int,
)
from udfx.semantics.numbers import (
    ARITHMETIC,
    COMPARISONS,
    FLOAT_INT_MAX,
    Binary,
    FloatPrimitives,
    float_divmod,
)
from udfx.types import INT64_MAX, INT64_MIN, Kind


def fits_int64(wide: pl.Expr) -> pl.Expr:
    return wide.is_between(INT64_MIN, INT64_MAX)


def representable(value: pl.Expr) -> pl.Expr:
    return value.is_between(-FLOAT_INT_MAX, FLOAT_INT_MAX)


def checked(operation: Binary) -> BinaryRule[pl.Expr]:
    def rule(left: pl.Expr, right: pl.Expr) -> Lowered[pl.Expr]:
        wide = operation(left.cast(pl.Int128), right.cast(pl.Int128))
        return Lowered(operation(left, right), fits_int64(wide))

    return rule


def floating(name: str, rule: BinaryRule[pl.Expr]) -> None:
    RULES(name, Kind.FLOAT, Kind.FLOAT)(rule)
    RULES(name, Kind.INT, Kind.FLOAT)(lambda left, right: rule(left.cast(pl.Float64), right))
    RULES(name, Kind.FLOAT, Kind.INT)(lambda left, right: rule(left, right.cast(pl.Float64)))


def widened(name: str, rule: Binary) -> None:
    def exact(value: pl.Expr) -> pl.Expr:
        return value.is_null() | representable(value)

    RULES(name, Kind.INT, Kind.FLOAT)(
        lambda left, right: Lowered(rule(left.cast(pl.Float64), right), exact(left))
    )
    RULES(name, Kind.FLOAT, Kind.INT)(
        lambda left, right: Lowered(rule(left, right.cast(pl.Float64)), exact(right))
    )


def comparison(name: str, compare: Binary) -> None:
    def ordered(left: pl.Expr, right: pl.Expr) -> pl.Expr:
        return compare(left, right) & left.is_not_nan() & right.is_not_nan()

    RULES(name, Kind.INT, Kind.INT)(compare)
    RULES(name, Kind.FLOAT, Kind.FLOAT)(ordered)
    widened(name, ordered)


def materialized(operation: Binary) -> Binary:
    return lambda left, right: operation(broadcast(left), broadcast(right))


def either_nan(left: pl.Expr, right: pl.Expr) -> pl.Expr:
    return left.is_nan().fill_null(False) | right.is_nan().fill_null(False)


def truncated_remainder(left: pl.Expr, right: pl.Expr) -> tuple[pl.Expr, pl.Expr]:
    whole_left, whole_right = left.cast(pl.Int64, strict=False), right.cast(pl.Int64, strict=False)
    floored = whole_left % whole_right
    truncated = when(
        (floored != 0) & ((left < 0.0) != (right < 0.0)), floored - whole_right, floored
    )
    undefined = ~left.is_finite() | right.is_nan()
    small = left.abs() < right.abs()
    whole = (whole_left.cast(pl.Float64) == left) & (whole_right.cast(pl.Float64) == right)
    expr = when(undefined, pl.lit(float("nan")), when(small, left, truncated.cast(pl.Float64)))
    return expr, undefined | small | whole.fill_null(False)


PRIMITIVES = FloatPrimitives(when, pl.Expr.floor, pl.Expr.sign, truncated_remainder)


def float_floordiv(left: pl.Expr, right: pl.Expr) -> Lowered[pl.Expr]:
    quotient, _, exact = float_divmod(PRIMITIVES, left, broadcast(right))
    return Lowered(quotient, exact)


def float_mod(left: pl.Expr, right: pl.Expr) -> Lowered[pl.Expr]:
    _, remainder, exact = float_divmod(PRIMITIVES, left, broadcast(right))
    return Lowered(remainder, exact)


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


for _name, _operation in ARITHMETIC.items():
    RULES(_name, Kind.INT, Kind.INT)(checked(_operation))
    floating(_name, materialized(_operation))

for _name, _compare in COMPARISONS.items():
    comparison(_name, _compare)

floating("__truediv__", materialized(operator.truediv))
floating("__floordiv__", float_floordiv)
floating("__mod__", float_mod)

RULES("__eq__", Kind.INT, Kind.INT)(pl.Expr.eq_missing)
RULES("__ne__", Kind.INT, Kind.INT)(pl.Expr.ne_missing)
RULES("__mod__", Kind.INT, Kind.INT)(pl.Expr.__mod__)
RULES("__neg__", Kind.FLOAT)(pl.Expr.__neg__)
RULES("abs", Kind.FLOAT)(pl.Expr.abs)
RULES("math.isnan", Kind.FLOAT)(pl.Expr.is_nan)
RULES("math.isinf", Kind.FLOAT)(pl.Expr.is_infinite)


@RULES("__truediv__", Kind.INT, Kind.INT)
def int_truediv(left: pl.Expr, right: pl.Expr) -> Lowered[pl.Expr]:
    return Lowered(broadcast(left) / broadcast(right), representable(left) & representable(right))


@RULES("__floordiv__", Kind.INT, Kind.INT)
def int_floordiv(left: pl.Expr, right: pl.Expr) -> Lowered[pl.Expr]:
    return Lowered(left // right, ~((left == INT64_MIN) & (right == -1)))


@RULES("__eq__", Kind.FLOAT, Kind.FLOAT)
def float_eq(left: pl.Expr, right: pl.Expr) -> pl.Expr:
    return left.eq_missing(right) & ~either_nan(left, right)


@RULES("__ne__", Kind.FLOAT, Kind.FLOAT)
def float_ne(left: pl.Expr, right: pl.Expr) -> pl.Expr:
    return left.ne_missing(right) | either_nan(left, right)


widened("__eq__", float_eq)
widened("__ne__", float_ne)


@RULES("__neg__", Kind.INT)
def int_neg(value: pl.Expr) -> Lowered[pl.Expr]:
    return Lowered(-value, value != INT64_MIN)


@RULES("abs", Kind.INT)
def int_abs(value: pl.Expr) -> Lowered[pl.Expr]:
    return Lowered(value.abs(), value != INT64_MIN)


@RULES("int", Kind.FLOAT)
def float_int(value: pl.Expr) -> Lowered[pl.Expr]:
    whole = value.cast(pl.Int64, strict=False)
    return Lowered(whole, whole.is_not_null() | ~value.is_finite())


@RULES("float", Kind.INT)
def int_float(value: pl.Expr) -> pl.Expr:
    return value.cast(pl.Float64)


@RULES("float", Kind.FLOAT)
def float_float(value: pl.Expr) -> pl.Expr:
    return value


@RULES("str", Kind.INT)
def int_str(value: pl.Expr) -> pl.Expr:
    return value.cast(pl.String).fill_null("None")


@RULES("str", Kind.FLOAT)
def float_str(value: pl.Expr) -> Lowered[pl.Expr]:
    fixed = value.abs().is_between(REPR_FIXED_MIN, REPR_FIXED_MAX, closed="left")
    return Lowered(
        value.cast(pl.String).fill_null("None"), value.is_null() | (value == 0.0) | fixed
    )


@RULES("str", Kind.BOOL)
def bool_str(value: pl.Expr) -> pl.Expr:
    named = when(value, pl.lit("True"), pl.lit("False"))
    return when(value.is_null(), pl.lit("None"), named)


@RULES("str", Kind.STR)
def str_str(value: pl.Expr) -> pl.Expr:
    return value.fill_null("None")


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


def log10(value: pl.Expr) -> Lowered[pl.Expr]:
    return Lowered(value.log10(), pl.lit(False))


RULES("math.log10", Kind.FLOAT)(log10)


@RULES("math.log10", Kind.INT)
def int_log10(value: pl.Expr) -> Lowered[pl.Expr]:
    return log10(value.cast(pl.Float64))
