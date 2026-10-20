from __future__ import annotations

import operator
from collections.abc import Callable

import polars as pl

from udfx.backends.base import Lowered
from udfx.backends.polars.rules import RULES, broadcast, when
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

Rule = Callable[[pl.Expr, pl.Expr], pl.Expr | Lowered[pl.Expr]]


def fits_int64(wide: pl.Expr) -> pl.Expr:
    return wide.is_between(INT64_MIN, INT64_MAX)


def representable(a: pl.Expr) -> pl.Expr:
    return a.is_between(-FLOAT_INT_MAX, FLOAT_INT_MAX)


def checked(operation: Binary) -> Rule:
    def rule(a: pl.Expr, b: pl.Expr) -> Lowered[pl.Expr]:
        wide = operation(a.cast(pl.Int128), b.cast(pl.Int128))
        return Lowered(operation(a, b), fits_int64(wide))

    return rule


def floating(name: str, rule: Rule) -> None:
    RULES(name, Kind.FLOAT, Kind.FLOAT)(rule)
    RULES(name, Kind.INT, Kind.FLOAT)(lambda a, b: rule(a.cast(pl.Float64), b))
    RULES(name, Kind.FLOAT, Kind.INT)(lambda a, b: rule(a, b.cast(pl.Float64)))


def comparison(name: str, compare: Binary) -> None:
    def ordered(a: pl.Expr, b: pl.Expr) -> pl.Expr:
        return compare(a, b) & a.is_not_nan() & b.is_not_nan()

    RULES(name, Kind.INT, Kind.INT)(compare)
    RULES(name, Kind.FLOAT, Kind.FLOAT)(ordered)
    RULES(name, Kind.INT, Kind.FLOAT)(
        lambda a, b: Lowered(ordered(a.cast(pl.Float64), b), representable(a))
    )
    RULES(name, Kind.FLOAT, Kind.INT)(
        lambda a, b: Lowered(ordered(a, b.cast(pl.Float64)), representable(b))
    )


def materialized(operation: Binary) -> Binary:
    return lambda left, right: operation(broadcast(left), broadcast(right))


def either_nan(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    return a.is_nan().fill_null(False) | b.is_nan().fill_null(False)


def truncated_remainder(a: pl.Expr, b: pl.Expr) -> tuple[pl.Expr, pl.Expr]:
    whole_a, whole_b = a.cast(pl.Int64, strict=False), b.cast(pl.Int64, strict=False)
    floored = whole_a % whole_b
    truncated = when((floored != 0) & ((a < 0.0) != (b < 0.0)), floored - whole_b, floored)
    undefined = ~a.is_finite() | b.is_nan()
    small = a.abs() < b.abs()
    whole = (whole_a.cast(pl.Float64) == a) & (whole_b.cast(pl.Float64) == b)
    expr = when(undefined, pl.lit(float("nan")), when(small, a, truncated.cast(pl.Float64)))
    return expr, undefined | small | whole.fill_null(False)


PRIMITIVES = FloatPrimitives(when, pl.Expr.floor, pl.Expr.sign, truncated_remainder)


def float_floordiv(a: pl.Expr, b: pl.Expr) -> Lowered[pl.Expr]:
    quotient, _, exact = float_divmod(PRIMITIVES, a, broadcast(b))
    return Lowered(quotient, exact)


def float_mod(a: pl.Expr, b: pl.Expr) -> Lowered[pl.Expr]:
    _, remainder, exact = float_divmod(PRIMITIVES, a, broadcast(b))
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
RULES("__mod__", Kind.INT, Kind.INT)(operator.mod)
RULES("__neg__", Kind.FLOAT)(pl.Expr.__neg__)
RULES("abs", Kind.FLOAT)(pl.Expr.abs)
RULES("math.isnan", Kind.FLOAT)(pl.Expr.is_nan)
RULES("math.isinf", Kind.FLOAT)(pl.Expr.is_infinite)


@RULES("__truediv__", Kind.INT, Kind.INT)
def int_truediv(a: pl.Expr, b: pl.Expr) -> Lowered[pl.Expr]:
    return Lowered(broadcast(a) / broadcast(b), representable(a) & representable(b))


@RULES("__floordiv__", Kind.INT, Kind.INT)
def int_floordiv(a: pl.Expr, b: pl.Expr) -> Lowered[pl.Expr]:
    return Lowered(a // b, ~((a == INT64_MIN) & (b == -1)))


@RULES("__eq__", Kind.FLOAT, Kind.FLOAT)
def float_eq(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    return a.eq_missing(b) & ~either_nan(a, b)


@RULES("__ne__", Kind.FLOAT, Kind.FLOAT)
def float_ne(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    return a.ne_missing(b) | either_nan(a, b)


@RULES("__neg__", Kind.INT)
def int_neg(a: pl.Expr) -> Lowered[pl.Expr]:
    return Lowered(-a, a != INT64_MIN)


@RULES("abs", Kind.INT)
def int_abs(a: pl.Expr) -> Lowered[pl.Expr]:
    return Lowered(a.abs(), a != INT64_MIN)


@RULES("int", Kind.FLOAT)
def float_int(a: pl.Expr) -> Lowered[pl.Expr]:
    whole = a.cast(pl.Int64, strict=False)
    return Lowered(whole, whole.is_not_null() | ~a.is_finite())


@RULES("float", Kind.INT)
def int_float(a: pl.Expr) -> pl.Expr:
    return a.cast(pl.Float64)


@RULES("float", Kind.FLOAT)
def float_float(value: pl.Expr) -> pl.Expr:
    return value


@RULES("str", Kind.INT)
def int_str(a: pl.Expr) -> pl.Expr:
    return a.cast(pl.String)


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


def log10(a: pl.Expr) -> Lowered[pl.Expr]:
    return Lowered(a.log10(), pl.lit(False))


RULES("math.log10", Kind.FLOAT)(log10)


@RULES("math.log10", Kind.INT)
def int_log10(a: pl.Expr) -> Lowered[pl.Expr]:
    return log10(a.cast(pl.Float64))
