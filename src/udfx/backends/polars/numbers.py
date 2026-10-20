from __future__ import annotations

import operator
import sys
from collections.abc import Callable

import polars as pl

from udfx.backends.base import Lowered
from udfx.backends.polars.rules import RULES
from udfx.semantics.literals import ASCII_SPACE, FLOAT_LITERAL, INT_LITERAL
from udfx.types import INT64_MAX, INT64_MIN, Kind

FLOAT_INT_MAX = 2**53

Binary = Callable[[pl.Expr, pl.Expr], pl.Expr]
Rule = Callable[[pl.Expr, pl.Expr], pl.Expr | Lowered[pl.Expr]]

ARITHMETIC: dict[str, Binary] = {
    "__add__": operator.add,
    "__sub__": operator.sub,
    "__mul__": operator.mul,
}

COMPARISONS: dict[str, Binary] = {
    "__lt__": operator.lt,
    "__le__": operator.le,
    "__gt__": operator.gt,
    "__ge__": operator.ge,
}


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


def dense(value: pl.Expr) -> pl.Expr:
    return pl.int_range(pl.len()).cast(pl.Float64) * -0.0 + value


def divide(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    return a / dense(b)


def either_nan(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    return a.is_nan().fill_null(False) | b.is_nan().fill_null(False)


def signed_zero(b: pl.Expr) -> pl.Expr:
    return pl.when(b < 0.0).then(-0.0).otherwise(0.0)


def truncated_remainder(a: pl.Expr, b: pl.Expr) -> tuple[pl.Expr, pl.Expr]:
    whole_a, whole_b = a.cast(pl.Int64, strict=False), b.cast(pl.Int64, strict=False)
    floored = whole_a % whole_b
    truncated = (
        pl.when((floored != 0) & ((a < 0.0) != (b < 0.0)))
        .then(floored - whole_b)
        .otherwise(floored)
    )
    undefined = ~a.is_finite() | b.is_nan()
    small = a.abs() < b.abs()
    whole = (whole_a.cast(pl.Float64) == a) & (whole_b.cast(pl.Float64) == b)
    expr = (
        pl.when(undefined)
        .then(float("nan"))
        .when(small)
        .then(a)
        .otherwise(truncated.cast(pl.Float64))
    )
    return expr, undefined | small | whole.fill_null(False)


def divmod_(a: pl.Expr, b: pl.Expr) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    remainder, exact = truncated_remainder(a, b)
    quotient = divide(a - remainder, b)
    adjust = (remainder != 0.0) & ((b < 0.0) != (remainder < 0.0))
    remainder = (
        pl.when(remainder == 0.0)
        .then(signed_zero(b))
        .when(adjust)
        .then(remainder + b)
        .otherwise(remainder)
    )
    quotient = pl.when(adjust).then(quotient - 1.0).otherwise(quotient)
    floored = quotient.floor()
    snapped = pl.when(quotient - floored > 0.5).then(floored + 1.0).otherwise(floored)
    quotient = pl.when(quotient != 0.0).then(snapped).otherwise(divide(a, b) * 0.0)
    return quotient, remainder, exact


def float_floordiv(a: pl.Expr, b: pl.Expr) -> Lowered[pl.Expr]:
    quotient, _, exact = divmod_(a, b)
    return Lowered(quotient, exact)


def float_mod(a: pl.Expr, b: pl.Expr) -> Lowered[pl.Expr]:
    _, remainder, exact = divmod_(a, b)
    return Lowered(remainder, exact)


def is_ascii(a: pl.Expr) -> pl.Expr:
    return a.str.len_bytes() == a.str.len_chars()


def cleaned(a: pl.Expr) -> pl.Expr:
    return a.str.strip_chars(ASCII_SPACE).str.replace_all("_", "", literal=True)


for _name, _operation in ARITHMETIC.items():
    RULES(_name, Kind.INT, Kind.INT)(checked(_operation))
    floating(_name, _operation)

for _name, _compare in COMPARISONS.items():
    comparison(_name, _compare)

floating("__truediv__", divide)
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
    return Lowered(divide(a, b), representable(a) & representable(b))


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
def str_int_parses(a: pl.Expr) -> Lowered[pl.Expr]:
    limit = sys.get_int_max_str_digits() or sys.maxsize
    return Lowered(a.str.contains(INT_LITERAL), is_ascii(a) & (a.str.len_bytes() <= limit))


@RULES("int", Kind.STR)
def str_int(a: pl.Expr) -> Lowered[pl.Expr]:
    parsed = cleaned(a).str.to_integer(strict=False)
    return Lowered(parsed, is_ascii(a) & (parsed.is_not_null() | ~a.str.contains(INT_LITERAL)))


@RULES("float_parses", Kind.STR)
def str_float_parses(a: pl.Expr) -> Lowered[pl.Expr]:
    return Lowered(a.str.contains(FLOAT_LITERAL), is_ascii(a))


@RULES("float", Kind.STR)
def str_float(a: pl.Expr) -> Lowered[pl.Expr]:
    parsed = cleaned(a).cast(pl.Float64, strict=False)
    return Lowered(parsed, is_ascii(a) & (parsed.is_not_null() | ~a.str.contains(FLOAT_LITERAL)))


def log10(a: pl.Expr) -> Lowered[pl.Expr]:
    return Lowered(a.log10(), pl.lit(False))


RULES("math.log10", Kind.FLOAT)(log10)


@RULES("math.log10", Kind.INT)
def int_log10(a: pl.Expr) -> Lowered[pl.Expr]:
    return log10(a.cast(pl.Float64))
