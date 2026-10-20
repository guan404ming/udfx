from __future__ import annotations

import operator

import polars as pl

from udfx.backends.base import (
    BinaryRule,
    Lowered,
    register_comparison,
    register_float_variants,
    register_widened_variants,
)
from udfx.backends.polars.rules import RULES, broadcast, when
from udfx.semantics.numbers import (
    ARITHMETIC,
    COMPARISONS,
    FLOAT_INT_MAX,
    Binary,
    FloatPrimitives,
    float_divmod,
)
from udfx.types import INT64_MAX, INT64_MIN, Kind


def cast_float64(value: pl.Expr) -> pl.Expr:
    return value.cast(pl.Float64)


def fits_int64(wide: pl.Expr) -> pl.Expr:
    return wide.is_between(INT64_MIN, INT64_MAX)


def within_exact_float_range(value: pl.Expr) -> pl.Expr:
    return value.is_between(-FLOAT_INT_MAX, FLOAT_INT_MAX)


def overflow_checked(operation: Binary) -> BinaryRule[pl.Expr]:
    def rule(left: pl.Expr, right: pl.Expr) -> Lowered[pl.Expr]:
        wide = operation(left.cast(pl.Int128), right.cast(pl.Int128))
        return Lowered(operation(left, right), fits_int64(wide))

    return rule


def widened_exact(value: pl.Expr) -> pl.Expr:
    return value.is_null() | within_exact_float_range(value)


def ordered(left: pl.Expr, right: pl.Expr) -> pl.Expr:
    return left.is_not_nan() & right.is_not_nan()


def broadcast_arguments(operation: Binary) -> Binary:
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
    whole = (cast_float64(whole_left) == left) & (cast_float64(whole_right) == right)
    expr = when(undefined, pl.lit(float("nan")), when(small, left, cast_float64(truncated)))
    return expr, undefined | small | whole.fill_null(False)


PRIMITIVES = FloatPrimitives(when, pl.Expr.floor, pl.Expr.sign, truncated_remainder)


def float_floordiv(left: pl.Expr, right: pl.Expr) -> Lowered[pl.Expr]:
    quotient, _, exact = float_divmod(PRIMITIVES, left, broadcast(right))
    return Lowered(quotient, exact)


def float_mod(left: pl.Expr, right: pl.Expr) -> Lowered[pl.Expr]:
    _, remainder, exact = float_divmod(PRIMITIVES, left, broadcast(right))
    return Lowered(remainder, exact)


for _name, _operation in ARITHMETIC.items():
    RULES(_name, Kind.INT, Kind.INT)(overflow_checked(_operation))
    register_float_variants(RULES, _name, broadcast_arguments(_operation), cast_float64)

for _name, _compare in COMPARISONS.items():
    register_comparison(RULES, _name, _compare, ordered, to_float=cast_float64, exact=widened_exact)

register_float_variants(RULES, "__truediv__", broadcast_arguments(operator.truediv), cast_float64)
register_float_variants(RULES, "__floordiv__", float_floordiv, cast_float64)
register_float_variants(RULES, "__mod__", float_mod, cast_float64)

RULES("__eq__", Kind.INT, Kind.INT)(pl.Expr.eq_missing)
RULES("__ne__", Kind.INT, Kind.INT)(pl.Expr.ne_missing)
RULES("__mod__", Kind.INT, Kind.INT)(pl.Expr.__mod__)
RULES("__neg__", Kind.FLOAT)(pl.Expr.__neg__)
RULES("abs", Kind.FLOAT)(pl.Expr.abs)
RULES("math.isnan", Kind.FLOAT)(pl.Expr.is_nan)
RULES("math.isinf", Kind.FLOAT)(pl.Expr.is_infinite)


@RULES("__truediv__", Kind.INT, Kind.INT)
def int_truediv(left: pl.Expr, right: pl.Expr) -> Lowered[pl.Expr]:
    return Lowered(
        broadcast(left) / broadcast(right),
        within_exact_float_range(left) & within_exact_float_range(right),
    )


@RULES("__floordiv__", Kind.INT, Kind.INT)
def int_floordiv(left: pl.Expr, right: pl.Expr) -> Lowered[pl.Expr]:
    return Lowered(left // right, ~((left == INT64_MIN) & (right == -1)))


@RULES("__eq__", Kind.FLOAT, Kind.FLOAT)
def float_eq(left: pl.Expr, right: pl.Expr) -> pl.Expr:
    return left.eq_missing(right) & ~either_nan(left, right)


@RULES("__ne__", Kind.FLOAT, Kind.FLOAT)
def float_ne(left: pl.Expr, right: pl.Expr) -> pl.Expr:
    return left.ne_missing(right) | either_nan(left, right)


register_widened_variants(RULES, "__eq__", float_eq, to_float=cast_float64, exact=widened_exact)
register_widened_variants(RULES, "__ne__", float_ne, to_float=cast_float64, exact=widened_exact)


@RULES("__neg__", Kind.INT)
def int_neg(value: pl.Expr) -> Lowered[pl.Expr]:
    return Lowered(-value, value != INT64_MIN)


@RULES("abs", Kind.INT)
def int_abs(value: pl.Expr) -> Lowered[pl.Expr]:
    return Lowered(value.abs(), value != INT64_MIN)


def log10(value: pl.Expr) -> Lowered[pl.Expr]:
    return Lowered(value.log10(), pl.lit(False))


RULES("math.log10", Kind.FLOAT)(log10)


@RULES("math.log10", Kind.INT)
def int_log10(value: pl.Expr) -> Lowered[pl.Expr]:
    return log10(cast_float64(value))
