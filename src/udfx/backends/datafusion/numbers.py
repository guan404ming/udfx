from __future__ import annotations

import operator

import pyarrow as pa
from datafusion import Expr, functions, lit

from udfx.backends.base import (
    BinaryRule,
    Lowered,
    register_comparison,
    register_equality,
    register_float_variants,
)
from udfx.backends.datafusion.rules import RULES, null_safe_equal, when
from udfx.semantics.numbers import (
    ARITHMETIC,
    COMPARISONS,
    FLOAT_INT_MAX,
    Binary,
    FloatPrimitives,
    IntegerPrimitives,
    OrderPrimitives,
    float_divmod,
    int_divmod,
)
from udfx.types import INT64_MAX, INT64_MIN, Kind

WIDE = pa.decimal128(19, 0)
INFINITY = float("inf")


def fits_int64(wide: Expr) -> Expr:
    return wide.between(lit(INT64_MIN), lit(INT64_MAX))


def within_exact_float_range(value: Expr) -> Expr:
    return value.between(lit(-FLOAT_INT_MAX), lit(FLOAT_INT_MAX))


def cast_float64(value: Expr) -> Expr:
    return value.cast(pa.float64())


def overflow_checked(operation: Binary) -> BinaryRule[Expr]:
    def rule(left: Expr, right: Expr) -> Lowered[Expr]:
        wide = operation(left.cast(WIDE), right.cast(WIDE))
        return Lowered(operation(left, right), fits_int64(wide))

    return rule


def is_nan(value: Expr) -> Expr:
    return functions.coalesce(functions.isnan(value), lit(False))


def either_nan(left: Expr, right: Expr) -> Expr:
    return is_nan(left) | is_nan(right)


PRIMITIVES = FloatPrimitives(
    when, functions.floor, functions.signum, lambda left, right: (left % right, None)
)


def float_floordiv(left: Expr, right: Expr) -> Lowered[Expr]:
    quotient, _, exact = float_divmod(PRIMITIVES, left, right)
    return Lowered(quotient, exact)


def float_mod(left: Expr, right: Expr) -> Lowered[Expr]:
    _, remainder, exact = float_divmod(PRIMITIVES, left, right)
    return Lowered(remainder, exact)


INTEGERS = IntegerPrimitives(lit, when, operator.truediv, operator.mod)


def finite(value: Expr) -> Expr:
    return ~functions.isnan(value) & (functions.abs(value) != lit(INFINITY))


ORDER = OrderPrimitives(when, either_nan, Expr.is_null, within_exact_float_range)


for _name, _operation in ARITHMETIC.items():
    RULES(_name, Kind.INT, Kind.INT)(overflow_checked(_operation))
    register_float_variants(RULES, _name, _operation, cast_float64)

for _name, _compare in COMPARISONS.items():
    register_comparison(RULES, _name, _compare, cast_float64, ORDER)

register_float_variants(RULES, "__truediv__", operator.truediv, cast_float64)
register_float_variants(RULES, "__floordiv__", float_floordiv, cast_float64)
register_float_variants(RULES, "__mod__", float_mod, cast_float64)
register_equality(RULES, null_safe_equal, cast_float64, ORDER)


@RULES("__truediv__", Kind.INT, Kind.INT)
def int_truediv(left: Expr, right: Expr) -> Lowered[Expr]:
    return Lowered(
        cast_float64(left) / cast_float64(right),
        within_exact_float_range(left) & within_exact_float_range(right),
    )


@RULES("__floordiv__", Kind.INT, Kind.INT)
def int_floordiv(left: Expr, right: Expr) -> Lowered[Expr]:
    quotient, _, exact = int_divmod(INTEGERS, left, right)
    return Lowered(quotient, exact)


@RULES("__mod__", Kind.INT, Kind.INT)
def int_mod(left: Expr, right: Expr) -> Expr:
    return int_divmod(INTEGERS, left, right)[1]


@RULES("__neg__", Kind.INT)
def int_neg(value: Expr) -> Lowered[Expr]:
    return Lowered(lit(0) - value, value != lit(INT64_MIN))


@RULES("__neg__", Kind.FLOAT)
def float_neg(value: Expr) -> Expr:
    return value * lit(-1.0)


@RULES("abs", Kind.INT)
def int_abs(value: Expr) -> Lowered[Expr]:
    return Lowered(
        functions.abs(functions.greatest(value, lit(-INT64_MAX))), value != lit(INT64_MIN)
    )


@RULES("abs", Kind.FLOAT)
def float_abs(value: Expr) -> Expr:
    return functions.abs(value)


@RULES("math.isnan", Kind.FLOAT)
def float_isnan(value: Expr) -> Expr:
    return functions.isnan(value)


@RULES("math.isinf", Kind.FLOAT)
def float_isinf(value: Expr) -> Expr:
    return functions.abs(value) == lit(INFINITY)


RULES("math.log10", Kind.FLOAT)(functions.log10)


@RULES("math.log10", Kind.INT)
def int_log10(value: Expr) -> Expr:
    return functions.log10(cast_float64(value))
