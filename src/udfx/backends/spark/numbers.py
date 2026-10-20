from __future__ import annotations

from collections.abc import Callable

from pyspark.sql import Column, functions

from udfx.backends.base import (
    BinaryRule,
    Lowered,
    register_comparison,
    register_equality,
    register_float_variants,
)
from udfx.backends.spark.rules import RULES, when
from udfx.semantics.numbers import (
    ARITHMETIC,
    COMPARISONS,
    FLOAT_INT_MAX,
    FloatPrimitives,
    IntegerPrimitives,
    OrderPrimitives,
    float_divmod,
    int_divmod,
)
from udfx.types import INT64_MAX, INT64_MIN, Kind

CHECKED: dict[str, Callable[[Column, Column], Column]] = {
    "__add__": functions.try_add,
    "__sub__": functions.try_subtract,
    "__mul__": functions.try_multiply,
}

INFINITY = float("inf")


def cast_float64(value: Column) -> Column:
    return value.cast("double")


def within_exact_float_range(value: Column) -> Column:
    return value.between(-FLOAT_INT_MAX, FLOAT_INT_MAX)


def overflow_checked(operation: Callable[[Column, Column], Column]) -> BinaryRule[Column]:
    def rule(left: Column, right: Column) -> Lowered[Column]:
        result = operation(left, right)
        return Lowered(result, result.isNotNull())

    return rule


def either_nan(left: Column, right: Column) -> Column:
    return functions.isnan(left) | functions.isnan(right)


def nonzero(divisor: Column) -> Column:
    return when(divisor == 0, functions.lit(1), divisor)


def floor(value: Column) -> Column:
    rounded = functions.rint(value)
    return when(rounded > value, rounded - 1.0, rounded)


PRIMITIVES = FloatPrimitives(
    when, floor, functions.signum, lambda left, right: (left % right, None)
)


def float_floordiv(left: Column, right: Column) -> Lowered[Column]:
    quotient, _, exact = float_divmod(PRIMITIVES, left, nonzero(right))
    return Lowered(quotient, exact)


def float_mod(left: Column, right: Column) -> Lowered[Column]:
    _, remainder, exact = float_divmod(PRIMITIVES, left, nonzero(right))
    return Lowered(remainder, exact)


INTEGERS = IntegerPrimitives(
    functions.lit, when, lambda left, right: functions.call_function("div", left, right)
)


def float_isinf(value: Column) -> Column:
    return functions.abs(value) == INFINITY


ORDER = OrderPrimitives(when, either_nan, Column.isNull, within_exact_float_range)


for _name, _operation in ARITHMETIC.items():
    RULES(_name, Kind.INT, Kind.INT)(overflow_checked(CHECKED[_name]))
    register_float_variants(RULES, _name, _operation, cast_float64)

for _name, _compare in COMPARISONS.items():
    register_comparison(RULES, _name, _compare, cast_float64, ORDER)

register_float_variants(RULES, "__truediv__", functions.try_divide, cast_float64)
register_float_variants(RULES, "__floordiv__", float_floordiv, cast_float64)
register_float_variants(RULES, "__mod__", float_mod, cast_float64)
register_equality(RULES, Column.eqNullSafe, cast_float64, ORDER)

RULES("__mod__", Kind.INT, Kind.INT)(lambda left, right: int_divmod(INTEGERS, left, right)[1])
RULES("__neg__", Kind.FLOAT)(Column.__neg__)
RULES("abs", Kind.FLOAT)(functions.abs)
RULES("math.isnan", Kind.FLOAT)(functions.isnan)
RULES("math.isinf", Kind.FLOAT)(float_isinf)


@RULES("__truediv__", Kind.INT, Kind.INT)
def int_truediv(left: Column, right: Column) -> Lowered[Column]:
    return Lowered(
        functions.try_divide(left, right),
        within_exact_float_range(left) & within_exact_float_range(right),
    )


@RULES("__floordiv__", Kind.INT, Kind.INT)
def int_floordiv(left: Column, right: Column) -> Lowered[Column]:
    quotient, _, exact = int_divmod(INTEGERS, left, right)
    return Lowered(quotient, exact)


@RULES("__neg__", Kind.INT)
def int_neg(value: Column) -> Lowered[Column]:
    return Lowered(functions.try_subtract(functions.lit(0), value), value != INT64_MIN)


@RULES("abs", Kind.INT)
def int_abs(value: Column) -> Lowered[Column]:
    return Lowered(
        functions.abs(functions.greatest(value, functions.lit(-INT64_MAX))), value != INT64_MIN
    )


def log10(value: Column) -> Lowered[Column]:
    return Lowered(functions.log10(value), functions.lit(False))


RULES("math.log10", Kind.FLOAT)(log10)


@RULES("math.log10", Kind.INT)
def int_log10(value: Column) -> Lowered[Column]:
    return log10(cast_float64(value))
