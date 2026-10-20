from __future__ import annotations

from collections.abc import Callable

from pyspark.sql import Column, functions

from udfx.backends.base import BinaryRule, Lowered
from udfx.backends.spark.rules import RULES, when
from udfx.semantics.numbers import (
    ARITHMETIC,
    COMPARISONS,
    FLOAT_INT_MAX,
    Binary,
    FloatPrimitives,
    float_divmod,
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


def representable(value: Column) -> Column:
    return value.between(-FLOAT_INT_MAX, FLOAT_INT_MAX)


def overflow_checked(operation: Callable[[Column, Column], Column]) -> BinaryRule[Column]:
    def rule(left: Column, right: Column) -> Lowered[Column]:
        result = operation(left, right)
        return Lowered(result, result.isNotNull())

    return rule


def register_float_variants(name: str, rule: BinaryRule[Column]) -> None:
    RULES(name, Kind.FLOAT, Kind.FLOAT)(rule)
    RULES(name, Kind.INT, Kind.FLOAT)(lambda left, right: rule(cast_float64(left), right))
    RULES(name, Kind.FLOAT, Kind.INT)(lambda left, right: rule(left, cast_float64(right)))


def either_nan(left: Column, right: Column) -> Column:
    return functions.isnan(left) | functions.isnan(right)


def register_widened_variants(name: str, rule: Binary) -> None:
    def exact(value: Column) -> Column:
        return value.isNull() | representable(value)

    RULES(name, Kind.INT, Kind.FLOAT)(
        lambda left, right: Lowered(rule(cast_float64(left), right), exact(left))
    )
    RULES(name, Kind.FLOAT, Kind.INT)(
        lambda left, right: Lowered(rule(left, cast_float64(right)), exact(right))
    )


def register_comparison(name: str, compare: Binary) -> None:
    def ordered(left: Column, right: Column) -> Column:
        return compare(left, right) & ~either_nan(left, right)

    RULES(name, Kind.INT, Kind.INT)(compare)
    RULES(name, Kind.FLOAT, Kind.FLOAT)(ordered)
    register_widened_variants(name, ordered)


def nonzero(divisor: Column, undefined: Column) -> Column:
    return when((divisor == 0) | undefined, functions.lit(1), divisor)


def floor(value: Column) -> Column:
    rounded = functions.rint(value)
    return when(rounded > value, rounded - 1.0, rounded)


PRIMITIVES = FloatPrimitives(
    when, floor, functions.signum, lambda left, right: (left % right, None)
)


def float_floordiv(left: Column, right: Column) -> Lowered[Column]:
    quotient, _, exact = float_divmod(PRIMITIVES, left, nonzero(right, functions.lit(False)))
    return Lowered(quotient, exact)


def float_mod(left: Column, right: Column) -> Lowered[Column]:
    _, remainder, exact = float_divmod(PRIMITIVES, left, nonzero(right, functions.lit(False)))
    return Lowered(remainder, exact)


def int_divmod(left: Column, right: Column) -> tuple[Column, Column]:
    divisor = nonzero(right, (left == INT64_MIN) & (right == -1))
    remainder = left % divisor
    adjust = (remainder != 0) & ((remainder < 0) != (right < 0))
    shift = when(adjust, functions.lit(1), functions.lit(0))
    quotient = functions.call_function("div", left, divisor) - shift
    modulo = when(adjust, remainder + divisor, remainder)
    return quotient, modulo


def infinite(value: Column) -> Column:
    return functions.abs(value) == INFINITY


for _name, _operation in ARITHMETIC.items():
    RULES(_name, Kind.INT, Kind.INT)(overflow_checked(CHECKED[_name]))
    register_float_variants(_name, _operation)

for _name, _compare in COMPARISONS.items():
    register_comparison(_name, _compare)

register_float_variants("__truediv__", functions.try_divide)
register_float_variants("__floordiv__", float_floordiv)
register_float_variants("__mod__", float_mod)

RULES("__eq__", Kind.INT, Kind.INT)(Column.eqNullSafe)
RULES("__mod__", Kind.INT, Kind.INT)(lambda left, right: int_divmod(left, right)[1])
RULES("__neg__", Kind.FLOAT)(Column.__neg__)
RULES("abs", Kind.FLOAT)(functions.abs)
RULES("math.isnan", Kind.FLOAT)(functions.isnan)
RULES("math.isinf", Kind.FLOAT)(infinite)


@RULES("__truediv__", Kind.INT, Kind.INT)
def int_truediv(left: Column, right: Column) -> Lowered[Column]:
    return Lowered(functions.try_divide(left, right), representable(left) & representable(right))


@RULES("__floordiv__", Kind.INT, Kind.INT)
def int_floordiv(left: Column, right: Column) -> Lowered[Column]:
    return Lowered(int_divmod(left, right)[0], ~((left == INT64_MIN) & (right == -1)))


@RULES("__ne__", Kind.INT, Kind.INT)
def int_ne(left: Column, right: Column) -> Column:
    return ~left.eqNullSafe(right)


@RULES("__eq__", Kind.FLOAT, Kind.FLOAT)
def float_eq(left: Column, right: Column) -> Column:
    return left.eqNullSafe(right) & ~either_nan(left, right)


@RULES("__ne__", Kind.FLOAT, Kind.FLOAT)
def float_ne(left: Column, right: Column) -> Column:
    return ~left.eqNullSafe(right) | either_nan(left, right)


register_widened_variants("__eq__", float_eq)
register_widened_variants("__ne__", float_ne)


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
