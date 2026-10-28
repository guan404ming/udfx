from __future__ import annotations

import operator
from collections.abc import Callable

from pyspark.sql import Column, functions

from udfx.backends.registration import (
    Lowered,
    register_comparison,
    register_equality,
    register_float_variants,
    register_int_truediv,
)
from udfx.backends.spark.rules import INTEGER_FITS, RULES, when
from udfx.frontend.core.sets import disjoint_ratio, empty_union
from udfx.semantics.arithmetic import IntegerRegionPrimitives
from udfx.semantics.facts import bounded_divisor
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
WIDE_INTEGERS = IntegerRegionPrimitives(
    maximum=10**38 - 1,
    size_limit=2**31 - 1,
    widen=lambda value: value.cast("decimal(38,0)"),
    narrow=lambda value: value.try_cast("long"),
    not_null=Column.isNotNull,
    operations=CHECKED,
)

INFINITY = float("inf")


def cast_float64(value: Column) -> Column:
    return value.cast("double")


def within_exact_float_range(value: Column) -> Column:
    return value.between(-FLOAT_INT_MAX, FLOAT_INT_MAX)


def overflow_checked(
    operation: Callable[[Column, Column], Column],
) -> Callable[..., Lowered[Column]]:
    def rule(left: Column, right: Column, *, bounded: bool = False) -> Lowered[Column]:
        result = operation(left, right)
        overflow = None if bounded else result.isNull() & left.isNotNull() & right.isNotNull()
        return Lowered(result, overflow=overflow)

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
    functions.lit,
    when,
    lambda left, right: functions.call_function("div", left, right),
    operator.mod,
)


def float_isinf(value: Column) -> Column:
    return functions.abs(value) == INFINITY


ORDER = OrderPrimitives(either_nan, Column.isNull, within_exact_float_range)


for _name, _operation in ARITHMETIC.items():
    RULES(_name, Kind.INT, Kind.INT, facts={"bounded": INTEGER_FITS})(
        overflow_checked(CHECKED[_name])
    )
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


register_int_truediv(RULES, functions.try_divide, ORDER)


@RULES("__floordiv__", Kind.INT, Kind.INT, facts={"bounded": bounded_divisor})
def int_floordiv(left: Column, right: Column, bounded: bool = False) -> Lowered[Column]:
    quotient, _, exact = int_divmod(INTEGERS, left, right, bounded)
    return Lowered(quotient, overflow=None if exact is None else ~exact)


@RULES("__neg__", Kind.INT)
def int_neg(value: Column) -> Lowered[Column]:
    return Lowered(functions.try_subtract(functions.lit(0), value), overflow=value == INT64_MIN)


@RULES("abs", Kind.INT)
def int_abs(value: Column) -> Lowered[Column]:
    return Lowered(
        functions.abs(functions.greatest(value, functions.lit(-INT64_MAX))),
        overflow=value == INT64_MIN,
    )


def log10(value: Column) -> Lowered[Column]:
    return Lowered(functions.log10(value), functions.lit(False))


RULES("math.log10", Kind.FLOAT)(log10)


@RULES("math.log10", Kind.INT)
def int_log10(value: Column) -> Lowered[Column]:
    return log10(cast_float64(value))


for name in ("__eq__", "__ne__"):
    RULES.rewrite(name, Kind.INT, Kind.INT)(empty_union)

for kind in (Kind.INT, Kind.FLOAT):
    RULES.rewrite("__truediv__", kind, Kind.INT)(disjoint_ratio)
