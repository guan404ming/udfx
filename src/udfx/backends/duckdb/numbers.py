from __future__ import annotations

import operator

import duckdb
from duckdb import Expression, sqltypes

from udfx.backends.duckdb.rules import (
    NAN,
    NULL,
    RULES,
    constant,
    function,
    null_safe_equal,
    when,
)
from udfx.backends.registration import (
    BinaryRule,
    Lowered,
    register_comparison,
    register_equality,
    register_float_variants,
    register_int_truediv,
)
from udfx.frontend.core.sets import empty_union
from udfx.semantics.facts import bounded_divisor
from udfx.semantics.numbers import (
    ARITHMETIC,
    COMPARISONS,
    FLOAT_INT_MAX,
    Binary,
    FloatPrimitives,
    IntegerPrimitives,
    OrderPrimitives,
    division_overflows,
    float_divmod,
    int_divmod,
)
from udfx.types import INT64_MAX, INT64_MIN, Kind

INT64_LIMIT = 2.0**63


def cast_int128(value: Expression) -> Expression:
    return value.cast(sqltypes.HUGEINT)


def checked_int64(value: Expression) -> Lowered[Expression]:
    fits = value.between(constant(INT64_MIN), constant(INT64_MAX))
    clamped = function(
        "least", function("greatest", value, constant(INT64_MIN)), constant(INT64_MAX)
    )
    return Lowered(clamped.cast(sqltypes.BIGINT), overflow=~fits)


def within_exact_float_range(value: Expression) -> Expression:
    return value.between(constant(-FLOAT_INT_MAX), constant(FLOAT_INT_MAX))


def cast_float64(value: Expression) -> Expression:
    return value.cast(sqltypes.DOUBLE)


def overflow_checked(operation: Binary) -> BinaryRule[Expression]:
    def rule(left: Expression, right: Expression) -> Lowered[Expression]:
        return checked_int64(operation(cast_int128(left), cast_int128(right)))

    return rule


def is_nan(value: Expression) -> Expression:
    return function("isnan", value)


def either_nan(left: Expression, right: Expression) -> Expression:
    return duckdb.CoalesceOperator(is_nan(left), constant(False)) | duckdb.CoalesceOperator(
        is_nan(right), constant(False)
    )


def try_truncate_int64(value: Expression) -> Expression:
    fits = (value >= constant(-INT64_LIMIT)) & (value < constant(INT64_LIMIT))
    return when(fits, function("trunc", value).cast(sqltypes.BIGINT), NULL)


def safe_divisor(left: Expression, right: Expression) -> Expression:
    return when(division_overflows(constant, left, right), constant(1), right)


def truncated_remainder(left: Expression, right: Expression) -> tuple[Expression, Expression]:
    whole_left, whole_right = try_truncate_int64(left), try_truncate_int64(right)
    truncated = whole_left % safe_divisor(whole_left, whole_right)
    undefined = ~function("isfinite", left) | is_nan(right)
    small = function("abs", left) < function("abs", right)
    exact = (cast_float64(whole_left) == left) & (cast_float64(whole_right) == right)
    expr = when(undefined, NAN, when(small, left, cast_float64(truncated)))
    return expr, undefined | small | duckdb.CoalesceOperator(exact, constant(False))


PRIMITIVES = FloatPrimitives(
    when,
    lambda value: function("floor", value),
    lambda value: function("sign", value),
    truncated_remainder,
)


def float_floordiv(left: Expression, right: Expression) -> Lowered[Expression]:
    quotient, _, exact = float_divmod(PRIMITIVES, left, right)
    return Lowered(quotient, exact)


def float_mod(left: Expression, right: Expression) -> Lowered[Expression]:
    _, remainder, exact = float_divmod(PRIMITIVES, left, right)
    return Lowered(remainder, exact)


INTEGERS = IntegerPrimitives(constant, when, operator.floordiv, operator.mod)


ORDER = OrderPrimitives(either_nan, Expression.isnull, within_exact_float_range)


for _name, _operation in ARITHMETIC.items():
    RULES(_name, Kind.INT, Kind.INT)(overflow_checked(_operation))
    register_float_variants(RULES, _name, _operation, cast_float64)

for _name, _compare in COMPARISONS.items():
    register_comparison(RULES, _name, _compare, cast_float64, ORDER)

register_float_variants(RULES, "__truediv__", operator.truediv, cast_float64)
register_float_variants(RULES, "__floordiv__", float_floordiv, cast_float64)
register_float_variants(RULES, "__mod__", float_mod, cast_float64)
register_equality(RULES, null_safe_equal, cast_float64, ORDER)

RULES("__neg__", Kind.FLOAT)(Expression.__neg__)
RULES("abs", Kind.FLOAT)(lambda value: function("abs", value))
RULES("math.isnan", Kind.FLOAT)(is_nan)


@RULES("math.isinf", Kind.FLOAT)
def float_isinf(value: Expression) -> Expression:
    return function("isinf", value)


register_int_truediv(RULES, lambda left, right: cast_float64(left) / cast_float64(right), ORDER)


@RULES("__floordiv__", Kind.INT, Kind.INT, facts={"bounded": bounded_divisor})
def int_floordiv(left: Expression, right: Expression, bounded: bool = False) -> Lowered[Expression]:
    quotient, _, exact = int_divmod(INTEGERS, left, right, bounded)
    return Lowered(quotient, overflow=None if exact is None else ~exact)


@RULES("__mod__", Kind.INT, Kind.INT)
def int_mod(left: Expression, right: Expression) -> Expression:
    return int_divmod(INTEGERS, left, right)[1]


@RULES("__neg__", Kind.INT)
def int_neg(value: Expression) -> Lowered[Expression]:
    return checked_int64(-cast_int128(value))


@RULES("abs", Kind.INT)
def int_abs(value: Expression) -> Lowered[Expression]:
    return checked_int64(function("abs", cast_int128(value)))


def log10(value: Expression) -> Expression:
    return function("log10", when(value > constant(0.0), value, constant(1.0)))


RULES("math.log10", Kind.FLOAT)(log10)


@RULES("math.log10", Kind.INT)
def int_log10(value: Expression) -> Expression:
    return log10(cast_float64(value))


for name in ("__eq__", "__ne__"):
    RULES.rewrite(name, Kind.INT, Kind.INT)(empty_union)
