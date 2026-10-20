from __future__ import annotations

import operator

import duckdb
from duckdb import Expression, sqltypes

from udfx.backends.base import BinaryRule, Lowered
from udfx.backends.duckdb.rules import (
    NAN,
    NULL,
    RULES,
    constant,
    function,
    null_safe_equal,
    when,
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

INT64_LIMIT = 2.0**63


def cast_int128(value: Expression) -> Expression:
    return value.cast(sqltypes.HUGEINT)


def checked_int64(value: Expression) -> Lowered[Expression]:
    fits = value.between(constant(INT64_MIN), constant(INT64_MAX))
    return Lowered(when(fits, value, constant(0)).cast(sqltypes.BIGINT), fits)


def within_exact_float_range(value: Expression) -> Expression:
    return value.between(constant(-FLOAT_INT_MAX), constant(FLOAT_INT_MAX))


def cast_float64(value: Expression) -> Expression:
    return value.cast(sqltypes.DOUBLE)


def overflow_checked(operation: Binary) -> BinaryRule[Expression]:
    def rule(left: Expression, right: Expression) -> Lowered[Expression]:
        return checked_int64(operation(cast_int128(left), cast_int128(right)))

    return rule


def register_float_variants(name: str, rule: BinaryRule[Expression]) -> None:
    RULES(name, Kind.FLOAT, Kind.FLOAT)(rule)
    RULES(name, Kind.INT, Kind.FLOAT)(lambda left, right: rule(cast_float64(left), right))
    RULES(name, Kind.FLOAT, Kind.INT)(lambda left, right: rule(left, cast_float64(right)))


def is_nan(value: Expression) -> Expression:
    return function("isnan", value)


def either_nan(left: Expression, right: Expression) -> Expression:
    return duckdb.CoalesceOperator(is_nan(left), constant(False)) | duckdb.CoalesceOperator(
        is_nan(right), constant(False)
    )


def register_widened_variants(name: str, rule: Binary) -> None:
    def exact(value: Expression) -> Expression:
        return value.isnull() | within_exact_float_range(value)

    RULES(name, Kind.INT, Kind.FLOAT)(
        lambda left, right: Lowered(rule(cast_float64(left), right), exact(left))
    )
    RULES(name, Kind.FLOAT, Kind.INT)(
        lambda left, right: Lowered(rule(left, cast_float64(right)), exact(right))
    )


def register_comparison(name: str, compare: Binary) -> None:
    def ordered(left: Expression, right: Expression) -> Expression:
        return compare(left, right) & ~is_nan(left) & ~is_nan(right)

    RULES(name, Kind.INT, Kind.INT)(compare)
    RULES(name, Kind.FLOAT, Kind.FLOAT)(ordered)
    register_widened_variants(name, ordered)


def try_truncate_int64(value: Expression) -> Expression:
    fits = (value >= constant(-INT64_LIMIT)) & (value < constant(INT64_LIMIT))
    return when(fits, function("trunc", value).cast(sqltypes.BIGINT), NULL)


def safe_divisor(left: Expression, right: Expression) -> Expression:
    return when((left == constant(INT64_MIN)) & (right == constant(-1)), constant(1), right)


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


def int_divmod(left: Expression, right: Expression) -> tuple[Expression, Expression, Expression]:
    divisor = safe_divisor(left, right)
    quotient, remainder = left // divisor, left % divisor
    adjust = (remainder != constant(0)) & ((remainder < constant(0)) != (right < constant(0)))
    exact = ~((left == constant(INT64_MIN)) & (right == constant(-1)))
    return (
        when(adjust, quotient - constant(1), quotient),
        when(adjust, remainder + right, remainder),
        exact,
    )


for _name, _operation in ARITHMETIC.items():
    RULES(_name, Kind.INT, Kind.INT)(overflow_checked(_operation))
    register_float_variants(_name, _operation)

for _name, _compare in COMPARISONS.items():
    register_comparison(_name, _compare)

register_float_variants("__truediv__", operator.truediv)
register_float_variants("__floordiv__", float_floordiv)
register_float_variants("__mod__", float_mod)

RULES("__eq__", Kind.INT, Kind.INT)(null_safe_equal)
RULES("__neg__", Kind.FLOAT)(Expression.__neg__)
RULES("abs", Kind.FLOAT)(lambda value: function("abs", value))
RULES("math.isnan", Kind.FLOAT)(is_nan)


@RULES("math.isinf", Kind.FLOAT)
def float_isinf(value: Expression) -> Expression:
    return function("isinf", value)


@RULES("__ne__", Kind.INT, Kind.INT)
def int_ne(left: Expression, right: Expression) -> Expression:
    return ~null_safe_equal(left, right)


@RULES("__truediv__", Kind.INT, Kind.INT)
def int_truediv(left: Expression, right: Expression) -> Lowered[Expression]:
    return Lowered(
        cast_float64(left) / cast_float64(right),
        within_exact_float_range(left) & within_exact_float_range(right),
    )


@RULES("__floordiv__", Kind.INT, Kind.INT)
def int_floordiv(left: Expression, right: Expression) -> Lowered[Expression]:
    quotient, _, exact = int_divmod(left, right)
    return Lowered(quotient, exact)


@RULES("__mod__", Kind.INT, Kind.INT)
def int_mod(left: Expression, right: Expression) -> Lowered[Expression]:
    _, remainder, exact = int_divmod(left, right)
    return Lowered(remainder, exact)


@RULES("__eq__", Kind.FLOAT, Kind.FLOAT)
def float_eq(left: Expression, right: Expression) -> Expression:
    return null_safe_equal(left, right) & ~either_nan(left, right)


@RULES("__ne__", Kind.FLOAT, Kind.FLOAT)
def float_ne(left: Expression, right: Expression) -> Expression:
    return ~null_safe_equal(left, right) | either_nan(left, right)


register_widened_variants("__eq__", float_eq)
register_widened_variants("__ne__", float_ne)


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
