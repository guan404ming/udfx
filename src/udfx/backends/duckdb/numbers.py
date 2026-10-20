from __future__ import annotations

import sys
from collections.abc import Callable

import duckdb
from duckdb import Expression, sqltypes

from udfx.backends.base import Lowered
from udfx.backends.duckdb.rules import (
    NAN,
    NULL,
    RULES,
    constant,
    function,
    length,
    nullable_equal,
    when,
)
from udfx.semantics.literals import ASCII_SPACE, FLOAT_LITERAL, INT_LITERAL
from udfx.semantics.numbers import ARITHMETIC, COMPARISONS, FLOAT_INT_MAX, Binary
from udfx.types import INT64_MAX, INT64_MIN, Kind

Rule = Callable[[Expression, Expression], Expression | Lowered[Expression]]

INT64_LIMIT = 2.0**63


def wide(a: Expression) -> Expression:
    return a.cast(sqltypes.HUGEINT)


def narrow(value: Expression) -> Lowered[Expression]:
    fits = value.between(constant(INT64_MIN), constant(INT64_MAX))
    return Lowered(when(fits, value, constant(0)).cast(sqltypes.BIGINT), fits)


def representable(a: Expression) -> Expression:
    return a.between(constant(-FLOAT_INT_MAX), constant(FLOAT_INT_MAX))


def floating(a: Expression) -> Expression:
    return a.cast(sqltypes.DOUBLE)


def checked(operation: Binary) -> Rule:
    def rule(a: Expression, b: Expression) -> Lowered[Expression]:
        return narrow(operation(wide(a), wide(b)))

    return rule


def mixed(name: str, rule: Rule) -> None:
    RULES(name, Kind.FLOAT, Kind.FLOAT)(rule)
    RULES(name, Kind.INT, Kind.FLOAT)(lambda a, b: rule(floating(a), b))
    RULES(name, Kind.FLOAT, Kind.INT)(lambda a, b: rule(a, floating(b)))


def is_nan(a: Expression) -> Expression:
    return function("isnan", a)


def either_nan(a: Expression, b: Expression) -> Expression:
    return duckdb.CoalesceOperator(is_nan(a), constant(False)) | duckdb.CoalesceOperator(
        is_nan(b), constant(False)
    )


def comparison(name: str, compare: Binary) -> None:
    def ordered(a: Expression, b: Expression) -> Expression:
        return compare(a, b) & ~is_nan(a) & ~is_nan(b)

    RULES(name, Kind.INT, Kind.INT)(compare)
    RULES(name, Kind.FLOAT, Kind.FLOAT)(ordered)
    RULES(name, Kind.INT, Kind.FLOAT)(
        lambda a, b: Lowered(ordered(floating(a), b), representable(a))
    )
    RULES(name, Kind.FLOAT, Kind.INT)(
        lambda a, b: Lowered(ordered(a, floating(b)), representable(b))
    )


def divide(a: Expression, b: Expression) -> Expression:
    return a / b


def signed_zero(b: Expression) -> Expression:
    return when(b < constant(0.0), constant(-0.0), constant(0.0))


def whole(a: Expression) -> Expression:
    fits = (a >= constant(-INT64_LIMIT)) & (a < constant(INT64_LIMIT))
    return when(fits, function("trunc", a).cast(sqltypes.BIGINT), NULL)


def safe_divisor(a: Expression, b: Expression) -> Expression:
    return when((a == constant(INT64_MIN)) & (b == constant(-1)), constant(1), b)


def truncated_remainder(a: Expression, b: Expression) -> tuple[Expression, Expression]:
    whole_a, whole_b = whole(a), whole(b)
    truncated = whole_a % safe_divisor(whole_a, whole_b)
    undefined = ~function("isfinite", a) | is_nan(b)
    small = function("abs", a) < function("abs", b)
    exact = (floating(whole_a) == a) & (floating(whole_b) == b)
    expr = when(undefined, NAN, when(small, a, floating(truncated)))
    return expr, undefined | small | duckdb.CoalesceOperator(exact, constant(False))


def divmod_(a: Expression, b: Expression) -> tuple[Expression, Expression, Expression]:
    remainder, exact = truncated_remainder(a, b)
    quotient = divide(a - remainder, b)
    adjust = (remainder != constant(0.0)) & ((b < constant(0.0)) != (remainder < constant(0.0)))
    remainder = when(
        remainder == constant(0.0),
        signed_zero(b),
        when(adjust, remainder + b, remainder),
    )
    quotient = when(adjust, quotient - constant(1.0), quotient)
    floored = function("floor", quotient)
    snapped = when(quotient - floored > constant(0.5), floored + constant(1.0), floored)
    quotient = when(quotient != constant(0.0), snapped, divide(a, b) * constant(0.0))
    return quotient, remainder, exact


def float_floordiv(a: Expression, b: Expression) -> Lowered[Expression]:
    quotient, _, exact = divmod_(a, b)
    return Lowered(quotient, exact)


def float_mod(a: Expression, b: Expression) -> Lowered[Expression]:
    _, remainder, exact = divmod_(a, b)
    return Lowered(remainder, exact)


def int_divmod(a: Expression, b: Expression) -> tuple[Expression, Expression, Expression]:
    divisor = safe_divisor(a, b)
    quotient, remainder = a // divisor, a % divisor
    adjust = (remainder != constant(0)) & ((remainder < constant(0)) != (b < constant(0)))
    exact = ~((a == constant(INT64_MIN)) & (b == constant(-1)))
    return (
        when(adjust, quotient - constant(1), quotient),
        when(adjust, remainder + b, remainder),
        exact,
    )


def is_ascii(a: Expression) -> Expression:
    return function("strlen", a) == length(a)


def cleaned(a: Expression) -> Expression:
    trimmed = function("trim", a, constant(ASCII_SPACE))
    return function("replace", trimmed, constant("_"), constant(""))


def matches(a: Expression, pattern: str) -> Expression:
    return function("regexp_matches", a, constant(pattern))


for _name, _operation in ARITHMETIC.items():
    RULES(_name, Kind.INT, Kind.INT)(checked(_operation))
    mixed(_name, _operation)

for _name, _compare in COMPARISONS.items():
    comparison(_name, _compare)

mixed("__truediv__", divide)
mixed("__floordiv__", float_floordiv)
mixed("__mod__", float_mod)

RULES("__eq__", Kind.INT, Kind.INT)(nullable_equal)
RULES("__neg__", Kind.FLOAT)(Expression.__neg__)
RULES("abs", Kind.FLOAT)(lambda a: function("abs", a))
RULES("math.isnan", Kind.FLOAT)(is_nan)
RULES("math.isinf", Kind.FLOAT)(lambda a: function("isinf", a))


@RULES("__ne__", Kind.INT, Kind.INT)
def int_ne(a: Expression, b: Expression) -> Expression:
    return ~nullable_equal(a, b)


@RULES("__truediv__", Kind.INT, Kind.INT)
def int_truediv(a: Expression, b: Expression) -> Lowered[Expression]:
    return Lowered(divide(floating(a), floating(b)), representable(a) & representable(b))


@RULES("__floordiv__", Kind.INT, Kind.INT)
def int_floordiv(a: Expression, b: Expression) -> Lowered[Expression]:
    quotient, _, exact = int_divmod(a, b)
    return Lowered(quotient, exact)


@RULES("__mod__", Kind.INT, Kind.INT)
def int_mod(a: Expression, b: Expression) -> Lowered[Expression]:
    _, remainder, exact = int_divmod(a, b)
    return Lowered(remainder, exact)


@RULES("__eq__", Kind.FLOAT, Kind.FLOAT)
def float_eq(a: Expression, b: Expression) -> Expression:
    return nullable_equal(a, b) & ~either_nan(a, b)


@RULES("__ne__", Kind.FLOAT, Kind.FLOAT)
def float_ne(a: Expression, b: Expression) -> Expression:
    return ~nullable_equal(a, b) | either_nan(a, b)


@RULES("__neg__", Kind.INT)
def int_neg(a: Expression) -> Lowered[Expression]:
    return narrow(-wide(a))


@RULES("abs", Kind.INT)
def int_abs(a: Expression) -> Lowered[Expression]:
    return narrow(function("abs", wide(a)))


@RULES("int", Kind.FLOAT)
def float_int(a: Expression) -> Lowered[Expression]:
    value = whole(a)
    return Lowered(value, value.isnotnull() | ~function("isfinite", a))


RULES("float", Kind.INT)(floating)


@RULES("float", Kind.FLOAT)
def float_float(value: Expression) -> Expression:
    return value


@RULES("str", Kind.INT)
def int_str(a: Expression) -> Expression:
    return a.cast(sqltypes.VARCHAR)


@RULES("int_parses", Kind.STR)
def str_int_parses(a: Expression) -> Lowered[Expression]:
    limit = sys.get_int_max_str_digits() or sys.maxsize
    return Lowered(matches(a, INT_LITERAL), is_ascii(a) & (length(a) <= constant(limit)))


@RULES("int", Kind.STR)
def str_int(a: Expression) -> Lowered[Expression]:
    digits = cleaned(a)
    parsable = matches(a, INT_LITERAL)
    parsed = narrow(when(parsable & (length(digits) <= constant(38)), wide(digits), NULL))
    exact = ~parsable | duckdb.CoalesceOperator(parsed.exact, constant(False))
    return Lowered(parsed.expr, is_ascii(a) & exact)


@RULES("float_parses", Kind.STR)
def str_float_parses(a: Expression) -> Lowered[Expression]:
    return Lowered(matches(a, FLOAT_LITERAL), is_ascii(a))


@RULES("float", Kind.STR)
def str_float(a: Expression) -> Lowered[Expression]:
    parsed = when(matches(a, FLOAT_LITERAL), floating(cleaned(a)), NULL)
    return Lowered(parsed, is_ascii(a))


def log10(a: Expression) -> Expression:
    return function("log10", when(a > constant(0.0), a, constant(1.0)))


RULES("math.log10", Kind.FLOAT)(log10)


@RULES("math.log10", Kind.INT)
def int_log10(a: Expression) -> Expression:
    return log10(floating(a))
