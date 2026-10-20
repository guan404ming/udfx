from __future__ import annotations

import operator
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
from udfx.semantics.literals import (
    FLOAT_LITERAL,
    INT_LITERAL,
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

Rule = Callable[[Expression, Expression], Expression | Lowered[Expression]]

INT64_LIMIT = 2.0**63


def wide(value: Expression) -> Expression:
    return value.cast(sqltypes.HUGEINT)


def narrow(value: Expression) -> Lowered[Expression]:
    fits = value.between(constant(INT64_MIN), constant(INT64_MAX))
    return Lowered(when(fits, value, constant(0)).cast(sqltypes.BIGINT), fits)


def representable(value: Expression) -> Expression:
    return value.between(constant(-FLOAT_INT_MAX), constant(FLOAT_INT_MAX))


def floating(value: Expression) -> Expression:
    return value.cast(sqltypes.DOUBLE)


def checked(operation: Binary) -> Rule:
    def rule(left: Expression, right: Expression) -> Lowered[Expression]:
        return narrow(operation(wide(left), wide(right)))

    return rule


def mixed(name: str, rule: Rule) -> None:
    RULES(name, Kind.FLOAT, Kind.FLOAT)(rule)
    RULES(name, Kind.INT, Kind.FLOAT)(lambda left, right: rule(floating(left), right))
    RULES(name, Kind.FLOAT, Kind.INT)(lambda left, right: rule(left, floating(right)))


def is_nan(value: Expression) -> Expression:
    return function("isnan", value)


def either_nan(left: Expression, right: Expression) -> Expression:
    return duckdb.CoalesceOperator(is_nan(left), constant(False)) | duckdb.CoalesceOperator(
        is_nan(right), constant(False)
    )


def widened(name: str, rule: Binary) -> None:
    def exact(value: Expression) -> Expression:
        return value.isnull() | representable(value)

    RULES(name, Kind.INT, Kind.FLOAT)(
        lambda left, right: Lowered(rule(floating(left), right), exact(left))
    )
    RULES(name, Kind.FLOAT, Kind.INT)(
        lambda left, right: Lowered(rule(left, floating(right)), exact(right))
    )


def comparison(name: str, compare: Binary) -> None:
    def ordered(left: Expression, right: Expression) -> Expression:
        return compare(left, right) & ~is_nan(left) & ~is_nan(right)

    RULES(name, Kind.INT, Kind.INT)(compare)
    RULES(name, Kind.FLOAT, Kind.FLOAT)(ordered)
    widened(name, ordered)


def whole(value: Expression) -> Expression:
    fits = (value >= constant(-INT64_LIMIT)) & (value < constant(INT64_LIMIT))
    return when(fits, function("trunc", value).cast(sqltypes.BIGINT), NULL)


def safe_divisor(left: Expression, right: Expression) -> Expression:
    return when((left == constant(INT64_MIN)) & (right == constant(-1)), constant(1), right)


def truncated_remainder(left: Expression, right: Expression) -> tuple[Expression, Expression]:
    whole_left, whole_right = whole(left), whole(right)
    truncated = whole_left % safe_divisor(whole_left, whole_right)
    undefined = ~function("isfinite", left) | is_nan(right)
    small = function("abs", left) < function("abs", right)
    exact = (floating(whole_left) == left) & (floating(whole_right) == right)
    expr = when(undefined, NAN, when(small, left, floating(truncated)))
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


def matches(text: Expression, pattern: str) -> Expression:
    return function("regexp_matches", text, constant(pattern))


def try_int(digits: Expression) -> Expression:
    parsable = matches(digits, INT_LITERAL) & (length(digits) <= constant(38))
    value = when(parsable, wide(digits), NULL)
    fits = value.between(constant(INT64_MIN), constant(INT64_MAX))
    return when(fits, value, NULL).cast(sqltypes.BIGINT)


def try_float(digits: Expression) -> Expression:
    return when(matches(digits, FLOAT_LITERAL), floating(digits), NULL)


TEXT = TextPrimitives(
    matches=matches,
    strip=lambda text, chars: function("trim", text, constant(chars)),
    replace=lambda text, old, new: function("replace", text, constant(old), constant(new)),
    byte_length=lambda text: function("strlen", text),
    char_length=length,
    not_null=Expression.isnotnull,
    try_int=try_int,
    try_float=try_float,
)


for _name, _operation in ARITHMETIC.items():
    RULES(_name, Kind.INT, Kind.INT)(checked(_operation))
    mixed(_name, _operation)

for _name, _compare in COMPARISONS.items():
    comparison(_name, _compare)

mixed("__truediv__", operator.truediv)
mixed("__floordiv__", float_floordiv)
mixed("__mod__", float_mod)

RULES("__eq__", Kind.INT, Kind.INT)(nullable_equal)
RULES("__neg__", Kind.FLOAT)(Expression.__neg__)
RULES("abs", Kind.FLOAT)(lambda value: function("abs", value))
RULES("math.isnan", Kind.FLOAT)(is_nan)
RULES("math.isinf", Kind.FLOAT)(lambda value: function("isinf", value))


@RULES("__ne__", Kind.INT, Kind.INT)
def int_ne(left: Expression, right: Expression) -> Expression:
    return ~nullable_equal(left, right)


@RULES("__truediv__", Kind.INT, Kind.INT)
def int_truediv(left: Expression, right: Expression) -> Lowered[Expression]:
    return Lowered(floating(left) / floating(right), representable(left) & representable(right))


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
    return nullable_equal(left, right) & ~either_nan(left, right)


@RULES("__ne__", Kind.FLOAT, Kind.FLOAT)
def float_ne(left: Expression, right: Expression) -> Expression:
    return ~nullable_equal(left, right) | either_nan(left, right)


widened("__eq__", float_eq)
widened("__ne__", float_ne)


@RULES("__neg__", Kind.INT)
def int_neg(value: Expression) -> Lowered[Expression]:
    return narrow(-wide(value))


@RULES("abs", Kind.INT)
def int_abs(value: Expression) -> Lowered[Expression]:
    return narrow(function("abs", wide(value)))


@RULES("int", Kind.FLOAT)
def float_int(value: Expression) -> Lowered[Expression]:
    value = whole(value)
    return Lowered(value, value.isnotnull() | ~function("isfinite", value))


RULES("float", Kind.INT)(floating)


@RULES("float", Kind.FLOAT)
def float_float(value: Expression) -> Expression:
    return value


def shown(text: Expression) -> Expression:
    return duckdb.CoalesceOperator(text, constant("None"))


@RULES("str", Kind.INT)
def int_str(value: Expression) -> Expression:
    return shown(value.cast(sqltypes.VARCHAR))


@RULES("str", Kind.FLOAT)
def float_str(value: Expression) -> Lowered[Expression]:
    return Lowered(
        shown(value.cast(sqltypes.VARCHAR)),
        ~duckdb.CoalesceOperator(is_nan(value), constant(False)),
    )


@RULES("str", Kind.BOOL)
def bool_str(value: Expression) -> Expression:
    named = when(value, constant("True"), constant("False"))
    return when(value.isnull(), constant("None"), named)


RULES("str", Kind.STR)(shown)


@RULES("int_parses", Kind.STR)
def str_int_parses(text: Expression) -> Lowered[Expression]:
    return Lowered(*int_parses(TEXT, text))


@RULES("int", Kind.STR)
def str_int(text: Expression) -> Lowered[Expression]:
    return Lowered(*parse_int(TEXT, text))


@RULES("float_parses", Kind.STR)
def str_float_parses(text: Expression) -> Lowered[Expression]:
    return Lowered(*float_parses(TEXT, text))


@RULES("float", Kind.STR)
def str_float(text: Expression) -> Lowered[Expression]:
    return Lowered(*parse_float(TEXT, text))


def log10(value: Expression) -> Expression:
    return function("log10", when(value > constant(0.0), value, constant(1.0)))


RULES("math.log10", Kind.FLOAT)(log10)


@RULES("math.log10", Kind.INT)
def int_log10(value: Expression) -> Expression:
    return log10(floating(value))
