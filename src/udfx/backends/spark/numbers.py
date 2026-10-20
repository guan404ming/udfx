from __future__ import annotations

from collections.abc import Callable

from pyspark.sql import Column, functions

from udfx.backends.base import BinaryRule, Lowered
from udfx.backends.spark.rules import RULES, when
from udfx.semantics.literals import (
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

CHECKED: dict[str, Callable[[Column, Column], Column]] = {
    "__add__": functions.try_add,
    "__sub__": functions.try_subtract,
    "__mul__": functions.try_multiply,
}

INFINITY = float("inf")
JAVA_FIXED_MIN = 1e-3
JAVA_FIXED_MAX = 1e7


def representable(value: Column) -> Column:
    return value.between(-FLOAT_INT_MAX, FLOAT_INT_MAX)


def checked(operation: Callable[[Column, Column], Column]) -> BinaryRule[Column]:
    def rule(left: Column, right: Column) -> Lowered[Column]:
        result = operation(left, right)
        return Lowered(result, result.isNotNull())

    return rule


def floating(name: str, rule: BinaryRule[Column]) -> None:
    RULES(name, Kind.FLOAT, Kind.FLOAT)(rule)
    RULES(name, Kind.INT, Kind.FLOAT)(lambda left, right: rule(left.cast("double"), right))
    RULES(name, Kind.FLOAT, Kind.INT)(lambda left, right: rule(left, right.cast("double")))


def either_nan(left: Column, right: Column) -> Column:
    return functions.isnan(left) | functions.isnan(right)


def widened(name: str, rule: Binary) -> None:
    def exact(value: Column) -> Column:
        return value.isNull() | representable(value)

    RULES(name, Kind.INT, Kind.FLOAT)(
        lambda left, right: Lowered(rule(left.cast("double"), right), exact(left))
    )
    RULES(name, Kind.FLOAT, Kind.INT)(
        lambda left, right: Lowered(rule(left, right.cast("double")), exact(right))
    )


def comparison(name: str, compare: Binary) -> None:
    def ordered(left: Column, right: Column) -> Column:
        return compare(left, right) & ~either_nan(left, right)

    RULES(name, Kind.INT, Kind.INT)(compare)
    RULES(name, Kind.FLOAT, Kind.FLOAT)(ordered)
    widened(name, ordered)


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


TEXT = TextPrimitives(
    matches=lambda text, pattern: text.rlike(pattern),
    strip=lambda text, chars: functions.btrim(text, functions.lit(chars)),
    replace=lambda text, old, new: functions.replace(text, functions.lit(old), functions.lit(new)),
    byte_length=functions.octet_length,
    char_length=functions.length,
    not_null=Column.isNotNull,
    try_int=lambda text: text.try_cast("bigint"),
    try_float=lambda text: text.try_cast("double"),
)


for _name, _operation in ARITHMETIC.items():
    RULES(_name, Kind.INT, Kind.INT)(checked(CHECKED[_name]))
    floating(_name, _operation)

for _name, _compare in COMPARISONS.items():
    comparison(_name, _compare)

floating("__truediv__", functions.try_divide)
floating("__floordiv__", float_floordiv)
floating("__mod__", float_mod)

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


widened("__eq__", float_eq)
widened("__ne__", float_ne)


@RULES("__neg__", Kind.INT)
def int_neg(value: Column) -> Lowered[Column]:
    return Lowered(functions.try_subtract(functions.lit(0), value), value != INT64_MIN)


@RULES("abs", Kind.INT)
def int_abs(value: Column) -> Lowered[Column]:
    return Lowered(
        functions.abs(functions.greatest(value, functions.lit(-INT64_MAX))), value != INT64_MIN
    )


@RULES("int", Kind.FLOAT)
def float_int(value: Column) -> Lowered[Column]:
    in_range = (value >= float(INT64_MIN)) & (value < float(INT64_MAX))
    return Lowered(value.try_cast("bigint"), in_range | functions.isnan(value) | infinite(value))


@RULES("float", Kind.INT)
def int_float(value: Column) -> Column:
    return value.cast("double")


@RULES("float", Kind.FLOAT)
def float_float(value: Column) -> Column:
    return value


def shown(text: Column) -> Column:
    return functions.coalesce(text, functions.lit("None"))


@RULES("str", Kind.INT)
def int_str(value: Column) -> Column:
    return shown(value.cast("string"))


@RULES("str", Kind.FLOAT)
def float_str(value: Column) -> Lowered[Column]:
    size = functions.abs(value)
    fixed = (size >= JAVA_FIXED_MIN) & (size < JAVA_FIXED_MAX)
    return Lowered(shown(value.cast("string")), value.isNull() | (value == 0.0) | fixed)


@RULES("str", Kind.BOOL)
def bool_str(value: Column) -> Column:
    named = when(value, functions.lit("True"), functions.lit("False"))
    return when(value.isNull(), functions.lit("None"), named)


RULES("str", Kind.STR)(shown)


@RULES("int_parses", Kind.STR)
def str_int_parses(text: Column) -> Lowered[Column]:
    return Lowered(*int_parses(TEXT, text))


@RULES("int", Kind.STR)
def str_int(text: Column) -> Lowered[Column]:
    return Lowered(*parse_int(TEXT, text))


@RULES("float_parses", Kind.STR)
def str_float_parses(text: Column) -> Lowered[Column]:
    return Lowered(*float_parses(TEXT, text))


@RULES("float", Kind.STR)
def str_float(text: Column) -> Lowered[Column]:
    return Lowered(*parse_float(TEXT, text))


def log10(value: Column) -> Lowered[Column]:
    return Lowered(functions.log10(value), functions.lit(False))


RULES("math.log10", Kind.FLOAT)(log10)


@RULES("math.log10", Kind.INT)
def int_log10(value: Column) -> Lowered[Column]:
    return log10(value.cast("double"))
