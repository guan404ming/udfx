from __future__ import annotations

from collections.abc import Callable

from pyspark.sql import Column, functions

from udfx.backends.base import Lowered
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

Rule = Callable[[Column, Column], Column | Lowered[Column]]

CHECKED: dict[str, Callable[[Column, Column], Column]] = {
    "__add__": functions.try_add,
    "__sub__": functions.try_subtract,
    "__mul__": functions.try_multiply,
}

INFINITY = float("inf")


def representable(a: Column) -> Column:
    return a.between(-FLOAT_INT_MAX, FLOAT_INT_MAX)


def checked(operation: Callable[[Column, Column], Column]) -> Rule:
    def rule(a: Column, b: Column) -> Lowered[Column]:
        result = operation(a, b)
        return Lowered(result, result.isNotNull())

    return rule


def floating(name: str, rule: Rule) -> None:
    RULES(name, Kind.FLOAT, Kind.FLOAT)(rule)
    RULES(name, Kind.INT, Kind.FLOAT)(lambda a, b: rule(a.cast("double"), b))
    RULES(name, Kind.FLOAT, Kind.INT)(lambda a, b: rule(a, b.cast("double")))


def either_nan(a: Column, b: Column) -> Column:
    return functions.isnan(a) | functions.isnan(b)


def comparison(name: str, compare: Binary) -> None:
    def ordered(a: Column, b: Column) -> Column:
        return compare(a, b) & ~either_nan(a, b)

    RULES(name, Kind.INT, Kind.INT)(compare)
    RULES(name, Kind.FLOAT, Kind.FLOAT)(ordered)
    RULES(name, Kind.INT, Kind.FLOAT)(
        lambda a, b: Lowered(ordered(a.cast("double"), b), representable(a))
    )
    RULES(name, Kind.FLOAT, Kind.INT)(
        lambda a, b: Lowered(ordered(a, b.cast("double")), representable(b))
    )


def nonzero(divisor: Column, undefined: Column) -> Column:
    return when((divisor == 0) | undefined, functions.lit(1), divisor)


def floor(value: Column) -> Column:
    rounded = functions.rint(value)
    return when(rounded > value, rounded - 1.0, rounded)


PRIMITIVES = FloatPrimitives(
    when, floor, functions.signum, lambda left, right: (left % right, None)
)


def float_floordiv(a: Column, b: Column) -> Lowered[Column]:
    quotient, _, exact = float_divmod(PRIMITIVES, a, nonzero(b, functions.lit(False)))
    return Lowered(quotient, exact)


def float_mod(a: Column, b: Column) -> Lowered[Column]:
    _, remainder, exact = float_divmod(PRIMITIVES, a, nonzero(b, functions.lit(False)))
    return Lowered(remainder, exact)


def int_divmod(a: Column, b: Column) -> tuple[Column, Column]:
    divisor = nonzero(b, (a == INT64_MIN) & (b == -1))
    remainder = a % divisor
    adjust = (remainder != 0) & ((remainder < 0) != (b < 0))
    shift = when(adjust, functions.lit(1), functions.lit(0))
    quotient = functions.call_function("div", a, divisor) - shift
    modulo = when(adjust, remainder + divisor, remainder)
    return quotient, modulo


def infinite(a: Column) -> Column:
    return functions.abs(a) == INFINITY


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
RULES("__mod__", Kind.INT, Kind.INT)(lambda a, b: int_divmod(a, b)[1])
RULES("__neg__", Kind.FLOAT)(Column.__neg__)
RULES("abs", Kind.FLOAT)(functions.abs)
RULES("math.isnan", Kind.FLOAT)(functions.isnan)
RULES("math.isinf", Kind.FLOAT)(infinite)


@RULES("__truediv__", Kind.INT, Kind.INT)
def int_truediv(a: Column, b: Column) -> Lowered[Column]:
    return Lowered(functions.try_divide(a, b), representable(a) & representable(b))


@RULES("__floordiv__", Kind.INT, Kind.INT)
def int_floordiv(a: Column, b: Column) -> Lowered[Column]:
    return Lowered(int_divmod(a, b)[0], ~((a == INT64_MIN) & (b == -1)))


@RULES("__ne__", Kind.INT, Kind.INT)
def int_ne(a: Column, b: Column) -> Column:
    return ~a.eqNullSafe(b)


@RULES("__eq__", Kind.FLOAT, Kind.FLOAT)
def float_eq(a: Column, b: Column) -> Column:
    return a.eqNullSafe(b) & ~either_nan(a, b)


@RULES("__ne__", Kind.FLOAT, Kind.FLOAT)
def float_ne(a: Column, b: Column) -> Column:
    return ~a.eqNullSafe(b) | either_nan(a, b)


@RULES("__neg__", Kind.INT)
def int_neg(a: Column) -> Lowered[Column]:
    return Lowered(functions.try_subtract(functions.lit(0), a), a != INT64_MIN)


@RULES("abs", Kind.INT)
def int_abs(a: Column) -> Lowered[Column]:
    return Lowered(functions.abs(functions.greatest(a, functions.lit(-INT64_MAX))), a != INT64_MIN)


@RULES("int", Kind.FLOAT)
def float_int(a: Column) -> Lowered[Column]:
    in_range = (a >= float(INT64_MIN)) & (a < float(INT64_MAX))
    return Lowered(a.try_cast("bigint"), in_range | functions.isnan(a) | infinite(a))


@RULES("float", Kind.INT)
def int_float(a: Column) -> Column:
    return a.cast("double")


@RULES("float", Kind.FLOAT)
def float_float(value: Column) -> Column:
    return value


@RULES("str", Kind.INT)
def int_str(a: Column) -> Column:
    return a.cast("string")


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


def log10(a: Column) -> Lowered[Column]:
    return Lowered(functions.log10(a), functions.lit(False))


RULES("math.log10", Kind.FLOAT)(log10)


@RULES("math.log10", Kind.INT)
def int_log10(a: Column) -> Lowered[Column]:
    return log10(a.cast("double"))
