from __future__ import annotations

import operator
from collections.abc import Callable

import pyarrow as pa
from datafusion import Expr, functions, lit

from udfx.backends.base import Lowered
from udfx.backends.datafusion.rules import RULES, length, not_distinct, when
from udfx.ops.numbers import REPR_FIXED_MAX, REPR_FIXED_MIN
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

Rule = Callable[[Expr, Expr], Expr | Lowered[Expr]]
WIDE = pa.decimal128(19, 0)
INFINITY = float("inf")


def fits_int64(wide: Expr) -> Expr:
    return wide.between(lit(INT64_MIN), lit(INT64_MAX))


def representable(value: Expr) -> Expr:
    return value.between(lit(-FLOAT_INT_MAX), lit(FLOAT_INT_MAX))


def floating(value: Expr) -> Expr:
    return value.cast(pa.float64())


def checked(operation: Binary) -> Rule:
    def rule(left: Expr, right: Expr) -> Lowered[Expr]:
        wide = operation(left.cast(WIDE), right.cast(WIDE))
        return Lowered(operation(left, right), fits_int64(wide))

    return rule


def mixed(name: str, rule: Rule) -> None:
    RULES(name, Kind.FLOAT, Kind.FLOAT)(rule)
    RULES(name, Kind.INT, Kind.FLOAT)(lambda left, right: rule(floating(left), right))
    RULES(name, Kind.FLOAT, Kind.INT)(lambda left, right: rule(left, floating(right)))


def is_nan(value: Expr) -> Expr:
    return functions.coalesce(functions.isnan(value), lit(False))


def widened(name: str, rule: Binary) -> None:
    def exact(value: Expr) -> Expr:
        return value.is_null() | representable(value)

    RULES(name, Kind.INT, Kind.FLOAT)(
        lambda left, right: Lowered(rule(floating(left), right), exact(left))
    )
    RULES(name, Kind.FLOAT, Kind.INT)(
        lambda left, right: Lowered(rule(left, floating(right)), exact(right))
    )


def comparison(name: str, compare: Binary) -> None:
    def ordered(left: Expr, right: Expr) -> Expr:
        return compare(left, right) & ~is_nan(left) & ~is_nan(right)

    RULES(name, Kind.INT, Kind.INT)(compare)
    RULES(name, Kind.FLOAT, Kind.FLOAT)(ordered)
    widened(name, ordered)


PRIMITIVES = FloatPrimitives(
    when, functions.floor, functions.signum, lambda left, right: (left % right, None)
)


def float_floordiv(left: Expr, right: Expr) -> Lowered[Expr]:
    quotient, _, exact = float_divmod(PRIMITIVES, left, right)
    return Lowered(quotient, exact)


def float_mod(left: Expr, right: Expr) -> Lowered[Expr]:
    _, remainder, exact = float_divmod(PRIMITIVES, left, right)
    return Lowered(remainder, exact)


def int_divmod(left: Expr, right: Expr) -> tuple[Expr, Expr]:
    unsafe = (right == lit(0)) | ((left == lit(INT64_MIN)) & (right == lit(-1)))
    divisor = when(unsafe, lit(1), right)
    quotient, remainder = left / divisor, left % divisor
    shift = (remainder != lit(0)) & ((remainder < lit(0)) != (divisor < lit(0)))
    return when(shift, quotient - lit(1), quotient), when(shift, remainder + divisor, remainder)


TEXT = TextPrimitives(
    matches=lambda text, pattern: functions.regexp_like(text, lit(pattern)),
    strip=lambda text, chars: functions.btrim(text, lit(chars)),
    replace=lambda text, old, new: functions.replace(text, lit(old), lit(new)),
    byte_length=functions.octet_length,
    char_length=length,
    not_null=Expr.is_not_null,
    try_int=lambda text: text.try_cast(pa.int64()),
    try_float=lambda text: text.try_cast(pa.float64()),
)


def finite(value: Expr) -> Expr:
    return ~functions.isnan(value) & (functions.abs(value) != lit(INFINITY))


for _name, _operation in ARITHMETIC.items():
    RULES(_name, Kind.INT, Kind.INT)(checked(_operation))
    mixed(_name, _operation)

for _name, _compare in COMPARISONS.items():
    comparison(_name, _compare)

mixed("__truediv__", operator.truediv)
mixed("__floordiv__", float_floordiv)
mixed("__mod__", float_mod)


@RULES("__truediv__", Kind.INT, Kind.INT)
def int_truediv(left: Expr, right: Expr) -> Lowered[Expr]:
    return Lowered(floating(left) / floating(right), representable(left) & representable(right))


@RULES("__floordiv__", Kind.INT, Kind.INT)
def int_floordiv(left: Expr, right: Expr) -> Lowered[Expr]:
    return Lowered(int_divmod(left, right)[0], ~((left == lit(INT64_MIN)) & (right == lit(-1))))


@RULES("__mod__", Kind.INT, Kind.INT)
def int_mod(left: Expr, right: Expr) -> Expr:
    return int_divmod(left, right)[1]


@RULES("__eq__", Kind.INT, Kind.INT)
def int_eq(left: Expr, right: Expr) -> Expr:
    return not_distinct(left, right)


@RULES("__ne__", Kind.INT, Kind.INT)
def int_ne(left: Expr, right: Expr) -> Expr:
    return ~not_distinct(left, right)


def float_eq(left: Expr, right: Expr) -> Expr:
    return not_distinct(left, right) & ~is_nan(left) & ~is_nan(right)


RULES("__eq__", Kind.FLOAT, Kind.FLOAT)(float_eq)


@RULES("__ne__", Kind.FLOAT, Kind.FLOAT)
def float_ne(left: Expr, right: Expr) -> Expr:
    return ~float_eq(left, right)


widened("__eq__", float_eq)
widened("__ne__", float_ne)


@RULES("__neg__", Kind.INT)
def int_neg(value: Expr) -> Lowered[Expr]:
    return Lowered(lit(0) - value, value != lit(INT64_MIN))


@RULES("__neg__", Kind.FLOAT)
def float_neg(value: Expr) -> Expr:
    return value * lit(-1.0)


@RULES("abs", Kind.INT)
def int_abs(value: Expr) -> Lowered[Expr]:
    small = value != lit(INT64_MIN)
    return Lowered(functions.abs(when(small, value, lit(0))), small)


@RULES("abs", Kind.FLOAT)
def float_abs(value: Expr) -> Expr:
    return functions.abs(value)


@RULES("math.isnan", Kind.FLOAT)
def float_isnan(value: Expr) -> Expr:
    return functions.isnan(value)


@RULES("math.isinf", Kind.FLOAT)
def float_isinf(value: Expr) -> Expr:
    return functions.abs(value) == lit(INFINITY)


@RULES("int", Kind.FLOAT)
def float_int(value: Expr) -> Lowered[Expr]:
    whole = value.try_cast(pa.int64())
    return Lowered(whole, whole.is_not_null() | ~finite(value))


@RULES("float", Kind.INT)
def int_float(value: Expr) -> Expr:
    return floating(value)


@RULES("float", Kind.FLOAT)
def float_float(value: Expr) -> Expr:
    return value


def shown(text: Expr) -> Expr:
    return functions.coalesce(text, lit("None"))


@RULES("str", Kind.INT)
def int_str(value: Expr) -> Expr:
    return shown(value.cast(pa.string()))


@RULES("str", Kind.FLOAT)
def float_str(value: Expr) -> Lowered[Expr]:
    size = functions.abs(value)
    fixed = (size >= lit(REPR_FIXED_MIN)) & (size < lit(REPR_FIXED_MAX))
    exact = value.is_null() | (value == lit(0.0)) | fixed
    return Lowered(shown(value.cast(pa.string())), exact)


@RULES("str", Kind.BOOL)
def bool_str(value: Expr) -> Expr:
    return when(value.is_null(), lit("None"), when(value, lit("True"), lit("False")))


RULES("str", Kind.STR)(shown)


@RULES("int_parses", Kind.STR)
def str_int_parses(text: Expr) -> Lowered[Expr]:
    return Lowered(*int_parses(TEXT, text))


@RULES("int", Kind.STR)
def str_int(text: Expr) -> Lowered[Expr]:
    return Lowered(*parse_int(TEXT, text))


@RULES("float_parses", Kind.STR)
def str_float_parses(text: Expr) -> Lowered[Expr]:
    return Lowered(*float_parses(TEXT, text))


@RULES("float", Kind.STR)
def str_float(text: Expr) -> Lowered[Expr]:
    return Lowered(*parse_float(TEXT, text))


RULES("math.log10", Kind.FLOAT)(functions.log10)


@RULES("math.log10", Kind.INT)
def int_log10(value: Expr) -> Expr:
    return functions.log10(floating(value))
