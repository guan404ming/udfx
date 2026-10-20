from __future__ import annotations

import operator
import sys
from collections.abc import Callable

import pyarrow as pa
from datafusion import Expr, functions, lit

from udfx.backends.base import Lowered
from udfx.backends.datafusion.rules import RULES, length, not_distinct, when
from udfx.semantics.literals import ASCII_SPACE, FLOAT_LITERAL, INT_LITERAL
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


def representable(a: Expr) -> Expr:
    return a.between(lit(-FLOAT_INT_MAX), lit(FLOAT_INT_MAX))


def floating(a: Expr) -> Expr:
    return a.cast(pa.float64())


def checked(operation: Binary) -> Rule:
    def rule(a: Expr, b: Expr) -> Lowered[Expr]:
        wide = operation(a.cast(WIDE), b.cast(WIDE))
        return Lowered(operation(a, b), fits_int64(wide))

    return rule


def mixed(name: str, rule: Rule) -> None:
    RULES(name, Kind.FLOAT, Kind.FLOAT)(rule)
    RULES(name, Kind.INT, Kind.FLOAT)(lambda a, b: rule(floating(a), b))
    RULES(name, Kind.FLOAT, Kind.INT)(lambda a, b: rule(a, floating(b)))


def is_nan(a: Expr) -> Expr:
    return functions.coalesce(functions.isnan(a), lit(False))


def comparison(name: str, compare: Binary) -> None:
    def ordered(a: Expr, b: Expr) -> Expr:
        return compare(a, b) & ~is_nan(a) & ~is_nan(b)

    RULES(name, Kind.INT, Kind.INT)(compare)
    RULES(name, Kind.FLOAT, Kind.FLOAT)(ordered)
    RULES(name, Kind.INT, Kind.FLOAT)(
        lambda a, b: Lowered(ordered(floating(a), b), representable(a))
    )
    RULES(name, Kind.FLOAT, Kind.INT)(
        lambda a, b: Lowered(ordered(a, floating(b)), representable(b))
    )


PRIMITIVES = FloatPrimitives(
    when, functions.floor, functions.signum, lambda left, right: (left % right, None)
)


def float_floordiv(a: Expr, b: Expr) -> Lowered[Expr]:
    quotient, _, exact = float_divmod(PRIMITIVES, a, b)
    return Lowered(quotient, exact)


def float_mod(a: Expr, b: Expr) -> Lowered[Expr]:
    _, remainder, exact = float_divmod(PRIMITIVES, a, b)
    return Lowered(remainder, exact)


def int_divmod(a: Expr, b: Expr) -> tuple[Expr, Expr]:
    unsafe = (b == lit(0)) | ((a == lit(INT64_MIN)) & (b == lit(-1)))
    divisor = when(unsafe, lit(1), b)
    quotient, remainder = a / divisor, a % divisor
    shift = (remainder != lit(0)) & ((remainder < lit(0)) != (divisor < lit(0)))
    return when(shift, quotient - lit(1), quotient), when(shift, remainder + divisor, remainder)


def is_ascii(a: Expr) -> Expr:
    return functions.octet_length(a) == functions.character_length(a)


def cleaned(a: Expr) -> Expr:
    return functions.replace(functions.btrim(a, lit(ASCII_SPACE)), lit("_"), lit(""))


def finite(a: Expr) -> Expr:
    return ~functions.isnan(a) & (functions.abs(a) != lit(INFINITY))


for _name, _operation in ARITHMETIC.items():
    RULES(_name, Kind.INT, Kind.INT)(checked(_operation))
    mixed(_name, _operation)

for _name, _compare in COMPARISONS.items():
    comparison(_name, _compare)

mixed("__truediv__", operator.truediv)
mixed("__floordiv__", float_floordiv)
mixed("__mod__", float_mod)


@RULES("__truediv__", Kind.INT, Kind.INT)
def int_truediv(a: Expr, b: Expr) -> Lowered[Expr]:
    return Lowered(floating(a) / floating(b), representable(a) & representable(b))


@RULES("__floordiv__", Kind.INT, Kind.INT)
def int_floordiv(a: Expr, b: Expr) -> Lowered[Expr]:
    return Lowered(int_divmod(a, b)[0], ~((a == lit(INT64_MIN)) & (b == lit(-1))))


@RULES("__mod__", Kind.INT, Kind.INT)
def int_mod(a: Expr, b: Expr) -> Expr:
    return int_divmod(a, b)[1]


@RULES("__eq__", Kind.INT, Kind.INT)
def int_eq(a: Expr, b: Expr) -> Expr:
    return not_distinct(a, b)


@RULES("__ne__", Kind.INT, Kind.INT)
def int_ne(a: Expr, b: Expr) -> Expr:
    return ~not_distinct(a, b)


def float_eq(a: Expr, b: Expr) -> Expr:
    return not_distinct(a, b) & ~is_nan(a) & ~is_nan(b)


RULES("__eq__", Kind.FLOAT, Kind.FLOAT)(float_eq)


@RULES("__ne__", Kind.FLOAT, Kind.FLOAT)
def float_ne(a: Expr, b: Expr) -> Expr:
    return ~float_eq(a, b)


@RULES("__neg__", Kind.INT)
def int_neg(a: Expr) -> Lowered[Expr]:
    return Lowered(lit(0) - a, a != lit(INT64_MIN))


@RULES("__neg__", Kind.FLOAT)
def float_neg(a: Expr) -> Expr:
    return a * lit(-1.0)


@RULES("abs", Kind.INT)
def int_abs(a: Expr) -> Lowered[Expr]:
    small = a != lit(INT64_MIN)
    return Lowered(functions.abs(when(small, a, lit(0))), small)


@RULES("abs", Kind.FLOAT)
def float_abs(a: Expr) -> Expr:
    return functions.abs(a)


@RULES("math.isnan", Kind.FLOAT)
def float_isnan(a: Expr) -> Expr:
    return functions.isnan(a)


@RULES("math.isinf", Kind.FLOAT)
def float_isinf(a: Expr) -> Expr:
    return functions.abs(a) == lit(INFINITY)


@RULES("int", Kind.FLOAT)
def float_int(a: Expr) -> Lowered[Expr]:
    whole = a.try_cast(pa.int64())
    return Lowered(whole, whole.is_not_null() | ~finite(a))


@RULES("float", Kind.INT)
def int_float(a: Expr) -> Expr:
    return floating(a)


@RULES("float", Kind.FLOAT)
def float_float(value: Expr) -> Expr:
    return value


@RULES("str", Kind.INT)
def int_str(a: Expr) -> Expr:
    return a.cast(pa.string())


@RULES("int_parses", Kind.STR)
def str_int_parses(a: Expr) -> Lowered[Expr]:
    limit = sys.get_int_max_str_digits() or sys.maxsize
    return Lowered(
        functions.regexp_like(a, lit(INT_LITERAL)), is_ascii(a) & (length(a) <= lit(limit))
    )


@RULES("int", Kind.STR)
def str_int(a: Expr) -> Lowered[Expr]:
    parsed = cleaned(a).try_cast(pa.int64())
    literal = functions.regexp_like(a, lit(INT_LITERAL))
    return Lowered(parsed, is_ascii(a) & (parsed.is_not_null() | ~literal))


@RULES("float_parses", Kind.STR)
def str_float_parses(a: Expr) -> Lowered[Expr]:
    return Lowered(functions.regexp_like(a, lit(FLOAT_LITERAL)), is_ascii(a))


@RULES("float", Kind.STR)
def str_float(a: Expr) -> Lowered[Expr]:
    parsed = cleaned(a).try_cast(pa.float64())
    literal = functions.regexp_like(a, lit(FLOAT_LITERAL))
    return Lowered(parsed, is_ascii(a) & (parsed.is_not_null() | ~literal))


RULES("math.log10", Kind.FLOAT)(functions.log10)


@RULES("math.log10", Kind.INT)
def int_log10(a: Expr) -> Expr:
    return functions.log10(floating(a))
