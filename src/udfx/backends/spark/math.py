from __future__ import annotations

from pyspark.sql import Column, functions

from udfx.backends.base import Lowered, register_extremum, register_float_variants
from udfx.backends.spark.conversions import float_int
from udfx.backends.spark.numbers import ORDER, cast_float64, floor
from udfx.backends.spark.rules import RULES, when
from udfx.semantics.numbers import RoundPrimitives, round_overflow_safe, scaled_round
from udfx.types import Kind


def identity(value: Column) -> Column:
    return value


ROUND = RoundPrimitives(
    functions.lit,
    functions.pow,
    Column.between,
    functions.abs,
    cast_float64,
    functions.rint,
    when,
)


def power(base: Column, exponent: Column) -> Lowered[Column]:
    return Lowered(functions.pow(base, exponent), functions.lit(False))


register_extremum(RULES, "max", functions.greatest, ORDER)
register_extremum(RULES, "min", functions.least, ORDER)

register_float_variants(RULES, "__pow__", power, cast_float64)
RULES("math.sqrt", Kind.FLOAT)(functions.sqrt)
RULES("math.sqrt", Kind.INT)(lambda value: functions.sqrt(cast_float64(value)))
RULES("math.ceil", Kind.FLOAT)(lambda value: float_int(-floor(-value)))
RULES("math.floor", Kind.FLOAT)(lambda value: float_int(floor(value)))
RULES("math.ceil", Kind.INT)(identity)
RULES("math.floor", Kind.INT)(identity)
RULES("round", Kind.FLOAT)(lambda value: float_int(functions.rint(value)))


@RULES("pow_overflows", Kind.FLOAT, Kind.FLOAT)
def pow_overflows(base: Column, exponent: Column) -> Lowered[Column]:
    return Lowered(functions.lit(False), functions.lit(False))


@RULES("round", Kind.FLOAT, Kind.INT)
def round_digits(value: Column, digits: Column) -> Lowered[Column]:
    rounding = scaled_round(ROUND, value, digits)
    rounded = functions.try_divide(functions.rint(rounding.scaled), rounding.scale)
    return Lowered(rounded, rounding.exact)


@RULES("round_overflows", Kind.FLOAT, Kind.INT)
def round_overflows(value: Column, digits: Column) -> Lowered[Column]:
    return Lowered(functions.lit(False), round_overflow_safe(ROUND, value, digits))
