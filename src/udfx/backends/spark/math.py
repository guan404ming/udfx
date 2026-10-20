from __future__ import annotations

import operator

from pyspark.sql import Column, functions

from udfx.backends.base import Lowered
from udfx.backends.spark.numbers import cast_float64, either_nan, float_int, floor
from udfx.backends.spark.rules import RULES
from udfx.semantics.numbers import (
    EXACT_POWERS_OF_TEN,
    HALF_INTEGER_LIMIT,
    ROUND_OVERFLOW_LIMIT,
    Binary,
)
from udfx.types import Kind


def identity(value: Column) -> Column:
    return value


def power(base: Column, exponent: Column) -> Lowered[Column]:
    return Lowered(functions.pow(base, exponent), functions.lit(False))


def register_extremum(name: str, compare: Binary) -> None:
    def ordered(left: Column, right: Column) -> Column:
        return functions.when(compare(right, left) & ~either_nan(left, right), right).otherwise(
            left
        )

    def total(left: Column, right: Column) -> Column:
        return functions.when(compare(right, left), right).otherwise(left)

    RULES(name, Kind.FLOAT, Kind.FLOAT)(ordered)
    RULES(name, Kind.INT, Kind.INT)(total)
    RULES(name, Kind.STR, Kind.STR)(total)


register_extremum("max", operator.gt)
register_extremum("min", operator.lt)

RULES("__pow__", Kind.FLOAT, Kind.FLOAT)(power)
RULES("__pow__", Kind.INT, Kind.FLOAT)(lambda base, exponent: power(cast_float64(base), exponent))
RULES("__pow__", Kind.FLOAT, Kind.INT)(lambda base, exponent: power(base, cast_float64(exponent)))
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
    scale = functions.pow(functions.lit(10.0), cast_float64(digits))
    scaled = value * scale
    tie = scaled - floor(scaled) == 0.5
    exact = (
        digits.between(0, EXACT_POWERS_OF_TEN)
        & (functions.abs(scaled) < HALF_INTEGER_LIMIT)
        & ((digits == 0) | ~tie)
    )
    return Lowered(functions.try_divide(functions.rint(scaled), scale), exact)


@RULES("round_overflows", Kind.FLOAT, Kind.INT)
def round_overflows(value: Column, digits: Column) -> Lowered[Column]:
    safe = (digits >= 0) | (functions.abs(value) < ROUND_OVERFLOW_LIMIT)
    return Lowered(functions.lit(False), safe)
