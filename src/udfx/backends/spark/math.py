from __future__ import annotations

import operator

from pyspark.sql import Column, functions

from udfx.backends.base import Lowered
from udfx.backends.spark.numbers import either_nan, float_int, floor
from udfx.backends.spark.rules import RULES
from udfx.ops.math import EXACT_POWERS_OF_TEN, HALF_INTEGER_LIMIT, ROUND_OVERFLOW_LIMIT
from udfx.semantics.numbers import Binary
from udfx.types import Kind


def real(value: Column) -> Column:
    return value.cast("double")


def power(a: Column, b: Column) -> Lowered[Column]:
    return Lowered(functions.pow(a, b), functions.lit(False))


def pick(name: str, compare: Binary) -> None:
    def ordered(a: Column, b: Column) -> Column:
        return functions.when(compare(b, a) & ~either_nan(a, b), b).otherwise(a)

    def total(a: Column, b: Column) -> Column:
        return functions.when(compare(b, a), b).otherwise(a)

    RULES(name, Kind.FLOAT, Kind.FLOAT)(ordered)
    RULES(name, Kind.INT, Kind.INT)(total)
    RULES(name, Kind.STR, Kind.STR)(total)


pick("max", operator.gt)
pick("min", operator.lt)

RULES("__pow__", Kind.FLOAT, Kind.FLOAT)(power)
RULES("__pow__", Kind.INT, Kind.FLOAT)(lambda a, b: power(real(a), b))
RULES("__pow__", Kind.FLOAT, Kind.INT)(lambda a, b: power(a, real(b)))
RULES("round", Kind.FLOAT)(lambda a: float_int(functions.rint(a)))


@RULES("pow_overflows", Kind.FLOAT, Kind.FLOAT)
def pow_overflows(a: Column, b: Column) -> Lowered[Column]:
    return Lowered(functions.lit(False), functions.lit(False))


@RULES("round", Kind.FLOAT, Kind.INT)
def round_digits(a: Column, digits: Column) -> Lowered[Column]:
    scale = functions.pow(functions.lit(10.0), real(digits))
    scaled = a * scale
    tie = scaled - floor(scaled) == 0.5
    exact = (
        digits.between(0, EXACT_POWERS_OF_TEN)
        & (functions.abs(scaled) < HALF_INTEGER_LIMIT)
        & ((digits == 0) | ~tie)
    )
    return Lowered(functions.try_divide(functions.rint(scaled), scale), exact)


@RULES("round_overflows", Kind.FLOAT, Kind.INT)
def round_overflows(a: Column, digits: Column) -> Lowered[Column]:
    safe = (digits >= 0) | (functions.abs(a) < ROUND_OVERFLOW_LIMIT)
    return Lowered(functions.lit(False), safe)
