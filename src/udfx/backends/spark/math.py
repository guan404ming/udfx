from __future__ import annotations

from pyspark.sql import Column, functions

from udfx.backends.base import Lowered
from udfx.backends.spark.rules import RULES
from udfx.types import Kind


def real(value: Column) -> Column:
    return value.cast("double")


def power(a: Column, b: Column) -> Lowered[Column]:
    return Lowered(functions.pow(a, b), functions.lit(False))


RULES("__pow__", Kind.FLOAT, Kind.FLOAT)(power)
RULES("__pow__", Kind.INT, Kind.FLOAT)(lambda a, b: power(real(a), b))
RULES("__pow__", Kind.FLOAT, Kind.INT)(lambda a, b: power(a, real(b)))


@RULES("pow_overflows", Kind.FLOAT, Kind.FLOAT)
def pow_overflows(a: Column, b: Column) -> Lowered[Column]:
    return Lowered(functions.lit(False), functions.lit(False))
