from __future__ import annotations

from collections.abc import Callable

import pyarrow as pa
from pyspark.sql import Column, functions

from udfx.backends.engines.spark.conversions import float_int
from udfx.backends.engines.spark.numbers import ORDER, cast_float64, floor
from udfx.backends.engines.spark.rules import PYTHON_UDFS, RULES, when
from udfx.backends.lowering.context import graph_enabled
from udfx.backends.registration.shared import (
    Lowered,
    register_extremum,
    register_float_variants,
    register_math,
    register_trigonometry,
)
from udfx.semantics.numbers.math import POWER_FINITE_BOUND, power_hazards
from udfx.semantics.numbers.operations import RoundPrimitives, round_overflow_safe, scaled_round
from udfx.semantics.scalars import numpy_values, python_power
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


def python_column(key: str, run: Callable[..., pa.Array], *values: Column) -> Column | None:
    used = PYTHON_UDFS.get()
    if used is None or graph_enabled():
        return None
    if key not in used:
        used[key] = functions.arrow_udf(run, "double")(*values)  # pyright: ignore[reportCallIssue, reportArgumentType]
    return used[key]


def python_power_column(base: Column, exponent: Column | float) -> Column | None:
    key = f"power {base} {exponent!r}"
    if isinstance(exponent, Column):

        def power_batch(bases: pa.Array, exponents: pa.Array) -> pa.Array:
            values = map(python_power, bases.to_pylist(), exponents.to_pylist())
            return pa.array(list(values), pa.float64())

        return python_column(key, power_batch, base, exponent)

    def power_literal(bases: pa.Array) -> pa.Array:
        return pa.array(
            [python_power(value, exponent) for value in bases.to_pylist()], pa.float64()
        )

    return python_column(key, power_literal, base)


def power(base: Column, exponent: Column | float) -> Lowered[Column]:
    value = python_power_column(base, exponent)
    if value is None:
        return Lowered(functions.pow(base, exponent), False)
    if not power_hazards(exponent)[1]:
        return Lowered(value)
    if not isinstance(exponent, Column):
        return Lowered(value, base >= 0.0)
    return Lowered(value, ~((base < 0.0) & (floor(exponent) != exponent)))


register_extremum(RULES, "max", functions.greatest, ORDER, when)
register_extremum(RULES, "min", functions.least, ORDER, when)

register_float_variants(RULES, "__pow__", power, cast_float64, literals=(1,))
RULES("math.sqrt", Kind.FLOAT)(functions.sqrt)
RULES("math.sqrt", Kind.INT)(lambda value: functions.sqrt(cast_float64(value)))
RULES("math.ceil", Kind.FLOAT)(lambda value: float_int(-floor(-value)))
RULES("math.floor", Kind.FLOAT)(lambda value: float_int(floor(value)))
RULES("math.ceil", Kind.INT)(identity)
RULES("math.floor", Kind.INT)(identity)
RULES("round", Kind.FLOAT)(lambda value: float_int(functions.rint(value)))


@RULES("pow_overflows", Kind.FLOAT, Kind.FLOAT, literals=(1,))
def pow_overflows(base: Column, exponent: Column | float) -> Lowered[Column]:
    value = python_power_column(base, exponent)
    if value is None:
        native = functions.abs(functions.pow(base, exponent))
        return Lowered(functions.lit(False), native < functions.lit(POWER_FINITE_BOUND))
    known = base.isNotNull() & functions.lit(exponent).isNotNull()
    return Lowered(value.isNull() & known)


@RULES("round", Kind.FLOAT, Kind.INT, literals=(1,))
def round_digits(value: Column, digits: Column | int) -> Lowered[Column]:
    rounding = scaled_round(ROUND, value, digits)
    rounded = functions.try_divide(functions.rint(rounding.scaled), rounding.scale)
    return Lowered(rounded, rounding.exact)


@RULES("round_overflows", Kind.FLOAT, Kind.INT, literals=(1,))
def round_overflows(value: Column, digits: Column | int) -> Lowered[Column]:
    return Lowered(functions.lit(False), round_overflow_safe(ROUND, value, digits))


def numpy_column(name: str, value: Column) -> Column | None:
    def numpy_batch(values: pa.Array) -> pa.Array:
        return pa.array(numpy_values(name, values.to_numpy(zero_copy_only=False)), pa.float64())

    return python_column(f"{name} {value}", numpy_batch, value)


register_math(
    RULES,
    ROUND,
    root=functions.sqrt,
    floor=floor,
    ceil=lambda value: -floor(-value),
    is_nan=functions.isnan,
    round_digits=round_digits,
    numpy_values=numpy_column,
)


register_trigonometry(
    RULES,
    ROUND,
    unary={"sin": functions.sin, "cos": functions.cos, "acos": functions.acos},
    atan2=functions.atan2,
    is_nan=functions.isnan,
    guarded=True,
)
