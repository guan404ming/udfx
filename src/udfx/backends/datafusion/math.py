from __future__ import annotations

from datafusion import Expr, functions, lit

from udfx.backends.base import Lowered
from udfx.backends.datafusion.numbers import INFINITY, finite, floating, is_nan
from udfx.backends.datafusion.rules import RULES
from udfx.types import Kind


def below_zero(value: Expr) -> Expr:
    return (value < lit(0.0)) & ~is_nan(value)


def power(a: Expr, b: Expr) -> Lowered[Expr]:
    undefined = (a == lit(0.0)) & below_zero(b)
    base = functions.when(undefined, lit(1.0)).otherwise(a)
    value = functions.when(undefined, lit(INFINITY)).otherwise(functions.power(base, b))
    complex_result = below_zero(a) & (functions.floor(b) != b)
    return Lowered(value, ~complex_result)


RULES("__pow__", Kind.FLOAT, Kind.FLOAT)(power)
RULES("__pow__", Kind.INT, Kind.FLOAT)(lambda a, b: power(floating(a), b))
RULES("__pow__", Kind.FLOAT, Kind.INT)(lambda a, b: power(a, floating(b)))


@RULES("pow_overflows", Kind.FLOAT, Kind.FLOAT)
def pow_overflows(a: Expr, b: Expr) -> Lowered[Expr]:
    result = power(a, b)
    overflow = functions.abs(result.expr) == lit(INFINITY)
    return Lowered(overflow & finite(a) & finite(b) & (a != lit(0.0)), result.exact)
