from __future__ import annotations

from duckdb import Expression

from udfx.backends.base import Lowered
from udfx.backends.duckdb.numbers import floating
from udfx.backends.duckdb.rules import RULES, constant, function
from udfx.types import Kind


def power(a: Expression, b: Expression) -> Lowered[Expression]:
    complex_result = (a < constant(0.0)) & (function("floor", b) != b)
    return Lowered(function("pow", a, b), ~complex_result)


RULES("__pow__", Kind.FLOAT, Kind.FLOAT)(power)
RULES("__pow__", Kind.INT, Kind.FLOAT)(lambda a, b: power(floating(a), b))
RULES("__pow__", Kind.FLOAT, Kind.INT)(lambda a, b: power(a, floating(b)))


@RULES("pow_overflows", Kind.FLOAT, Kind.FLOAT)
def pow_overflows(a: Expression, b: Expression) -> Lowered[Expression]:
    result = power(a, b)
    finite = function("isfinite", a) & function("isfinite", b) & (a != constant(0.0))
    return Lowered(function("isinf", result.expr) & finite, result.exact)
