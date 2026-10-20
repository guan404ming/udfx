from __future__ import annotations

import math
import operator
from collections.abc import Callable

from udfx.ir import Node, lit
from udfx.ops.registry import Fault, Op, apply, register
from udfx.semantics.numbers import ARITHMETIC, COMPARISONS
from udfx.types import BOOL, FLOAT, INT, STR, Kind, Type

PAIRS: tuple[tuple[Type, Type], ...] = ((INT, INT), (FLOAT, FLOAT), (INT, FLOAT), (FLOAT, INT))

REPR_FIXED_MIN = 1e-4
REPR_FIXED_MAX = 1e16


def zero(node: Node) -> Node:
    return lit(0 if node.type.kind is Kind.INT else 0.0)


def divisor_zero(_: Node, b: Node) -> tuple[Fault, ...]:
    return (Fault(ZeroDivisionError, apply("__eq__", b, zero(b))),)


def not_positive(x: Node) -> tuple[Fault, ...]:
    return (Fault(ValueError, apply("__le__", x, zero(x))),)


def not_finite(x: Node) -> tuple[Fault, ...]:
    return (
        Fault(ValueError, apply("math.isnan", x)),
        Fault(OverflowError, apply("math.isinf", x)),
    )


def parses(convert: Callable[[str], object]) -> Callable[[str], bool]:
    def check(value: str) -> bool:
        try:
            convert(value)
        except ValueError:
            return False
        return True

    return check


def unparsable(name: str) -> Callable[[Node], tuple[Fault, ...]]:
    def faults(value: Node) -> tuple[Fault, ...]:
        return (Fault(ValueError, apply("not", apply(f"{name}_parses", value))),)

    return faults


for a, b in PAIRS:
    result = INT if (a, b) == (INT, INT) else FLOAT
    for name, operation in ARITHMETIC.items():
        register(Op(name, (a, b), result, operation))
    register(Op("__truediv__", (a, b), FLOAT, operator.truediv, divisor_zero))
    register(Op("__floordiv__", (a, b), result, operator.floordiv, divisor_zero))
    register(Op("__mod__", (a, b), result, operator.mod, divisor_zero))
    for name, compare in COMPARISONS.items():
        register(Op(name, (a, b), BOOL, compare))

for scalar in (INT, FLOAT):
    register(Op("__eq__", (scalar.or_none(), scalar.or_none()), BOOL, operator.eq))
    register(Op("__ne__", (scalar.or_none(), scalar.or_none()), BOOL, operator.ne))
    register(Op("__neg__", (scalar,), scalar, operator.neg))
    register(Op("abs", (scalar,), scalar, abs))
    register(Op("math.log10", (scalar,), FLOAT, math.log10, not_positive))

register(Op("int", (STR,), INT, int, unparsable("int")))
register(Op("int", (FLOAT,), INT, int, not_finite))
register(Op("float", (STR,), FLOAT, float, unparsable("float")))
register(Op("float", (INT,), FLOAT, float))
register(Op("float", (FLOAT,), FLOAT, float))
for scalar in (INT, FLOAT, BOOL, STR):
    register(Op("str", (scalar.or_none(),), STR, str))
register(Op("int_parses", (STR,), BOOL, parses(int)))
register(Op("float_parses", (STR,), BOOL, parses(float)))
register(Op("math.isnan", (FLOAT,), BOOL, math.isnan))
register(Op("math.isinf", (FLOAT,), BOOL, math.isinf))
