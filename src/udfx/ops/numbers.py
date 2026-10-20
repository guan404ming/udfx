from __future__ import annotations

import math
import operator
from collections.abc import Callable

from udfx.ir import Node, lit
from udfx.ops.registry import Fault, Op, apply, register
from udfx.semantics.numbers import ARITHMETIC, COMPARISONS
from udfx.types import BOOL, FLOAT, INT, STR, Kind, Type

PAIRS: tuple[tuple[Type, Type], ...] = ((INT, INT), (FLOAT, FLOAT), (INT, FLOAT), (FLOAT, INT))


def zero(node: Node) -> Node:
    return lit(0 if node.type.kind is Kind.INT else 0.0)


def divisor_zero(_: Node, divisor: Node) -> tuple[Fault, ...]:
    return (Fault(ZeroDivisionError, apply("__eq__", divisor, zero(divisor))),)


def not_positive(value: Node) -> tuple[Fault, ...]:
    return (Fault(ValueError, apply("__le__", value, zero(value))),)


def not_finite(value: Node) -> tuple[Fault, ...]:
    return (
        Fault(ValueError, apply("math.isnan", value)),
        Fault(OverflowError, apply("math.isinf", value)),
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


for left, right in PAIRS:
    result = INT if (left, right) == (INT, INT) else FLOAT
    for name, operation in ARITHMETIC.items():
        register(Op(name, (left, right), result, operation))
    register(Op("__truediv__", (left, right), FLOAT, operator.truediv, divisor_zero))
    register(Op("__floordiv__", (left, right), result, operator.floordiv, divisor_zero))
    register(Op("__mod__", (left, right), result, operator.mod, divisor_zero))
    for name, compare in COMPARISONS.items():
        register(Op(name, (left, right), BOOL, compare))
    register(Op("__eq__", (left.or_none(), right.or_none()), BOOL, operator.eq))
    register(Op("__ne__", (left.or_none(), right.or_none()), BOOL, operator.ne))

for scalar in (INT, FLOAT):
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
