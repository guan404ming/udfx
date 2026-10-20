from __future__ import annotations

import operator
from collections.abc import Callable

from udfx.ir import Node, lit
from udfx.ops.numbers import not_finite
from udfx.ops.registry import Fault, Op, apply, register
from udfx.types import BOOL, FLOAT, INT, STR, Kind

EXACT_POWERS_OF_TEN = 22
HALF_INTEGER_LIMIT = 2.0**52
ROUND_OVERFLOW_LIMIT = 1e308


def overflows(function: Callable[..., object]) -> Callable[..., bool]:
    def check(*args: object) -> bool:
        try:
            function(*args)
        except OverflowError:
            return True
        except ArithmeticError:
            return False
        return False

    return check


def real(node: Node) -> Node:
    return node if node.type.kind is Kind.FLOAT else apply("float", node)


def power_faults(a: Node, b: Node) -> tuple[Fault, ...]:
    base, exponent = real(a), real(b)
    finite = apply("not", apply("math.isinf", exponent))
    negative = apply("and", apply("__lt__", exponent, lit(0.0)), finite)
    return (
        Fault(ZeroDivisionError, apply("and", apply("__eq__", base, lit(0.0)), negative)),
        Fault(OverflowError, apply("pow_overflows", base, exponent)),
    )


def round_faults(value: Node, digits: Node) -> tuple[Fault, ...]:
    return (Fault(OverflowError, apply("round_overflows", value, digits)),)


register(Op("pow_overflows", (FLOAT, FLOAT), BOOL, overflows(operator.pow)))
register(Op("round_overflows", (FLOAT, INT), BOOL, overflows(round)))

for a, b in ((FLOAT, FLOAT), (INT, FLOAT), (FLOAT, INT)):
    register(Op("__pow__", (a, b), FLOAT, operator.pow, power_faults))

for scalar in (INT, FLOAT, STR):
    register(Op("max", (scalar, scalar), scalar, max))
    register(Op("min", (scalar, scalar), scalar, min))

register(Op("round", (FLOAT,), INT, round, not_finite))
register(Op("round", (FLOAT, INT), FLOAT, round, round_faults))
