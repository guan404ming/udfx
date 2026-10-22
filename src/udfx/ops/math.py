from __future__ import annotations

import math
import operator
from collections.abc import Callable
from functools import partial

from udfx.ir import Node, lit
from udfx.ops.numbers import not_finite, not_positive, zero
from udfx.ops.registry import Fault, Op, apply, register
from udfx.semantics.math import EXPONENTIAL_LIMIT
from udfx.semantics.scalars import numpy_scalar
from udfx.types import BOOL, FLOAT, INT, STR, Kind


def overflows(function: Callable[..., object]) -> Callable[..., bool]:
    def check(*arguments: object) -> bool:
        try:
            function(*arguments)
        except OverflowError:
            return True
        except ArithmeticError:
            return False
        return False

    return check


def real(node: Node) -> Node:
    return node if node.type.kind is Kind.FLOAT else apply("float", node)


def power_faults(base: Node, exponent: Node) -> tuple[Fault, ...]:
    base, exponent = real(base), real(exponent)
    finite = apply("not", apply("math.isinf", exponent))
    negative = apply("and", apply("__lt__", exponent, lit(0.0)), finite)
    return (
        Fault(ZeroDivisionError, apply("and", apply("__eq__", base, lit(0.0)), negative)),
        Fault(OverflowError, apply("pow_overflows", base, exponent)),
    )


def round_faults(value: Node, digits: Node) -> tuple[Fault, ...]:
    return (Fault(OverflowError, apply("round_overflows", value, digits)),)


def below_zero(value: Node) -> tuple[Fault, ...]:
    return (Fault(ValueError, apply("__lt__", value, zero(value))),)


def logarithm_faults(value: Node, base: Node) -> tuple[Fault, ...]:
    return (
        *not_positive(value),
        *not_positive(base),
        Fault(ZeroDivisionError, apply("__eq__", base, lit(1))),
    )


def exponential_faults(value: Node) -> tuple[Fault, ...]:
    value = real(value)
    overflow = apply("__gt__", value, lit(EXPONENTIAL_LIMIT))
    finite = apply("not", apply("math.isinf", value))
    return (Fault(OverflowError, apply("and", overflow, finite)),)


register(Op("pow_overflows", (FLOAT, FLOAT), BOOL, overflows(operator.pow)))
register(Op("round_overflows", (FLOAT, INT), BOOL, overflows(round)))

for left, right in ((FLOAT, FLOAT), (INT, FLOAT), (FLOAT, INT)):
    register(Op("__pow__", (left, right), FLOAT, operator.pow, power_faults))

for scalar in (INT, FLOAT, STR):
    register(Op("max", (scalar, scalar), scalar, max))
    register(Op("min", (scalar, scalar), scalar, min))

for scalar in (INT, FLOAT):
    register(Op("math.sqrt", (scalar,), FLOAT, math.sqrt, below_zero))

register(Op("round", (FLOAT,), INT, round, not_finite))
register(Op("round", (FLOAT, INT), FLOAT, round, round_faults))
register(Op("math.ceil", (FLOAT,), INT, math.ceil, not_finite))
register(Op("math.floor", (FLOAT,), INT, math.floor, not_finite))
register(Op("math.ceil", (INT,), INT, math.ceil))
register(Op("math.floor", (INT,), INT, math.floor))

for scalar in (INT, FLOAT, BOOL):
    register(Op("numpy.isnan", (scalar,), BOOL, partial(numpy_scalar, "isnan")))

for scalar in (INT, FLOAT):
    for name in ("log", "log2"):
        register(Op(f"math.{name}", (scalar,), FLOAT, getattr(math, name), not_positive))
    register(Op("math.exp", (scalar,), FLOAT, math.exp, exponential_faults))
    for base in (INT, FLOAT):
        register(Op("math.log", (scalar, base), FLOAT, math.log, logarithm_faults))

for name in ("log", "log1p", "log2", "log10", "sqrt", "exp", "expm1"):
    for scalar in (INT, FLOAT):
        register(Op(f"numpy.{name}", (scalar,), FLOAT, partial(numpy_scalar, name)))

for name in ("abs", "floor", "ceil", "round"):
    register(Op(f"numpy.{name}", (FLOAT,), FLOAT, partial(numpy_scalar, name)))

register(Op("numpy.round", (FLOAT, INT), FLOAT, partial(numpy_scalar, "round")))
