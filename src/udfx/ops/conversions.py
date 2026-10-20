from __future__ import annotations

from collections.abc import Callable

from udfx.ir import Node
from udfx.ops.numbers import not_finite
from udfx.ops.registry import Fault, Op, apply, register
from udfx.types import BOOL, FLOAT, INT, STR


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


register(Op("int", (STR,), INT, int, unparsable("int")))
register(Op("int", (FLOAT,), INT, int, not_finite))
register(Op("float", (STR,), FLOAT, float, unparsable("float")))
register(Op("float", (INT,), FLOAT, float))
register(Op("float", (FLOAT,), FLOAT, float))
for scalar in (INT, FLOAT, BOOL, STR):
    register(Op("str", (scalar.or_none(),), STR, str))
register(Op("int_parses", (STR,), BOOL, parses(int)))
register(Op("float_parses", (STR,), BOOL, parses(float)))
