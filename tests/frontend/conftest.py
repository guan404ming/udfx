import operator
from collections.abc import Iterator

import pytest

from udfx.ir import Node
from udfx.ops import Fault, Op, apply, register, resolve
from udfx.ops.registry import scoped
from udfx.types import BOOL, INT, STR, list_of


def empty(s: Node) -> tuple[Fault, ...]:
    return (Fault(ValueError, apply("not", apply("bool", s))),)


STAND_INS = (
    Op("int", (STR,), INT, int, empty),
    Op("__neg__", (INT,), INT, operator.neg),
    Op("__contains__", (STR, STR), BOOL, operator.contains),
    Op("__contains__", (list_of(STR), STR), BOOL, operator.contains),
    Op("__contains__", (list_of(INT), INT), BOOL, operator.contains),
    Op("__eq__", (STR, STR), BOOL, operator.eq),
    Op("__eq__", (INT, INT), BOOL, operator.eq),
    Op(".find", (STR, STR), INT, str.find),
    Op("__getitem__", (STR, INT), STR, operator.getitem),
    Op("slice", (STR, INT.or_none(), INT.or_none()), STR, lambda s, a, b: s[a:b]),
)


@pytest.fixture
def stand_ins() -> Iterator[None]:
    with scoped():
        for op in STAND_INS:
            if resolve(op.name, op.key.kinds) is None:
                register(op)
        yield
