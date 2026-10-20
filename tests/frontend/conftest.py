import operator
from collections.abc import Iterator

import pytest

from udfx.ops import Op, register, resolve
from udfx.ops.registry import scoped
from udfx.types import BOOL, INT, list_of

STAND_INS = (Op("__contains__", (list_of(INT), INT), BOOL, operator.contains),)


@pytest.fixture
def stand_ins() -> Iterator[None]:
    with scoped():
        for op in STAND_INS:
            if resolve(op.name, op.key.kinds) is None:
                register(op)
        yield
