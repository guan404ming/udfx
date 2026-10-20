from __future__ import annotations

import operator

from udfx.ops.registry import Op, register
from udfx.types import BOOL, FLOAT, INT, STR, Type

NULLABLE_SCALARS: tuple[Type, ...] = (STR.or_none(), INT.or_none(), FLOAT.or_none(), BOOL.or_none())


def is_none(value: object) -> bool:
    return value is None


register(Op("not", (BOOL,), BOOL, operator.not_))
register(Op("and", (BOOL, BOOL), BOOL, operator.and_))
register(Op("or", (BOOL, BOOL), BOOL, operator.or_))

for scalar in NULLABLE_SCALARS:
    register(Op("is_none", (scalar,), BOOL, is_none))
    register(Op("bool", (scalar,), BOOL, bool))
