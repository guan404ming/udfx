from __future__ import annotations

import operator

from udfx.ops.registry import Op, register
from udfx.semantics.scalars import is_missing, not_missing
from udfx.types import BOOL, FLOAT, INT, STR, Type, list_of

NULLABLE_TYPES: tuple[Type, ...] = (
    STR.or_none(),
    INT.or_none(),
    FLOAT.or_none(),
    BOOL.or_none(),
    list_of(STR.or_none()).or_none(),
)


def is_none(value: object) -> bool:
    return value is None


register(Op("not", (BOOL,), BOOL, operator.not_))
register(Op("and", (BOOL, BOOL), BOOL, operator.and_))
register(Op("or", (BOOL, BOOL), BOOL, operator.or_))

for nullable in NULLABLE_TYPES:
    register(Op("is_none", (nullable,), BOOL, is_none))
    register(Op("bool", (nullable,), BOOL, bool))

for scalar in (STR, INT, FLOAT, BOOL):
    register(Op("pandas.isna", (scalar.or_none(),), BOOL, is_missing))
    register(Op("pandas.notna", (scalar.or_none(),), BOOL, not_missing))
