from __future__ import annotations

import pyarrow as pa
from datafusion import Expr, functions, lit

from udfx.backends.arrow import dtype
from udfx.backends.base import RuleSet
from udfx.types import Type

RULES: RuleSet[Expr] = RuleSet()


def typed(value: object, type: Type) -> Expr:
    return lit(pa.scalar(value, type=dtype(type)))


def length(text: Expr) -> Expr:
    return functions.character_length(text).cast(pa.int64())


def not_distinct(a: Expr, b: Expr) -> Expr:
    return functions.coalesce(a == b, a.is_null() & b.is_null())


def when(condition: Expr, then: Expr, orelse: Expr) -> Expr:
    return functions.when(condition, then).otherwise(orelse)
