from __future__ import annotations

import pyarrow as pa
from datafusion import Expr, col, functions, lit

from udfx.backends.arrow import dtype
from udfx.backends.base import RuleSet
from udfx.types import Type

RULES: RuleSet[Expr] = RuleSet()


def column(name: str) -> Expr:
    return col('"' + name.replace('"', '""') + '"')


def typed_literal(value: object, type: Type) -> Expr:
    if value is None:
        return lit(None).cast(dtype(type))
    return lit(pa.scalar(value, type=dtype(type)))


def length(text: Expr) -> Expr:
    return functions.character_length(text).cast(pa.int64())


def null_safe_equal(left: Expr, right: Expr) -> Expr:
    return functions.coalesce(left == right, left.is_null() & right.is_null())


def when(condition: Expr, then: Expr, orelse: Expr) -> Expr:
    return functions.when(condition, then).otherwise(orelse)
