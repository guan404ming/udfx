from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar

import pyarrow as pa
from datafusion import Expr, col, functions, lit

from udfx.backends.arrow import dtype
from udfx.backends.base import RESERVED
from udfx.backends.registration import RuleSet
from udfx.types import Kind, Type

RULES: RuleSet[Expr] = RuleSet()
LAMBDA_SCOPE: ContextVar[int] = ContextVar("udfx datafusion lambda", default=0)
ARRAY_SCOPING: ContextVar[bool] = ContextVar("udfx datafusion array scoping", default=True)


def array_lambda(transform: Callable[[Expr], Expr]) -> Expr:
    level = LAMBDA_SCOPE.get()
    name = f"{RESERVED}element {level}"
    token = LAMBDA_SCOPE.set(level + 1)
    try:
        body = transform(functions.lambda_var(name))
    finally:
        LAMBDA_SCOPE.reset(token)
    return functions.lambda_([name], body)


def array_result(value: Expr, result: pa.DataType | None = None) -> Expr:
    if result is not None:
        value = value.cast(pa.list_(result))
    if LAMBDA_SCOPE.get() or not ARRAY_SCOPING.get():
        return value
    nested = functions.array_transform(
        functions.make_array(lit(0)),
        array_lambda(lambda element: value),
    )
    return functions.array_element(nested, 1)


def array_transform(
    values: Expr, transform: Callable[[Expr], Expr], result: pa.DataType | None = None
) -> Expr:
    return array_result(functions.array_transform(values, array_lambda(transform)), result)


def array_filter(values: Expr, predicate: Callable[[Expr], Expr]) -> Expr:
    return array_result(functions.array_filter(values, array_lambda(predicate)))


def column(name: str) -> Expr:
    return col('"' + name.replace('"', '""') + '"')


def typed_literal(value: object, type: Type) -> Expr:
    native = pa.string_view() if type.kind in {Kind.STR, Kind.JSON} else dtype(type)
    if value is None:
        return lit(None).cast(native)
    return lit(pa.scalar(value, type=native))


def length(text: Expr) -> Expr:
    return functions.character_length(text).cast(pa.int64())


def null_safe_equal(left: Expr, right: Expr) -> Expr:
    return functions.coalesce(left == right, left.is_null() & right.is_null())


def when(condition: Expr, then: Expr, orelse: Expr) -> Expr:
    return functions.when(condition, then).otherwise(orelse)
