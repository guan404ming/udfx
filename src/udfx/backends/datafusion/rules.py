from __future__ import annotations

from collections.abc import Callable
from itertools import count

import pyarrow as pa
from datafusion import Expr, col, functions, lit

from udfx.backends.arrow import dtype
from udfx.backends.base import RESERVED
from udfx.backends.registration import RuleSet
from udfx.types import Type

RULES: RuleSet[Expr] = RuleSet()
LAMBDAS = count()


def array_lambda(transform: Callable[[Expr], Expr]) -> Expr:
    name = f"{RESERVED}element {next(LAMBDAS)}"
    return functions.lambda_([name], transform(functions.lambda_var(name)))


def array_result(value: Expr, result: pa.DataType | None = None) -> Expr:
    nested = functions.array_transform(
        functions.make_array(lit(0)),
        array_lambda(lambda element: value if result is None else value.cast(pa.list_(result))),
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
    if value is None:
        return lit(None).cast(dtype(type))
    return lit(pa.scalar(value, type=dtype(type)))


def length(text: Expr) -> Expr:
    return functions.character_length(text).cast(pa.int64())


def null_safe_equal(left: Expr, right: Expr) -> Expr:
    return functions.coalesce(left == right, left.is_null() & right.is_null())


def when(condition: Expr, then: Expr, orelse: Expr) -> Expr:
    return functions.when(condition, then).otherwise(orelse)
