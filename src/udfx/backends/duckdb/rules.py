from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import duckdb
from duckdb import Expression

from udfx.backends.lowering_context import lowering_context
from udfx.backends.registration import RuleSet
from udfx.semantics.text import strip_pattern
from udfx.semantics.text_predicates import AsciiPrimitives

RULES: RuleSet[Expression] = RuleSet()
ITEM = "udfx item"

NAN = duckdb.SQLExpression("'nan'::DOUBLE")


def function(name: str, *arguments: Expression) -> Expression:
    if name in {"list_transform", "list_filter"}:
        with lowering_context() as context:
            context.has_bindings = True
    return duckdb.FunctionExpression(name, *arguments)


def column(name: str) -> Expression:
    return duckdb.SQLExpression('"' + name.replace('"', '""') + '"')


def constant(value: object) -> Expression:
    if isinstance(value, float):
        return duckdb.ConstantExpression(repr(value)).cast("DOUBLE")
    if isinstance(value, str) and "\x00" in value:
        return function("decode", constant(value.encode("utf-8")))
    return duckdb.ConstantExpression(cast("Any", value))


def bind(
    value: Expression, transform: Callable[[Expression], Expression], name: str = ITEM
) -> Expression:
    return function(
        "list_extract",
        function(
            "list_transform",
            function("list_value", value),
            duckdb.LambdaExpression(name, transform(column(name))),
        ),
        constant(1),
    )


NULL = constant(None)


def when(condition: Expression, then: Expression, orelse: Expression) -> Expression:
    return duckdb.CaseExpression(condition, then).otherwise(orelse)


def null_safe_equal(left: Expression, right: Expression) -> Expression:
    return duckdb.CoalesceOperator(left == right, left.isnull() & right.isnull())


def length(value: Expression) -> Expression:
    return function("len", value)


def concatenate(items: Expression) -> Expression:
    return duckdb.CoalesceOperator(function("array_to_string", items, constant("")), constant(""))


def trim(text: Expression, characters: str, method: str) -> Expression:
    return function(
        "regexp_replace",
        text,
        constant(strip_pattern(characters, method)),
        constant(""),
        constant("g"),
    )


ASCII = AsciiPrimitives(
    is_ascii=lambda value: function("strlen", value) == length(value),
    when=when,
    literal=constant,
)
