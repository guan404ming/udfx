from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, cast

import duckdb
from duckdb import Expression

from udfx.backends.registration import RuleSet
from udfx.semantics.text import strip_pattern

RULES: RuleSet[Expression] = RuleSet()
ITEM = "udfx item"

function = duckdb.FunctionExpression
NAN = duckdb.SQLExpression("'nan'::DOUBLE")


def column(name: str) -> Expression:
    return duckdb.SQLExpression('"' + name.replace('"', '""') + '"')


def constant(value: object) -> Expression:
    if isinstance(value, float) and math.isnan(value):
        return NAN
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


def trim(text: Expression, characters: str, method: str) -> Expression:
    return function(
        "regexp_replace",
        text,
        constant(strip_pattern(characters, method)),
        constant(""),
        constant("g"),
    )
