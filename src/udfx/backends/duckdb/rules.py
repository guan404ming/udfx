from __future__ import annotations

from typing import Any, cast

import duckdb
from duckdb import Expression

from udfx.backends.base import RuleSet

RULES: RuleSet[Expression] = RuleSet()

column = duckdb.ColumnExpression
function = duckdb.FunctionExpression
NAN = duckdb.SQLExpression("'nan'::DOUBLE")


def constant(value: object) -> Expression:
    return duckdb.ConstantExpression(cast("Any", value))


NULL = constant(None)


def when(condition: Expression, then: Expression, orelse: Expression) -> Expression:
    return duckdb.CaseExpression(condition, then).otherwise(orelse)


def null_safe_equal(left: Expression, right: Expression) -> Expression:
    return duckdb.CoalesceOperator(left == right, left.isnull() & right.isnull())


def length(value: Expression) -> Expression:
    return function("len", value)
