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


def nullable_equal(a: Expression, b: Expression) -> Expression:
    return duckdb.CoalesceOperator(a == b, a.isnull() & b.isnull())


def length(a: Expression) -> Expression:
    return function("len", a)
