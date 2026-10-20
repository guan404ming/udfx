from __future__ import annotations

import polars as pl

from udfx.backends.base import RuleSet

RULES: RuleSet[pl.Expr] = RuleSet()


def when(condition: pl.Expr, then: pl.Expr, orelse: pl.Expr) -> pl.Expr:
    return pl.when(condition).then(then).otherwise(orelse)


def broadcast(value: pl.Expr) -> pl.Expr:
    return when(pl.int_range(pl.len()) % 2 == 0, value, value)


def literal_value(value: pl.Expr) -> object | None:
    return pl.select(value).item() if value.meta.is_literal() else None
