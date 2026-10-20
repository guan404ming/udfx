from __future__ import annotations

import polars as pl

from udfx.backends.base import RuleSet

RULES: RuleSet[pl.Expr] = RuleSet()


def broadcast(value: pl.Expr) -> pl.Expr:
    if not value.meta.is_literal():
        return value
    return value.gather(pl.int_range(pl.len()) * 0)


def when(condition: pl.Expr, then: pl.Expr, orelse: pl.Expr) -> pl.Expr:
    return pl.when(condition).then(then).otherwise(orelse)
