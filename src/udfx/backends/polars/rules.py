from __future__ import annotations

import polars as pl

from udfx.backends.base import RuleSet

RULES: RuleSet[pl.Expr] = RuleSet()


def broadcast(value: pl.Expr) -> pl.Expr:
    if not value.meta.is_literal():
        return value
    return value.gather(pl.int_range(pl.len()) * 0)
