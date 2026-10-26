from __future__ import annotations

import polars as pl

from udfx.backends.registration import RuleSet
from udfx.errors import Unsupported

RULES: RuleSet[pl.Expr] = RuleSet()


def when(condition: pl.Expr, then: pl.Expr, orelse: pl.Expr) -> pl.Expr:
    return pl.when(condition).then(then).otherwise(orelse)


def broadcast(value: pl.Expr) -> pl.Expr:
    try:
        if value.meta.is_length_preserving():
            return value
    except (pl.exceptions.ColumnNotFoundError, pl.exceptions.InvalidOperationError):
        pass
    return when(pl.int_range(pl.len()) % 2 == 0, value, value)


def evaluate_list(source: pl.Expr, body: pl.Expr) -> pl.Expr:
    if any(body.meta.root_names()):
        raise Unsupported("Polars list expressions cannot capture columns")
    return source.list.eval(body)


def literal_value(value: pl.Expr) -> object | None:
    return pl.select(value).item() if value.meta.is_literal() else None
