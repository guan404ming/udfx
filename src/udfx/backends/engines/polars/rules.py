from __future__ import annotations

import polars as pl

from udfx.backends.registration.shared import RuleSet
from udfx.errors import Unsupported

RULES: RuleSet[pl.Expr] = RuleSet()


def when(condition: pl.Expr, then: pl.Expr, orelse: pl.Expr) -> pl.Expr:
    return pl.when(condition).then(then).otherwise(orelse)


def broadcast(value: pl.Expr) -> pl.Expr:
    try:
        if value.meta.is_length_preserving():
            return value
    except (pl.exceptions.ColumnNotFoundError, pl.exceptions.InvalidOperationError):
        if value.meta.root_names():
            return value
    return pl.struct(value.alias("0"), pl.int_range(pl.len()).alias("1")).struct.field("0")


def evaluate_list(source: pl.Expr, body: pl.Expr) -> pl.Expr:
    if any(body.meta.root_names()):
        raise Unsupported("Polars list expressions cannot capture columns")
    return source.list.eval(body)


def literal_value(value: pl.Expr) -> object | None:
    return pl.select(value).item() if value.meta.is_literal() else None
