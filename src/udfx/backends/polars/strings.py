from __future__ import annotations

import polars as pl

from udfx.backends.polars.rules import RULES
from udfx.types import Kind


@RULES(".lower", Kind.STR)
def lower(a: pl.Expr) -> pl.Expr:
    return a.str.to_lowercase()


@RULES(".upper", Kind.STR)
def upper(a: pl.Expr) -> pl.Expr:
    return a.str.to_uppercase()
