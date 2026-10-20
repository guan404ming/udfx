from __future__ import annotations

import polars as pl

from udfx.backends.base import RuleSet

RULES: RuleSet[pl.Expr] = RuleSet()
