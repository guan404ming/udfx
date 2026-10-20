from __future__ import annotations

from pyspark.sql import Column

from udfx.backends.base import RuleSet

RULES: RuleSet[Column] = RuleSet()
