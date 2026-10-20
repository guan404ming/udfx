from __future__ import annotations

from pyspark.sql import Column, functions

from udfx.backends.base import RuleSet

RULES: RuleSet[Column] = RuleSet()


def when(condition: Column, then: Column, orelse: Column) -> Column:
    return functions.when(condition, then).otherwise(orelse)
