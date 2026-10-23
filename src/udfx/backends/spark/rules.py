from __future__ import annotations

from pyspark.sql import Column, functions

from udfx.backends.registration import RuleSet
from udfx.semantics.text_predicates import AsciiPrimitives

RULES: RuleSet[Column] = RuleSet()


def when(condition: Column, then: Column, orelse: Column) -> Column:
    return functions.when(condition, then).otherwise(orelse)


ASCII = AsciiPrimitives(
    is_ascii=lambda value: functions.octet_length(value) == functions.char_length(value),
    when=when,
)


def split_literal(text: Column, separator: Column) -> Column:
    escaped = functions.replace(separator, functions.lit("\\E"), functions.lit("\\E\\\\E\\Q"))
    pattern = functions.concat(functions.lit("\\Q"), escaped, functions.lit("\\E"))
    return functions.split(text, pattern, -1)
