from __future__ import annotations

from functools import partial

from pyspark.sql import Column, functions

from udfx.backends.registration import RuleSet
from udfx.semantics.facts import integer_fits
from udfx.semantics.text_predicates import AsciiPrimitives

RULES: RuleSet[Column] = RuleSet()
INT32_MAX = 2**31 - 1
INTEGER_FITS = partial(integer_fits, size_limit=INT32_MAX)


def when(condition: Column, then: Column, orelse: Column) -> Column:
    return functions.when(condition, then).otherwise(orelse)


ASCII = AsciiPrimitives(
    is_ascii=lambda value: functions.octet_length(value) == functions.char_length(value),
    when=when,
    literal=functions.lit,
)


def split_literal(text: Column, separator: Column, limit: Column | int = -1) -> Column:
    pattern = functions.regexp_replace(separator, r"([\\.^$|?*+()\[\]{}])", r"\\$1")
    return functions.when(~functions.contains(text, separator), functions.array(text)).otherwise(
        functions.split(text, pattern, limit)
    )
