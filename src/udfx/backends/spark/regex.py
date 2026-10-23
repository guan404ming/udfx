from __future__ import annotations

from pyspark.sql import Column, functions

from udfx.backends.registration import register_regex
from udfx.backends.spark.rules import ASCII, RULES
from udfx.semantics.regex import RegexPrimitives


def literal(value: object) -> Column | bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, list):
        return functions.array().cast("array<string>")
    return functions.lit(value).cast("string") if value is None else functions.lit(value)


def extract(text: Column, pattern: str) -> Column:
    return functions.when(
        functions.regexp_like(text, functions.lit(pattern)),
        functions.regexp_extract(text, pattern, 0),
    ).otherwise(functions.lit(None).cast("string"))


register_regex(
    RULES,
    RegexPrimitives(
        dialect="java",
        literal=literal,
        contains=lambda text, pattern: functions.regexp_like(text, functions.lit(pattern)),
        extract=extract,
        replace=lambda text, pattern, replacement, count: functions.regexp_replace(
            text, pattern, replacement
        ),
        split=lambda text, separator: functions.split(text, separator, -1),
        findall=lambda text, pattern, group: functions.regexp_extract_all(
            text, functions.lit(pattern), group
        ),
        maximum_count=0,
    ),
    ASCII,
)
