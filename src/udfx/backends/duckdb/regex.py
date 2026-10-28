from __future__ import annotations

import duckdb
from duckdb import Expression

from udfx.backends.duckdb.rules import RULES, constant, function, when
from udfx.backends.registration import register_regex
from udfx.semantics.regex import RegexPrimitives


def findall(text: Expression, pattern: str, group: int) -> Expression:
    return function(
        "list_transform",
        function("regexp_extract_all", text, constant(pattern), constant(group)),
        duckdb.LambdaExpression(
            "value", duckdb.CoalesceOperator(duckdb.ColumnExpression("value"), constant(""))
        ),
    )


def extract(text: Expression, pattern: str, group: int) -> Expression:
    return when(
        function("regexp_matches", text, constant(pattern + r"[\s\S]*")),
        function("regexp_extract", text, constant(pattern), constant(group)),
        constant(None).cast("VARCHAR"),
    )


def first(text: Expression, pattern: str, group: int, nonempty: bool) -> Expression:
    return (
        function("list_extract", findall(text, pattern, group), constant(1))
        if nonempty
        else extract(text, pattern, group)
    )


register_regex(
    RULES,
    RegexPrimitives(
        dialect="re2",
        literal=lambda value: (
            constant(value).cast("VARCHAR[]") if isinstance(value, list) else constant(value)
        ),
        contains=lambda text, pattern: function("regexp_matches", text, constant(pattern)),
        extract=lambda text, pattern, empty: extract(text, pattern, 0),
        replace=lambda text, pattern, replacement, count: function(
            "regexp_replace",
            text,
            constant(pattern),
            constant(replacement),
            constant("" if count else "g"),
        ),
        split=lambda text, separator: function("string_split", text, constant(separator)),
        findall=findall,
        first=first,
        count_matches=lambda text, pattern: function(
            "len", function("regexp_extract_all", text, constant(pattern), constant(0))
        ),
        maximum_count=1,
    ),
)
