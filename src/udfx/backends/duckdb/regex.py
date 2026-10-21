from __future__ import annotations

import duckdb

from udfx.backends.duckdb.rules import RULES, constant, function
from udfx.backends.registration import register_regex
from udfx.semantics.regex import RegexPrimitives

register_regex(
    RULES,
    RegexPrimitives(
        dialect="re2",
        literal=lambda value: (
            constant(value).cast("VARCHAR[]") if isinstance(value, list) else constant(value)
        ),
        contains=lambda text, pattern: function("regexp_matches", text, constant(pattern)),
        extract=lambda text, pattern: function(
            "nullif", function("regexp_extract", text, constant(pattern), constant(0)), constant("")
        ),
        replace=lambda text, pattern, replacement, count: function(
            "regexp_replace",
            text,
            constant(pattern),
            constant(replacement),
            constant("" if count else "g"),
        ),
        split=lambda text, separator: function("string_split", text, constant(separator)),
        findall=lambda text, pattern, group: function(
            "list_transform",
            function("regexp_extract_all", text, constant(pattern), constant(group)),
            duckdb.LambdaExpression(
                "value", duckdb.CoalesceOperator(duckdb.ColumnExpression("value"), constant(""))
            ),
        ),
        maximum_count=1,
    ),
)
