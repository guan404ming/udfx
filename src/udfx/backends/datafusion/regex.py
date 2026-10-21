from __future__ import annotations

import pyarrow as pa
from datafusion import Expr, functions, lit

from udfx.backends.datafusion.rules import RULES
from udfx.backends.registration import register_regex
from udfx.semantics.regex import SEPARATOR, RegexPrimitives


def literal(value: object) -> Expr:
    if isinstance(value, list):
        return lit(pa.scalar(value, type=pa.list_(pa.string())))
    return lit(value).cast(pa.string()) if value is None else lit(value)


def findall(text: Expr, pattern: str, group: int) -> Expr:
    replaced = functions.regexp_replace(
        text, lit(pattern), lit(f"{SEPARATOR}${{{group}}}{SEPARATOR}"), lit("g")
    )
    return functions.array_slice(
        functions.string_to_array(replaced, lit(SEPARATOR)), 2, 2**31 - 1, 2
    )


register_regex(
    RULES,
    RegexPrimitives(
        dialect="datafusion",
        literal=literal,
        contains=lambda text, pattern: functions.regexp_like(text, lit(pattern)),
        extract=lambda text, pattern: functions.array_element(
            functions.regexp_match(text, lit("(" + pattern + ")")), 1
        ),
        replace=lambda text, pattern, replacement, count: functions.regexp_replace(
            text, lit(pattern), lit(replacement), None if count else lit("g")
        ),
        split=lambda text, separator: functions.when(
            text == lit(""), functions.make_array(lit(""))
        ).otherwise(functions.string_to_array(text, lit(separator))),
        findall=findall,
        maximum_count=1,
    ),
)
