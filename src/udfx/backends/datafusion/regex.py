from __future__ import annotations

import pyarrow as pa
from datafusion import Expr, functions, lit

from udfx.backends.datafusion.rules import RULES
from udfx.backends.registration import register_regex
from udfx.frontend.core.text import removed_suffix
from udfx.semantics.regex import SEPARATOR, RegexPrimitives
from udfx.types import Kind

RULES.rewrite("re.sub", Kind.STR, Kind.STR, Kind.INT, Kind.STR, Kind.INT)(removed_suffix)


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


def first(text: Expr, pattern: str, group: int, nonempty: bool) -> Expr:
    matched = functions.regexp_match(text, lit(pattern if group else "(" + pattern + ")"))
    if nonempty:
        return functions.array_element(matched, 1)
    value = functions.coalesce(functions.array_element(matched, 1), lit(""))
    return functions.when(matched.is_not_null(), value).otherwise(lit(None).cast(pa.string()))


register_regex(
    RULES,
    RegexPrimitives(
        dialect="datafusion",
        literal=literal,
        contains=lambda text, pattern: functions.regexp_like(text, lit(pattern)),
        extract=lambda text, pattern, empty: functions.array_element(
            functions.regexp_match(text, lit("(" + pattern + ")")), 1
        ),
        replace=lambda text, pattern, replacement, count: functions.regexp_replace(
            text, lit(pattern), lit(replacement), None if count else lit("g")
        ),
        split=lambda text, separator: functions.when(
            text == lit(""), functions.make_array(lit(""))
        ).otherwise(functions.string_to_array(text, lit(separator))),
        findall=findall,
        first=first,
        count_matches=lambda text, pattern: functions.regexp_count(text, lit(pattern)),
        maximum_count=1,
    ),
)
