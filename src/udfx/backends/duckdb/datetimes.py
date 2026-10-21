from __future__ import annotations

from duckdb import Expression

from udfx.backends.duckdb.rules import RULES, constant, function, null_safe_equal, when
from udfx.backends.registration import register_datetimes
from udfx.semantics.datetimes import DatetimePrimitives


def part(value: Expression, name: str) -> Expression:
    if name == "epoch_day":
        return function("date_diff", constant("day"), constant("1970-01-01").cast("DATE"), value)
    result = function("date_part", constant(name), value)
    return result % constant(1000000) if name == "microsecond" else result


register_datetimes(
    RULES,
    DatetimePrimitives(
        literal=constant,
        part=part,
        pad=lambda value, width: function(
            "lpad",
            value.cast("VARCHAR"),
            function("greatest", constant(width), function("length", value.cast("VARCHAR"))).cast(
                "INTEGER"
            ),
            constant("0"),
        ),
        concatenate=lambda parts: function("concat", *parts),
        extract=lambda text, pattern, group: function(
            "regexp_extract", text, constant(pattern), constant(group)
        ),
        parse=lambda text: function("try_strptime", text, constant("%Y-%m-%d %H:%M:%S.%f")),
        to_integer=lambda value: function("nullif", value, constant("")).cast("BIGINT"),
        select=when,
        is_null=Expression.isnull,
    ),
    null_safe_equal,
)
