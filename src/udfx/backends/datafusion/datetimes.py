from __future__ import annotations

import pyarrow as pa
from datafusion import Expr, functions, lit

from udfx.backends.datafusion.rules import RULES, null_safe_equal, when
from udfx.backends.registration import register_datetimes
from udfx.semantics.datetimes import DatetimePrimitives


def part(value: Expr, name: str) -> Expr:
    if name == "epoch_day":
        return value.cast(pa.date32()).cast(pa.int64())
    result = functions.date_part(name, value).cast(pa.int64())
    return result % lit(1000000) if name == "microsecond" else result


register_datetimes(
    RULES,
    DatetimePrimitives(
        literal=lit,
        part=part,
        pad=lambda value, width: functions.lpad(
            value.cast(pa.string()),
            functions.greatest(lit(width), functions.character_length(value.cast(pa.string()))),
            "0",
        ),
        concatenate=lambda parts: functions.concat(*parts),
        extract=lambda text, pattern, group: functions.array_element(
            functions.regexp_match(text, lit(pattern)), group
        ),
        parse=lambda text: text.try_cast(pa.timestamp("us")),
        to_integer=lambda value: value.try_cast(pa.int64()),
        select=when,
        is_null=Expr.is_null,
    ),
    null_safe_equal,
)
