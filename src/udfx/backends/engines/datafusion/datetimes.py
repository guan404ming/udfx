from __future__ import annotations

import pyarrow as pa
from datafusion import Expr, functions, lit

from udfx.backends.engines.datafusion.rules import RULES, null_safe_equal, when
from udfx.backends.registration.shared import (
    register_date_arithmetic,
    register_datetime_arithmetic,
    register_datetimes,
)
from udfx.semantics.temporal.dates import DateArithmeticPrimitives
from udfx.semantics.temporal.datetimes import (
    EPOCH_ORDINAL,
    TIMESTAMP_LITERAL,
    DatetimeArithmeticPrimitives,
    DatetimePrimitives,
)


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
        parse=lambda text: when(
            functions.regexp_like(text, lit(TIMESTAMP_LITERAL)),
            text.try_cast(pa.timestamp("us")),
            lit(None),
        ),
        to_integer=lambda value: value.try_cast(pa.int64()),
        select=when,
        is_null=Expr.is_null,
    ),
    null_safe_equal,
    lambda value: value.cast(pa.date32()),
)


register_datetime_arithmetic(
    RULES,
    DatetimeArithmeticPrimitives(
        literal=lit,
        difference=lambda left, right: left.cast(pa.int64()) - right.cast(pa.int64()),
        to_microseconds=lambda value: value.cast(pa.int64()),
        from_microseconds=lambda value: value.cast(pa.timestamp("us")),
        select=when,
    ),
)


register_date_arithmetic(
    RULES,
    DateArithmeticPrimitives(
        literal=lit,
        to_ordinal=lambda value: value.cast(pa.int64()) + lit(EPOCH_ORDINAL),
        from_ordinal=lambda value: (value - lit(EPOCH_ORDINAL)).cast(pa.int32()).cast(pa.date32()),
        select=when,
    ),
)
