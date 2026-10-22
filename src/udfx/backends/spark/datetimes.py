from __future__ import annotations

from pyspark.sql import Column, functions

from udfx.backends.registration import register_datetime_arithmetic, register_datetimes
from udfx.backends.spark.rules import RULES
from udfx.semantics.datetimes import DatetimeArithmeticPrimitives, DatetimePrimitives


def part(value: Column, name: str) -> Column:
    if name == "epoch_day":
        return functions.datediff(
            value.cast("date"), functions.lit("1970-01-01").cast("date")
        ).cast("long")
    if name == "doy":
        return functions.dayofyear(value).cast("long")
    if name == "microsecond":
        return (functions.date_part(functions.lit("SECOND"), value) * 1000000).cast(
            "long"
        ) % 1000000
    return functions.date_part(functions.lit(name), value).cast("long")


register_datetimes(
    RULES,
    DatetimePrimitives(
        literal=functions.lit,
        part=part,
        pad=lambda value, width: functions.lpad(
            value.cast("string"),
            functions.greatest(functions.lit(width), functions.length(value.cast("string"))),
            functions.lit("0"),
        ),
        concatenate=lambda parts: functions.concat(*parts),
        extract=functions.regexp_extract,
        parse=lambda text: text.try_cast("timestamp_ntz"),
        to_integer=lambda value: value.try_cast("long"),
        select=lambda condition, then, orelse: functions.when(condition, then).otherwise(orelse),
        is_null=Column.isNull,
    ),
    Column.eqNullSafe,
)


register_datetime_arithmetic(
    RULES,
    DatetimeArithmeticPrimitives(
        literal=functions.lit,
        to_microseconds=lambda value: functions.timestamp_diff(
            "MICROSECOND", functions.lit("1970-01-01").cast("timestamp_ntz"), value
        ),
        from_microseconds=lambda value: functions.timestamp_add(
            "MICROSECOND", value, functions.lit("1970-01-01").cast("timestamp_ntz")
        ),
        select=lambda condition, then, orelse: functions.when(condition, then).otherwise(orelse),
    ),
)
