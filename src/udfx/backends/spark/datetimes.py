from __future__ import annotations

from pyspark.sql import Column, functions

from udfx.backends.registration import register_datetime_arithmetic, register_datetimes
from udfx.backends.spark.rules import RULES, when
from udfx.semantics.calendar import CalendarPrimitives
from udfx.semantics.datetimes import (
    MICROSECONDS_PER_SECOND,
    TIMESTAMP_LITERAL,
    DatetimeArithmeticPrimitives,
    DatetimePrimitives,
)
from udfx.semantics.literals import INT_LITERAL


def interval_microseconds(value: Column) -> Column:
    seconds = value.cast("long") * MICROSECONDS_PER_SECOND
    fraction = (functions.date_part(functions.lit("SECOND"), value) * MICROSECONDS_PER_SECOND).cast(
        "long"
    ) % MICROSECONDS_PER_SECOND
    return seconds + fraction


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
        parse=lambda text: functions.when(
            text.rlike(TIMESTAMP_LITERAL), text.try_cast("timestamp_ntz")
        ),
        to_integer=lambda value: functions.when(value.rlike(INT_LITERAL), value.try_cast("long")),
        select=when,
        is_null=Column.isNull,
        calendar=CalendarPrimitives(
            literal=functions.lit,
            select=when,
            matches=Column.rlike,
            translate=functions.translate,
            replace=lambda value, original, replacement: functions.replace(
                value, functions.lit(original), functions.lit(replacement)
            ),
            parse=lambda value: value.try_cast("date").cast("timestamp_ntz"),
            year=functions.year,
        ),
    ),
    Column.eqNullSafe,
)


register_datetime_arithmetic(
    RULES,
    DatetimeArithmeticPrimitives(
        literal=functions.lit,
        difference=lambda left, right: interval_microseconds(left - right),
        to_microseconds=lambda value: interval_microseconds(
            value - functions.lit("1970-01-01").cast("timestamp_ntz")
        ),
        from_microseconds=lambda value: (
            functions.lit("1970-01-01").cast("timestamp_ntz")
            + functions.make_dt_interval(secs=value.cast("decimal(20,0)") / MICROSECONDS_PER_SECOND)
        ),
        select=lambda condition, then, orelse: functions.when(condition, then).otherwise(orelse),
    ),
)
